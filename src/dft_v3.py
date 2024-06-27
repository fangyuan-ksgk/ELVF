from .lcdpo import *
import torch
import numpy as np
from torch import nn
from transformers import PreTrainedModel

def convert_to_tensor_with_pad(example_list, pad_value=-100):
    tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(v) for v in example_list], batch_first=True, padding_value=pad_value)
    return tensor

def convert_batch(batch, ignore_index=-100, pad_token_id = 0):
    for (key, item) in batch.items():
        if "mask" in key:
            batch[key] = convert_to_tensor_with_pad(item, 0)
        elif "labels" in key:
            batch[key] = convert_to_tensor_with_pad(item, ignore_index)
        else:
            batch[key] = convert_to_tensor_with_pad(item, pad_token_id)
    return batch


def get_completion_only_labels(tokenizer, response_template, input_ids: list[list[int]]) -> list[list[int]]:
    # This should be correct since the initialization went through (unless some hidden error appears)
    labels = torch.tensor(input_ids).clone()
    response_token_ids_end_idx = None

    # Find location on string level
    format_prompt = tokenizer.decode(input_ids)
    idx = format_prompt.find(response_template)
    if idx != -1:
        prefix = format_prompt[:idx + len(response_template)]
        suffix = format_prompt[idx + len(response_template):]
        # Backward propagate to token level | Want the model to predict the next token for us
        prefix_tokens = tokenizer.tokenize(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.tokenize(suffix, add_special_tokens=False)
        diff = len(input_ids) - len(prefix_tokens) - len(suffix_tokens)
        response_token_ids_end_idx = len(prefix_tokens) + diff

    if response_token_ids_end_idx is None:
        print("Issuing Input Id Type: ", type(input_ids))
        print("Format Prompt: ", format_prompt)
        labels[:] = -100
    else:
        # Make pytorch loss function ignore all tokens up through the end of the response key
        labels[:response_token_ids_end_idx] = -100
    return labels.tolist()


def convert_feature(feature, tokenizer):
    teacher_format_prompt = tokenizer.apply_chat_template([{"role": "user", "content": feature["teacher_prompt"]},
                                    {"role": "assistant", "content": feature["chosen"]}], tokenize=False)
    student_format_prompt = tokenizer.apply_chat_template([{"role": "user", "content": feature["prompt"]},
                                    {"role": "assistant", "content": feature["chosen"]}], tokenize=False)
    feature["teacher_format_prompt"] = teacher_format_prompt
    feature["student_format_prompt"] = student_format_prompt
    return feature


def compute_self_distillation_loss(
        teacher_labels,
        teacher_logits,
        student_labels,
        student_logits,
        ignore_index = -100,
        avg_over_sequence = True
):
    # Student & Teacher 
    slice_teacher_logits = teacher_logits[torch.where(teacher_labels != ignore_index)]
    slice_student_logits = student_logits[torch.where(student_labels != ignore_index)]

    min_seq_length = min(len(slice_teacher_logits), len(slice_student_logits))
    student_logprobs = slice_student_logits[:min_seq_length, :].log_softmax(-1)
    teacher_logprobs = slice_teacher_logits[:min_seq_length, :].log_softmax(-1)

    per_token_kls = (teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)).sum(-1) # (T,)
    if avg_over_sequence:
        per_sequence_kls = per_token_kls.sum(-1) / per_token_kls.shape[-1]
    else:
        per_sequence_kls = per_token_kls.sum(-1)
    
    self_distillation_loss = per_sequence_kls.mean()
    return self_distillation_loss


class DFTTrainer(DPOTrainer):
    """Modified DPO trainer that additionally applies a knowledge distillation loss to out-of-domain data.

    While the DPO trainer expects a dataset with columns "prompt", "chosen", and "rejected", this trainer
    expects a dataset with columns "prompt", "chosen", "rejected", "hard_negative", and "hard_negative"
    """
    def __init__(self, *args, 
                 teacher_formatting_func,
                 response_template: str = "[/INST]", sigma_sd = 1.0, **kwargs):
        self.teacher_formatting_func = teacher_formatting_func
        self.response_template = response_template
        self.sigma_sd = sigma_sd
        super().__init__(*args, **kwargs)

    def compute_sd_loss(
            self, model, inputs, train_eval: str = "train"
    ):
        
        student_inputs = {
            "input_ids": inputs["student_input_ids"],
            "attention_mask": inputs["student_attention_mask"],
            "labels": inputs["student_labels"],
        }

        teacher_inputs = {
            "input_ids": inputs["teacher_input_ids"],
            "attention_mask": inputs["teacher_attention_mask"],
            "labels": inputs["teacher_labels"],
        }

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Convert to Tensor | Different Batch has different sized tensors, how do they deal with that? Should it be already token care of in the dataloader?
        student_batch = convert_batch(student_inputs, ignore_index=-100, pad_token_id=pad_token_id)
        teacher_batch = convert_batch(teacher_inputs, ignore_index=-100, pad_token_id=pad_token_id)

        # Sudo
        # Need student logits / teacher logits and etc
        student_output = self.model(**student_batch)
        with torch.no_grad():
            teacher_output = self.model(**teacher_batch)

        sd_loss = compute_self_distillation_loss(teacher_batch["labels"], teacher_output["logits"], student_batch["labels"], student_output["logits"], ignore_index=-100, avg_over_sequence=True)        

        return sd_loss, {"sd_loss": sd_loss.item()}

     # This is precisely where we over-write the loss computation function towards ANY customary loss functional
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        """Input dict has key "in_domain" which is binary flag """
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            dpo_loss, dpo_metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

            sd_loss, sd_metrics = 0, {}
            if self.sigma_sd > 0:
                sd_loss, sd_metrics = self.compute_sd_loss(model, inputs, train_eval="train")
            
            # Compute combined loss
            loss = dpo_loss * 0 + self.sigma_sd * sd_loss

            loss = dpo_loss
            metrics = {
                **dpo_metrics,
                **sd_metrics
            }

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    
    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        https://huggingface.co/blog/dpo-trl
        According to the blog above, DPO requires the prompt, chosen, rejected text field to be pre-formatted (chat template)

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        feature = convert_feature(feature, self.tokenizer)
        batch = super().tokenize_row(feature, model)
        # DPO tokenizer_row only process out accept & reject information etc. 

        # Here we use the feature again to include extra columns into the batch which will then be used to compute loss function value
        # teacher_formatting_prompt_func = self.teacher_formatting_func

        # Here we are allowed to add in extra columns (we can even discard the original loss)
        teacher_input = self.tokenizer(
            feature["teacher_format_prompt"], truncation=True, max_length=self.max_length, add_special_tokens=False
        )
        student_input = self.tokenizer(
            feature["student_format_prompt"], truncation=True, max_length=self.max_length, add_special_tokens=False
        )
        teacher_labels = get_completion_only_labels(self.tokenizer, self.response_template, teacher_input["input_ids"])
        student_labels = get_completion_only_labels(self.tokenizer, self.response_template, student_input["input_ids"])

        batch["student_input_ids"] = student_input["input_ids"]
        batch["student_attention_mask"] = student_input["attention_mask"]
        batch["student_labels"] = student_labels
        batch["teacher_input_ids"] = teacher_input["input_ids"]
        batch["teacher_attention_mask"] = teacher_input["attention_mask"]
        batch["teacher_labels"] = teacher_labels

        return batch