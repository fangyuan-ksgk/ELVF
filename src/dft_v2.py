import warnings
from typing import Union, Dict, Any, Tuple, List, Literal, Optional

import torch
import numpy as np
from torch import nn
from trl import SFTTrainer
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
        labels[:] = -100
    else:
        # Make pytorch loss function ignore all tokens up through the end of the response key
        labels[:response_token_ids_end_idx] = -100
    return labels.tolist()

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

def masked_lm_loss(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask: torch.LongTensor,
) -> torch.FloatTensor:

    # Ensure we set ignore indices where there's no attention
    labels[attention_mask == 0] = -100
    # Shift so that tokens < n predict n | i-th logit prediction is for i-th token | shift to match pred with GT
    shift_logits = logits[..., :-1, :].contiguous() # (B, T-1, D)
    
    shift_labels = labels[..., 1:].contiguous() # (B, T-1)

    B, T = shift_labels.shape
    shift_logits = shift_logits.view(B, -1, T) # (B, D, T-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits, shift_labels)
    return loss.sum(-1).mean(0)



class DFTTrainer(SFTTrainer):
    def __init__(self, *args, sigma_soft: float = 0.3, sigma_hard: float = 0.3, 
                 student_formatting_func, 
                 teacher_formatting_func,
                 response_template: str = "[/INST]", 
                 kd_temperature: float = 1, kd_lambda: float = 0.5, use_avg_kl: bool = False, ignore_index: int = -100, **kwargs):
        self.response_template = response_template
        self.tokenizer = kwargs["tokenizer"]
        self.student_formatting_func = student_formatting_func
        self.teacher_formatting_func = teacher_formatting_func
        self.ignore_index = ignore_index
        self.sigma_soft = sigma_soft
        self.sigma_hard = sigma_hard
        self.kd_temperature = kd_temperature
        self.kd_lambda = kd_lambda
        self.use_avg_kl = use_avg_kl
        super().__init__(*args, **kwargs)
    
    def compute_distillation_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            student_batch: Dict[str, Union[List, torch.LongTensor]],
            teacher_batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train", # Literal is likely similar to Enum
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        
        metrics = {}

        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                student_outputs = model(**student_batch)  
            with torch.cuda.amp.autocast():
                teacher_outputs = model(**teacher_batch) 
            
        else:
            model.to("cpu")
            student_outputs = model(**student_batch)
            teacher_outputs = model(**teacher_batch)

        teacher_outputs.requires_grad = False
            
        teacher_labels = teacher_batch["labels"]
        teacher_logits = teacher_outputs.logits / self.kd_temperature
        student_labels = student_batch["labels"]
        student_logits = student_outputs.logits / self.kd_temperature

        attention_mask_student = student_batch["attention_mask"]

        # Compute Self Distillation Loss 
        self_distill_loss = compute_self_distillation_loss(teacher_labels, teacher_logits, student_labels, student_logits)

        return self_distill_loss


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        

        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

        teacher_inputs = {
            "input_ids": inputs["teacher_input_ids"],
            "attention_mask": inputs["teacher_attention_mask"],
            "labels": inputs["teacher_labels"],
        }

        ignore_index = self.ignore_index
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Convert to Tensor | Different Batch has different sized tensors, how do they deal with that? Should it be already token care of in the dataloader?
        student_batch = convert_batch(student_inputs, ignore_index=ignore_index, pad_token_id=pad_token_id)
        teacher_batch = convert_batch(teacher_inputs, ignore_index=self.ignore_index, pad_token_id=pad_token_id)
        
        sd_loss = self.compute_distillation_loss(model, student_batch, teacher_batch, train_eval="train")

        target_loss = super().compute_loss(model, student_batch, return_outputs=False)

        loss = (1 - self.kd_lambda) * target_loss + self.kd_lambda * sd_loss

        target_loss_metric = {"target_loss": target_loss.detach().cpu().item()}
        distill_loss_metric = {"distill_loss": sd_loss.detach().cpu().item()}
        overall_loss_metric = {"loss": loss.detach().cpu().item()}
        metric = {**target_loss_metric, **distill_loss_metric, **overall_loss_metric}
        return (loss, metric) if return_outputs else loss
    
    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=False,
    ):
        # print("Check the keys within the original dataset: ", dataset[0].keys())
        pdataset = super()._prepare_non_packed_dataloader(
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func = self.student_formatting_func,
            add_special_tokens = add_special_tokens,
            remove_unused_columns=False,
        )
        
        response_template = self.response_template
        teacher_formatting_prompt_func = self.teacher_formatting_func

        def tokenize_teacher(row):
            row["labels"] = get_completion_only_labels(tokenizer, response_template, row["input_ids"])
            row["format_teacher_prompt"] = teacher_formatting_prompt_func(row)
            teacher_input = tokenizer(
                    row["format_teacher_prompt"], truncation=True, padding=False, max_length=max_seq_length, add_special_tokens=False
                )
            teacher_labels = get_completion_only_labels(tokenizer, response_template, teacher_input["input_ids"])
            row["teacher_input_ids"] = teacher_input["input_ids"]
            row["teacher_attention_mask"] = teacher_input["attention_mask"]
            row["teacher_labels"] = teacher_labels
            return row

        qdataset = pdataset.map(tokenize_teacher, batched=False)
        qdataset = qdataset.remove_columns(["prompt", "teacher_prompt", "completion", "format_teacher_prompt"])
        
        return qdataset