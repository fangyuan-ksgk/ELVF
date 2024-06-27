import sys, os
base_path =os.path.join(os.path.dirname(__file__), '..')
sys.path.append(base_path)


from src.utils_v2 import ModelArguments, PeftArguments
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser
from src.custom_collator import (DataCollatorForCompletionOnlyLM_v2, 
                                 get_format_func, 
                                 get_teacher_format_func,
                                 infer_response_template)
from src.dft_v2 import DFTTrainer
from src.dataset.feedback_utils_v2 import Feedback
from src.dataset.format_v2 import to_distill_sft
import json
import argparse


def train_peft(arg_file, dataset, run_id: str = "1"):

    # Load Argument Configuration & Get the Modes etc.
    with open(arg_file, "r") as f:
        arg_dict = json.load(f)

    # Load Model
    model_arg_parser = HfArgumentParser((ModelArguments,))
    model_args: ModelArguments = model_arg_parser.parse_dict(arg_dict["model_args"])[0]
    model, tokenizer = model_args.make()

    # Load LoRA arguments
    peft_args: PeftArguments = HfArgumentParser((PeftArguments,)).parse_dict(arg_dict["lora_args"])[0]
    peft_config = peft_args.make()

    # Load Training Arguments
    args = HfArgumentParser((TrainingArguments,)).parse_dict(arg_dict["training_args"])[0]
    args.output_dir = f"{args.output_dir}_{run_id}"

    # Trainer Preparation
    response_template = infer_response_template(tokenizer)
    collator = DataCollatorForCompletionOnlyLM_v2(response_template, tokenizer=tokenizer)
    formatting_prompt_func = get_format_func(tokenizer)
    teacher_formatting_prompt_func = get_teacher_format_func(tokenizer)

    algo = arg_dict["algorithm"]
    max_seq_length = 1024

    if algo == "sft":
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            # dataset_text_field="text", # Question: I do NOT think 'text' is one of the key in the dataset ??
            formatting_func=formatting_prompt_func,
            data_collator=collator,
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,  # We template with special tokens | Mistral v0.3 does not recognize this argument
                "append_concat_token": False, # No need to add additional separator token
            }
        )
    elif algo == "dft":
        trainer = DFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            formatting_func=formatting_prompt_func,
            student_formatting_func=formatting_prompt_func,
            teacher_formatting_func=teacher_formatting_prompt_func,
            data_collator=collator,
            response_template = response_template,
            dataset_kwargs={
                "add_special_tokens": False,  # We template with special tokens
                "append_concat_token": False, # No need to add additional separator token
            }
        )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--arg_file", type=str, default="configs/config_dft_v1.json")
    parser.add_argument("--arg_file", type=str, default="configs/config_sft_v1.json")
    parser.add_argument("--run_id", type=str, default="1")
    parser.add_argument("--augment", type=str, default="aug_r1")
    args = parser.parse_args()

    # feedback = Feedback(content = "Do not talk about elephant")
    feedback = Feedback(content="Roleplay as a philippine customer")

    # Apply Augmentation 
    feedback.boostrap_augment(args.augment)
    # Prepare Dataset
    dataset = to_distill_sft(feedback)
    # Train it
    train_peft(args.arg_file, dataset, run_id=args.run_id)