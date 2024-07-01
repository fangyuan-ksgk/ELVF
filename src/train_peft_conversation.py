import sys, os, json, glob, argparse
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
from database.pro.clean import collect_prompt_completion, mix_dataset
import glob
from transformers import AutoTokenizer
from datasets import Dataset


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
        args.remove_unused_columns=True
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


# Pending Dataset preparation (huggingface ver.)

if __name__ == "__main__":
    from huggingface_hub import login
    HF_TOKEN = "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"
    login(
      token=HF_TOKEN,
      add_to_git_credential=True
    )

    # Collect Conversation Data: Prompt & Completion Pairs
    conv_files = glob.glob("database/pro/transcript/transcript_*.txt")
    ooc_files = glob.glob("database/pro/transcript/ooc_transcript_*.txt")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # ratio = [4, 0.5, 8] # Ratio of Conversation, Self-Recognition, General Cognition Data Points
    ratio = [4., 1., 4.] # Enhancing Self-Recognition, Reduce General Cognition
    dataset = mix_dataset(conv_files, ratio, tokenizer)

    # Genius Trainer
    # from datasets import load_dataset
    # from datasets import Dataset
    # # Load dataset from Hugging Face
    # dataset = load_dataset("Ksgk-fy/genius_upload")
    # # Shuffle the dataset and SubSample 1.5K datapoints
    # subset = dataset["train"].shuffle(seed=42).select(range(1800))
    # dataset = {"train": subset}

    # Train the model
    train_peft("configs/config_sft_v12.json", dataset, run_id="Maria_ooc_patch_v1")