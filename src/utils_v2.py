import torch
from typing import Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import pyreft
from .represent import parse_positions


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32
}

@dataclass
class ModelArguments:
    base_model_id: Optional[str] = "meta-llama/Meta-Llama-3-8B-Instruct"
    new_model_id: Optional[str] = "Yo-01"
    use_quant: Optional[bool] = True
    load_in_4bit: Optional[bool] = True
    bnb_4bit_use_double_quant: Optional[bool] = True
    bnb_4bit_quant_type: Optional[str] = "nf4"
    bnb_4bit_compute_dtype: Optional[torch.dtype] = torch.bfloat16
    device_map: Optional[str] = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"
    torch_dtype: Optional[torch.dtype] = torch.bfloat16

    def make(self, return_terminator: bool = False):
        """ 
        Make the args into LLM model
        """
        if self.use_quant:
            # BitsAndBytesConfig int-4 config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit, bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant, 
                bnb_4bit_quant_type=self.bnb_4bit_quant_type, bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype
            )
        else:
            bnb_config = None

        if torch.cuda.is_available() and self.use_quant:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                device_map="auto",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config
            )
        elif torch.cuda.is_available() and not self.use_quant:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto")
            

        else:
            print("Loading LLM without GPU")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                device_map="mps") # MLX should have some support already for LoRA script
            
        model_max_length = 4096
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, 
            model_max_length=model_max_length, 
            padding_side="right", 
            use_fast=False)

        if "Meta-Llama-3-" in self.base_model_id:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer.pad_token = tokenizer.unk_token

        terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        if return_terminator:
            return model, tokenizer, terminators
        else:
            return model, tokenizer
    

@dataclass
class PeftArguments:
    lora_alpha: Optional[int] = 128
    lora_dropout: Optional[float] = 0.05
    r: Optional[int] = 256
    bias: Optional[str] = "none"
    target_modules: Optional[str] = "all-linear"
    task_type: Optional[str] = "CAUSAL_LM"

    def make(self):
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias=self.bias,
            target_modules=self.target_modules,
            task_type=self.task_type
        )
    

@dataclass 
class ReftArguments:
    low_rank_dimension: Optional[int] = 2
    intervention_layers: Optional[str] = "8+16+24"
    intervention_positions: Optional[str] = "f1+l1"
    share_weights: Optional[bool] = True

    def make_config(self, model):
        intervention_layers = self.intervention_layers.split("+")
        reft_config = pyreft.ReftConfig(representations=[{
                    "layer": l, "component": "block_output",
                    "low_rank_dimension": self.low_rank_dimension,
                    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
                    low_rank_dimension=self.low_rank_dimension)} for l in intervention_layers],
                    intervention_positions=self.intervention_positions,
                    share_weights=self.share_weights)
        
        return reft_config
    

arg_to_repo = lambda arg_file: arg_file.split("/config_")[-1].replace(".json", "_elvf")
# repo_to_arg = lambda repo_id: f"configs/config_{repo_id.split("Ksgk-fy/")[-1].replace('_elvf', '')}.json"
def repo_to_arg(repo_id):
    repo_name = repo_id.split("Ksgk-fy/")[-1]
    repo_name_to_arg = lambda repo_id: f"configs/config_{repo_id.replace('_elvf', '')}.json"
    return repo_name_to_arg(repo_name)