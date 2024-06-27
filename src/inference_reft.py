from huggingface_hub import login
login(
  token=HF_TOKEN, # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

import torch
import transformers
import pyreft 
from pyreft import ReftModel
from datasets import load_dataset
device = "cuda" if torch.cuda.is_available() else "cpu"

########################
# Load Llama3-8B model #
########################

model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

model_max_length = 2048
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=model_max_length, 
    padding_side="right", use_fast=False)
if "Meta-Llama-3-" in model_name_or_path:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
else:
    tokenizer.pad_token = tokenizer.unk_token

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

##################### 
# Load Reft adaptor #
#####################

reft_model = ReftModel.load("Ksgk-fy/Zalinger02_reft_llama3", model, from_huggingface_hub=True)
reft_model.set_device("cuda")

# Load dataset
system_prompt = "Follow the instruction closely and provide your answer."
dataset = load_dataset("Ksgk-fy/alignment-sft-test2-mode-1", split="test")
data = dataset[3]

#####################
#   Run Inference   # 
#####################

# tokenize and prepare the input
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": system_prompt}, {"role": "user", "content": data['prompt']}], 
    tokenize=False)
prompt = tokenizer(prompt, return_tensors="pt").to(device)

# get reft model configuration
reft_config = pyreft.ReftConfig(representations=[{
    "layer": l, "component": "block_output",
    "low_rank_dimension": 2,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=2)} for l in [8, 16, 24]])
share_weights = True # whether the prefix and suffix interventions sharing weights.
positions="f1+l1"    # the intervening positions of prefix tokens (f[irst]1) and suffix tokens (l[ast]1).
first_n, last_n = pyreft.parse_positions(positions)

unit_locations = torch.IntTensor([pyreft.get_intervention_locations(
    last_position=prompt["input_ids"].shape[-1], 
    first_n=first_n, 
    last_n=last_n,
    pad_mode="last",
    num_interventions=len(reft_config.representations),
    share_weights=share_weights
)]).permute(1, 0, 2).tolist()

_, reft_response = reft_model.generate(
    prompt, unit_locations={"sources->base": (None, unit_locations)},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
    eos_token_id=terminators, early_stopping=True
)
response = tokenizer.decode(reft_response[0])