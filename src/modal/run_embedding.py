import os
import json
import copy
from typing import Any
from modal import gpu, Mount
from src.modal.common import stub, VOLUME_CONFIG
from src.modal.utils import copy_json_files_recursively
from src.dataset.feedback_utils_v2 import Feedback
from src.dataset.format_v2 import to_distill_sft
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.represent import *
from tqdm import tqdm as tqdm
import pandas as pd



@stub.function(
    volumes=VOLUME_CONFIG,
    cpu=4.0,
    image=stub.gpu_image,
    gpu=gpu.A100(count=1),
    timeout=3600 * 12,
    concurrency_limit=512,
    mounts=[
        Mount.from_local_dir("configs", remote_path="/root/configs")
    ]
)
def get_hidden_embedding(model_name, trainset, avg_layer=[1,2,3], positions="f1+l1"):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded!")
    pb = tqdm(total=len(trainset), desc="Running inference to get hidden embeddings...")
    embed_vecs = []
    for data in trainset:
        prompt = prepare_prompt_reft(data, tokenizer)
        hidden_states = get_hidden_states(prompt, model, tokenizer)
        avg_hidden_states = get_average_of_layers(hidden_states, avg_layer)
        embed_vec = get_average_of_positions(avg_hidden_states, positions)
        embed_vecs.append(embed_vec.detach().cpu().numpy().tolist())
        pb.update(1)
    pb.close()
    print("Inference done!")
    return embed_vecs



# entry point
@stub.local_entrypoint()  # Runs locally to kick off remote training job.
def main():
    feedback = Feedback(content = "Do not talk about elephant")
    dataset = to_distill_sft(feedback)

    model_name = "microsoft/Phi-3-mini-128k-instruct"
    trainset = dataset["train"]
    avg_layer = [1,2,3]
    positions = "f1+l1"
    embed_vecs = get_hidden_embedding.remote(model_name, trainset, avg_layer, positions)

    pd.DataFrame(embed_vecs).to_csv("embeddings.csv", index=False)