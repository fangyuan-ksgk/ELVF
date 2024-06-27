from .search_complete import *
from .models import OpenAIModel, OpenRouterModel
from .inference import ReftInferencer, run_ft_inference
from .dataset.format_v2 import to_distill_sft

import time
from tqdm import tqdm as tqdm
import pandas as pd

def run_evaluation_gpt(df, feedback):
    """ 
    Evaluation with GPT-4o and OpenRouter - Reasoning & Self-Play focused evaluation
    """
    # Run Evaluation
    oai_model = OpenAIModel("gpt-4o")
    route_model = OpenRouterModel()

    # Reasoning is playing chess in your own mind: you need to think and judge your steps
    pb = tqdm(total=len(df))
    performance_infos = []
    for idx in range(len(df)):
        prompt = df["prompt"].iloc[idx]
        completion = df["pred"].iloc[idx]  
        success, n_attempt = False, 0
        while not success and n_attempt < 3:
            try:
                accept, rationale = check_performance_detail(feedback, prompt, completion, 
                        get_oai_response = oai_model.get_completion,
                        get_openrouter_response = route_model.get_completion)
                success = True
            except Exception as e:
                n_attempt += 1
                time.sleep(4)    
        performance_infos.append({"prompt": prompt, "pred": completion, "accept":accept, "rationale": rationale})
        pb.update(1)
    return pd.DataFrame(performance_infos)


def run_gpt_check(df, feedback):
    """ 
    Use GPT-4 to scrutinize on the prometheus evaluation result
    """
    # Run Evaluation
    oai_model = OpenAIModel("gpt-4o")
    route_model = OpenRouterModel()

    # Reasoning is playing chess in your own mind: you need to think and judge your steps
    pb = tqdm(total=len(df))
    performance_infos = []
    for idx in range(len(df)):
        prompt = df["prompt"].iloc[idx]
        completion = df["pred"].iloc[idx]  
        score, rationale, gt = df[["score", "feedback", "gt"]].iloc[idx]
        accept = int(score) > 3
        success, n_attempt = accept, 0 # Only Check the Un-Accepted Judgement --> Prometheus hallucinate a bit more in there
        while not success and n_attempt < 3:
            try:
                accept, rationale = check_performance_detail(feedback, prompt, completion, 
                        get_oai_response = oai_model.get_completion,
                        get_openrouter_response = route_model.get_completion)
                success = True
                pb.write("Performance Check completed! Next ...")
            except Exception as e:
                print(e)
                pb.write("Attempt Unsuccessful, retrying...")
                n_attempt += 1
                time.sleep(4)    
        
        performance_infos.append({"prompt": prompt, "pred": completion, "accept":accept, "rationale": rationale, "gt": gt})
        pb.update(1)
    return pd.DataFrame(performance_infos)


