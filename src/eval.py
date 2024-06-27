from .search_complete import *
from .models import OpenAIModel, OpenRouterModel
from .inference import ReftInferencer, run_ft_inference
from .dataset.format_v2 import to_distill_sft
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
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



def run_eval_prometheus(df, feedback):
    """ 
    Evaluation with Prometheus - Language Model Evaluation - SpeedUp with vLLM serving
    """
    # Load Prometheus judge
    judge = PrometheusEval(model_id="prometheus-eval/prometheus-7b-v2.0", absolute_grade_template=ABSOLUTE_PROMPT)

    # Prepare Instruction, Rubric, Reference Answer and Prediction
    rubric_data = {
        "criteria": f"Is the model following the feedback -- {feedback.content} -- and providing a response to the query that makes sense?",
        "score1_description": "The model is not following the feedback and providing a response that does not make sense.",
        "score2_description": "The model is not following the feedback but is providing a response that make sense.",
        "score3_description": "The model is following the feedback but is providing a response that does not makes sense.",
        "score4_description": "The model is following the feedback and providing a response that makes sense.",
        "score5_description": "The model is following the feedback and providing a response that makes sense."
    }

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    instructions = df["prompt"].tolist()
    reference_answers = df["gt"].tolist()
    responses = df["pred"].tolist()

    # Run Evaluation
    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        params={},
        rubric=score_rubric,
        reference_answers=reference_answers
    )

    del judge
    
    return feedbacks, scores

def process_eval_(feedbacks, scores, df, feedback, adaptor_id):
    eval_infos = []
    for idx in range(len(df)):
        prompt, pred, gt = df[["prompt", "pred", "gt"]].iloc[idx]
        score = scores[idx]
        feedback_ = feedbacks[idx]
        accept = (score >= 4)
        d = {"prompt": prompt, "pred": pred, "gt": gt, "score": score, "feedback": feedback_}
        eval_infos.append(d)

    adaptor_id = adaptor_id.split("Ksgk-fy/")[-1]
    pd.DataFrame(eval_infos).to_csv(f"database/{feedback.file_name}/{adaptor_id}_eval.csv")
    return eval_infos


# Inference 
from huggingface_hub import login
login(
  token="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", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)


# adaptor_id = "Ksgk-fy/reft_v1_elvf"
def run_reft_eval(adaptor_id, feedback):
    """ 
    Run evaluation for REFT model
    """
    # Load Fast Adaptor
    f = ReftInferencer(adaptor_id)

    # Run Inference
    dataset = to_distill_sft(feedback)
    df_pred = run_ft_inference(f, dataset, train=True, run_id="1")

    # Basically anything above or equal 4 in score it a good response, otherwise it's bad 
    feedbacks, scores = run_eval_prometheus(df_pred, feedback)

    # Process Evaluation
    df_eval = process_eval_(feedbacks, scores, df_pred, feedback, adaptor_id)

    return df_eval