from src.dataset.feedback_utils_v2 import Feedback
from src.dataset.format_v2 import to_distill_sft
from tqdm import tqdm as tqdm
import pandas as pd
import os
from src import run_gpt_check, OpenAIModel, OpenRouterModel
from src.search_complete import *
from src.dataset.prompts_v2 import parse_prompt_from_response, AUGMENT_QUERY_TEMPLATE


def augment_on_corner(feedback, adaptor_id, aug_name="aug_r1"):
    """ 
    On Feedback, trained adaptor with id's evaluation result will be used to conduct targetted augmentation
    based on the corner cases found in the evaluation result. 
    """
    prom_eval_file_path = f"database/{feedback.file_name}/{adaptor_id}_eval.csv" # Evaluation result file path from Prometheus
    gpt_eval_file_path = f"database/{feedback.file_name}/{adaptor_id}_eval_gpt.csv" # Evaluation result file path from GPT-4o
    if not os.path.exists(prom_eval_file_path):
        raise FileNotFoundError("Prometheus Evaluation File not found")
    
    # Dataframe
    df = pd.read_csv(prom_eval_file_path).drop(columns=["Unnamed: 0"])
    # Get unique counts on score column
    score_counts = df['score'].value_counts()
    total_scores = score_counts.sum()
    high_scores = score_counts[score_counts.index >= 4].sum()
    high_score_pct = high_scores / total_scores * 100
    print(f"Prometheus Judgement: {high_score_pct:.2f}% of responses are accepted")

    # GPT4 double checks
    if not os.path.exists(gpt_eval_file_path):
        df_revise = run_gpt_check(df, feedback)
        df_revise.to_csv(gpt_eval_file_path)

    df_revise = pd.read_csv(gpt_eval_file_path).drop(columns=["Unnamed: 0"])
    accept_counts = df_revise['accept'].value_counts()
    total_accepts = accept_counts.sum()
    accepted = accept_counts[accept_counts.index == True].sum()
    accepted_pct = accepted / total_accepts * 100
    print(f"GPT-4 Judgement: {accepted_pct:.2f}% of responses are accepted")

    # Augmentation Time
    oai_model = OpenAIModel("gpt-4o")
    route_model = OpenRouterModel()
    corner_cases = df_revise[df_revise["accept"] == False]
    augmented_queries = []
    for idx, row in corner_cases.iterrows():
        prompt = row['prompt']
        
        # Agumentation Query
        augment_query = AUGMENT_QUERY_TEMPLATE.format(prompt = prompt, feedback_content = feedback.content)
        augment_response = oai_model.get_completion(prompt = augment_query, system_prompt = "You are a helpful assistant. Skilled in complex reasoning.")
        queries = parse_prompt_from_response(augment_response)
        
        # Search for Query-Specific Completion | Save to search_infos
        pb = tqdm(queries, desc="Augmentation Query Search")
        for prompt in queries:
            try:
                search_completion(prompt = prompt, 
                    feedback=feedback,
                    max_depth=4, 
                    get_oai_response = oai_model.get_completion, 
                    get_openrouter_response = route_model.get_completion,
                    save_folder=aug_name)
                
            except: 
                print("Error in Search Completion, Skip")
                continue
            pb.update(1)

if __name__ == "__main__":
    feedback = Feedback(content="Do not talk about elephant")
    adaptor_id = "elvf_sft_v2_1"
    aug_name = "aug_r1"
    augment_on_corner(feedback, adaptor_id=adaptor_id, aug_name=aug_name)