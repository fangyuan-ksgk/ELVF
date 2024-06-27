# Simplistic Sample Approach
from openai import OpenAI
from os import getenv
from os import remove
from dataset.feedback_utils_v2 import Feedback
import json
from tqdm import tqdm
from models import OpenAIModel, OpenRouterModel
from search_complete import search_completion, sample_prompts

oai_model = OpenAIModel("gpt-4o")
route_model = OpenRouterModel()



# Sampling Feedback based Prompt Completions | With Consistency Check and Iteration Limit

if __name__ == "__main__":
    no_elephant_feedback = Feedback(content = "Do not talk about elephant")

    with open("database/prompts.json", "r") as f:
        prompts = json.load(f)

    run_errors = []
    pb = tqdm(prompts, desc="Feedback Adapted Response Search w. Consistency Check")
    for prompt in prompts:
        try:
            search_completion(prompt = prompt, 
                              feedback_content=no_elephant_feedback.content, 
                              get_oai_response = oai_model.get_response, 
                              get_openrouter_response = route_model.get_response)
        except Exception as e:
            info = {"prompt": prompt, "error": str(e)}
            run_errors.append(info)

        pb.update(1)

    # Update Search Completion Information for each prompts
    no_elephant_feedback.update_feedback_search_completion()

    with open("database/run_errors.json", "w") as outfile:
        json.dump(run_errors, outfile, indent=2)
    print(f"Run errors saved to run_errors.json. Total errors: {len(run_errors)}")
