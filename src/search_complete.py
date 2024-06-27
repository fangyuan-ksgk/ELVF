# Think before you talk -- Search is required before generate final response
from .dataset.prompts_v2 import (
    EXTRAPOLATE_TEMPLATE, 
    SELF_PROMPT_TEMPLATE, 
    SEARCH_TEMPLATE,
    MAKE_SENSE_CHECK_TEMPLATE,
    GENERATE_PROMPT_TEMPLATE,
    parse_search_node,
    parse_make_sense_check,
    parse_prompt_from_response
    )
from .dataset.feedback_utils_v2 import Feedback
from datasets import Dataset
import json, os

# Sample Prompts
def sample_prompts(feedback: Feedback,
                   get_oai_response: callable = None,
                   get_openrouter_response: callable = None):
    response_oai = get_oai_response(GENERATE_PROMPT_TEMPLATE.format(content = feedback.content))
    response_route = get_openrouter_response(GENERATE_PROMPT_TEMPLATE.format(content = feedback.content), idx = 1)
    prompts_router = parse_prompt_from_response(response_route)
    prompts_oai = parse_prompt_from_response(response_oai)
    prompts = prompts_oai + prompts_router
    feedback.prompts = Dataset.from_dict({"prompt": prompts})
    return feedback.prompts

def check_performance(feedback: Feedback,
                      prompt: str,
                      pred_completion: str,
                      get_oai_response: callable = None,
                      get_openrouter_response: callable = None):
    """
    Used to check on the performance of the prediction
    """
    feedback_content = feedback.content
    # Check & Search
    search_node = get_oai_response(prompt = SEARCH_TEMPLATE.format(content = feedback_content, prompt = prompt, icl_complete = pred_completion), system_prompt = "You are a helpful assistant. Skilled in complex reasoning. ")
    search_edge = parse_search_node(search_node)
    make_sense_response = get_openrouter_response(prompt = MAKE_SENSE_CHECK_TEMPLATE.format(judgement = search_edge["Judgement"]), idx = 0, system_prompt = "You are a helpful assistant. Skilled in complex reasoning. ")
    # Accept or Reject
    accept = parse_make_sense_check(make_sense_response)
    return accept


def check_performance_detail(feedback: Feedback,
                      prompt: str,
                      pred_completion: str,
                      get_oai_response: callable = None,
                      get_openrouter_response: callable = None):
    """
    Used to check on the performance of the prediction
    """
    feedback_content = feedback.content
    # Check & Search
    search_node = get_oai_response(prompt = SEARCH_TEMPLATE.format(content = feedback_content, prompt = prompt, icl_complete = pred_completion), system_prompt = "You are a helpful assistant. Skilled in complex reasoning. ")
    search_edge = parse_search_node(search_node)
    make_sense_response = get_openrouter_response(prompt = MAKE_SENSE_CHECK_TEMPLATE.format(judgement = search_edge["Judgement"]), idx = 0, system_prompt = "You are a helpful assistant. Skilled in complex reasoning. ")
    # Accept or Reject
    accept = parse_make_sense_check(make_sense_response)
    return accept, search_edge["Issue"]


# 
def search_revision(prompt: str,
                    feedback: Feedback,
                    pred_completion: str,
                    max_depth: int = 5,
                    get_oai_response: callable = None,
                    get_openrouter_response: callable = None):
    """ 
    Given a (feedback, prompt) pair, search for a completion that satisfies the feedback and not contradicting previous knowledge
    - Pred_Completion: The completion that needs to be revised
    """
    # Self Consistency Search
    id = prompt.replace(" ","-").replace(".","")
    feedback_content = feedback.content
    infos = []
    accept, curr_depth = False, 0
    while (not accept and curr_depth < max_depth):
        # Check & Search
        search_node = get_oai_response(prompt = SEARCH_TEMPLATE.format(content = feedback_content, prompt = prompt, icl_complete = icl_complete), system_prompt = "You are a helpful assistant. Skilled in complex reasoning. ")
        search_edge = parse_search_node(search_node)
        make_sense_response = get_openrouter_response(prompt = MAKE_SENSE_CHECK_TEMPLATE.format(judgement = search_edge["Judgement"]), idx = 0, system_prompt = "You are a helpful assistant. Skilled in complex reasoning. ")
        # Accept or Reject
        accept = parse_make_sense_check(make_sense_response)
        # Info Recording
        info = {"prompt": prompt, "feedback": feedback_content, "icl_complete": icl_complete, "accept": accept, "judgement": search_edge["Judgement"], "advise": search_edge["Advice"]}
        infos.append(info)
        # Update Completion for Next Iteration
        icl_complete = search_edge["Revised Response"] if not accept else icl_complete
        curr_depth += 1

    if curr_depth == max_depth: # For these case, extra filtering is required || Wait till later
        last_info = {"prompt": prompt, "feedback": feedback_content, "icl_complete": search_edge["Revised Response"], "accept": True, "judgement": "", "advise": ""}
        infos.append(last_info)

    # Store Search Information
    os.makedirs(f"database/{feedback.file_name}", exist_ok=True)
    with open(f"database/{feedback.file_name}/search_info_{id}.json", "w") as outfile:
        json.dump(infos, outfile, indent=2)

    print(f"Search information saved to search_infos.json. Total iterations: {len(infos)}")
    

# Comprehension of a concept (feedback) involves extrapolation (for instance ..) | assumption: how about I say ... | verification: that does not make sense | iteration: try again
def search_completion(prompt: str, 
                      feedback: Feedback, 
                      max_depth: int = 5,
                      get_oai_response: callable = None,
                      get_openrouter_response: callable = None,
                      save_folder: str = ""):
    """ 
    Given a (feedback, prompt) pair, search for a completion that satisfies the feedback and not contradicting previous knowledge
    - [Self Extrpolation]: for instance here are a few (prompt, completion) examples which follow the feedback | Increase context
    - [In Context Prompt Completion]: Give the examples, provide a proper response
    - [Consistency Check]: Does the response make sense? | If not, provide advice on how to better follow the feedback
    - [Iterative Search]: Iterative refinement untill response is accepted, or maximum iteration is reached
    """

    id = prompt.replace(" ","-").replace(".","")
    feedback_content = feedback.content
    # Make Sense Check & Revision (If does not make sense)
    self_few_shot_prompt = get_oai_response(prompt = EXTRAPOLATE_TEMPLATE.format(content = feedback_content), 
                                            system_prompt = "")

    # Self Extrapolation | The 'Give me 100 prompts' is a bigger scaled self-extrapolation here
    icl_complete = get_oai_response(prompt = prompt, system_prompt = "You are a helpful assistant. Skilled in complex reasoning. " + SELF_PROMPT_TEMPLATE.format(content = feedback_content, self_prompt = self_few_shot_prompt))

    # Self Consistency Search
    infos = []
    accept, curr_depth = False, 0
    while (not accept and curr_depth < max_depth):
        # Check & Search
        search_node = get_oai_response(prompt = SEARCH_TEMPLATE.format(content = feedback_content, prompt = prompt, icl_complete = icl_complete), system_prompt = "You are a helpful assistant. Skilled in complex reasoning. ")
        search_edge = parse_search_node(search_node)
        make_sense_response = get_openrouter_response(prompt = MAKE_SENSE_CHECK_TEMPLATE.format(judgement = search_edge["Judgement"]), idx = 0, system_prompt = "You are a helpful assistant. Skilled in complex reasoning. ")
        # Accept or Reject
        accept = parse_make_sense_check(make_sense_response)
        # Info Recording
        info = {"prompt": prompt, "feedback": feedback_content, "icl_complete": icl_complete, "accept": accept, "judgement": search_edge["Judgement"], "advise": search_edge["Advice"]}
        infos.append(info)
        # Update Completion for Next Iteration
        icl_complete = search_edge["Revised Response"] if not accept else icl_complete
        curr_depth += 1

    if curr_depth == max_depth: # For these case, extra filtering is required || Wait till later
        last_info = {"prompt": prompt, "feedback": feedback_content, "icl_complete": search_edge["Revised Response"], "accept": True, "judgement": "", "advise": ""}
        infos.append(last_info)

    # Store the infos in a file
    os.makedirs(f"database/{feedback.file_name}", exist_ok=True)
    if save_folder == "":
        dir_path = f"database/{feedback.file_name}/"
    else:
        dir_path = f"database/{feedback.file_name}/{save_folder}/"
        os.makedirs(dir_path, exist_ok=True)
    with open(f"{dir_path}search_info_{id}.json", "w") as outfile:
        json.dump(infos, outfile, indent=2)
    print(f"Search information saved to search_infos.json. Total iterations: {len(infos)}")