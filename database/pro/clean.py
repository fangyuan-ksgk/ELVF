import re
from datasets import Dataset

def clear_reflect_tag(txt):
    """ 
    Clear <reflect> ... </reflect> tags and everything in-between them. 
    """
    txt_cleaned = re.sub(r'<reflect>.*?</reflect>', '', txt)
    return txt_cleaned

def align_name(txt):
    """ 
    Replace Assistant with Customer
    """
    txt = re.sub(r'Assistant:', 'Customer:', txt)
    txt = re.sub(r"Maria:", "Customer:", txt)
    txt = re.sub(r'Agent:', 'Agent:', txt)
    return txt


def clean_conv_data(txt, edit_reflect=True):
    txt = align_name(txt)
    if edit_reflect:
        clean_txt = clear_reflect_tag(txt)
    else:
        clean_txt = txt
    # One transcript, multiple rounds, and incremental chat is the simplest way forward ... 
    # Convert to standard messages format 
    messages = []
    for t in clean_txt.split("\n"):
        if "Agent:" in t:
            agent_t = t.split("Agent: ")[-1].strip("\n")
            messages.append({"role": "user", "content": agent_t})
        elif "Customer: " in t:
            customer_t = t.split("Customer: ")[-1].strip("\n")
            messages.append({"role": "assistant", "content": customer_t})
    return messages

def split_file(txt):
    """ 
    Split files into a list of txts, each txt is a complete conversation
    -- User
    /// Conversation no:
    Essentially when the line does not start with Agent / Maria, we need to split them 
    """
    txts = []
    curr_txt = []
    for t in txt.split("\n"):
        if t.strip().startswith(("Agent:", "Maria:", "Customer:")):
            curr_txt.append(t)
        elif t.strip().startswith("/// Conversation no:"):
            if curr_txt:
                txts.append("\n".join(curr_txt))
            curr_txt = [t]
        else:
            if curr_txt:
                curr_txt.append(t)
    
    # Add the last conversation if it exists
    if curr_txt:
        txts.append("\n".join(curr_txt))
    
    return txts



def clean_response(txt):
    """ 
    Parse & Concatenate Agent & Customer Responses
    """
    txt= align_name(txt)
    agent_rs = []
    customer_rs = []
    clean_txt = clear_reflect_tag(txt)
    for t in clean_txt.split("\n"):
        if "Agent:" in t:
            agent_t = t.split("Agent: ")[-1].strip("\n")
            agent_rs.append(agent_t)
        elif "Customer: " in t:
            customer_t = t.split("Customer: ")[-1].strip("\n")
            customer_rs.append(customer_t)
    return agent_rs, customer_rs

def clean_conv_data_agent(txt, edit_reflect=True):
    txt = align_name(txt)
    if edit_reflect:
        clean_txt = clear_reflect_tag(txt)
    else:
        clean_txt = txt
    # One transcript, multiple rounds, and incremental chat is the simplest way forward ... 
    # Convert to standard messages format 
    messages = []
    for t in clean_txt.split("\n"):
        if "Agent:" in t:
            agent_t = t.split("Agent: ")[-1].strip("\n")
            messages.append({"role": "assistant", "content": agent_t})
        elif "Customer: " in t:
            customer_t = t.split("Customer: ")[-1].strip("\n")
            messages.append({"role": "user", "content": customer_t})
    return messages


def clean_genius_conv(txt, name):
    """ 
    Clean up Genius Conversation Transcript
    """
    clean_txt = txt
    # One transcript, multiple rounds, and incremental chat is the simplest way forward ... 
    # Convert to standard messages format 
    messages = []
    names = [f"{name}:", "Lex Fridman:"]
    for t in clean_txt.split("\n"):
        if names[0] in t:
            text = t.split(names[0]+" ")[-1].strip("\n")
            messages.append({"role":"user", "content": text})
        elif names[1] in t:
            text = t.split(names[1]+" ")[-1].strip("\n")
            messages.append({"role":"assistant", "content": text})
    return messages


def format_prompt_completion(ms, tokenizer):
    # Ms contains multiple rounds of user - assistant iterations
    format_prompt = tokenizer.apply_chat_template(ms, tokenize=False)
    utterance = ms[-1]["content"].strip()
    # Parse out the last utterance from assistant
    # print("Line 60 utterance to split: ", format_prompt)
    query_prompt, end_prompt = format_prompt.split(utterance)
    # Form the prompt & completion pair
    prompt, completion = query_prompt, utterance + end_prompt
    return prompt, completion

# Process the messages top-down to find assistant utterance as target completion
def process_prompt_completion_pairs(messages, tokenizer):
    prompts, completions = [], []
    for i in range(len(messages)):
        if messages[i]["role"] == "assistant" and i != 0:
            ms = messages[:i+1]
            try:
                prompt, completion = format_prompt_completion(ms, tokenizer)
                prompts.append(prompt)
                completions.append(completion)
            except:
                continue
    return prompts, completions

# Global Function Wrapper
def process_transcript(txt, tokenizer, customer=True):
    if customer:
        messages = clean_conv_data(txt)
    else:
        messages = clean_conv_data_agent(txt)
    prompts, completions = process_prompt_completion_pairs(messages, tokenizer)
    return prompts, completions


def process_genius_transcript(txt, tokenizer, name):
    messages = clean_genius_conv(txt, name)
    prompts, completions = process_prompt_completion_pairs(messages, tokenizer)
    return prompts, completions

from huggingface_hub import HfApi, HfFolder
def upload_dataset_to_huggingface(dataset, dataset_name="my_dataset"):
    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face token not found. Please login using `huggingface-cli login`.")
    user = api.whoami(token=token)['name']
    repo_url = api.create_repo(token=token, repo_id =dataset_name, repo_type="dataset", exist_ok=True)
    dataset.push_to_hub(repo_id=dataset_name, private=False, token=token)
    print(f"Dataset uploaded to {repo_url}")



# Collect Prompt & Completion Pairs
def collect_prompt_completion(txt_files, tokenizer):
    prompts = []
    completions = []
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            txt = f.read()
            tmp_prompts, tmp_completions = process_transcript(txt, tokenizer, customer=False)
        prompts.extend(tmp_prompts)
        completions.extend(tmp_completions)
    return prompts, completions


def mix_dataset(conv_prompts, conv_completions, ooc_prompts, ooc_completions, ooc_ratio=0.2):

    # Calculate the number of times to repeat the OOC dataset
    num_conv = len(conv_prompts)
    num_ooc = len(ooc_prompts)
    target_ooc = int(num_conv * ooc_ratio / (1 - ooc_ratio))
    repeat_factor = max(1, target_ooc // num_ooc)

    # Augment (repeat) the OOC dataset
    augmented_ooc_prompts = ooc_prompts * repeat_factor
    augmented_ooc_completions = ooc_completions * repeat_factor

    # Combine the datasets
    combined_prompts = conv_prompts + augmented_ooc_prompts
    combined_completions = conv_completions + augmented_ooc_completions

    # Create a combined dataset
    dataset_dict = {
        "prompt": combined_prompts,
        "completion": combined_completions
    }
    dataset = Dataset.from_dict(dataset_dict)
    dataset = {"train": dataset}

    # Shuffle the dataset to mix the two types of data
    dataset["train"] = dataset["train"].shuffle(seed=42)

    return dataset 


def process_messages_list(messages_list, tokenizer):
    """ 
    Process a list of messages (list of dict) into prompts and completions
    """
    all_prompts, all_completions = [], []
    for messages in messages_list:
        prompts, completions = process_prompt_completion_pairs(messages, tokenizer)
        all_prompts.extend(prompts)
        all_completions.extend(completions)
    return all_prompts, all_completions