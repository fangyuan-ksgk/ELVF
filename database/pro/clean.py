import re
from datasets import Dataset
from prep import *

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


# def mix_dataset(conv_prompts, conv_completions, ooc_prompts, ooc_completions, ooc_ratio=0.2):

#     # Calculate the number of times to repeat the OOC dataset
#     num_conv = len(conv_prompts)
#     num_ooc = len(ooc_prompts)
#     target_ooc = int(num_conv * ooc_ratio / (1 - ooc_ratio))
#     repeat_factor = max(1, target_ooc // num_ooc)

#     # Augment (repeat) the OOC dataset
#     augmented_ooc_prompts = ooc_prompts * repeat_factor
#     augmented_ooc_completions = ooc_completions * repeat_factor

#     # Combine the datasets
#     combined_prompts = conv_prompts + augmented_ooc_prompts
#     combined_completions = conv_completions + augmented_ooc_completions

#     # Create a combined dataset
#     dataset_dict = {
#         "prompt": combined_prompts,
#         "completion": combined_completions
#     }
#     dataset = Dataset.from_dict(dataset_dict)
#     dataset = {"train": dataset}

#     # Shuffle the dataset to mix the two types of data
#     dataset["train"] = dataset["train"].shuffle(seed=42)

#     return dataset 


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


def combine_dataset(data_list):
    """ 
    Fix for now -- proportion fixed as well
    """
    from datasets import Dataset 
    combined_prompts = []
    combined_completions = []
    for data in data_list:
        combined_prompts.extend(data[0])
        combined_completions.extend(data[1])
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


import random
from datasets import load_dataset

#########################
# Cognitive Enhancement #
#########################

# General Cognitive Dataset
# -- Instruction Following 
def prepare_general_cognitive_dataset(n=8000, seed=42):
    """
    Use UltraChat Dataset to enhance the general cognitive capacity of fine-tuned model
    - Keep Random Seed & Number of Messages as Input
    """
    
    # Load UltraChat Dataset from Huggingface
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    
    full_dataset = ds["train_sft"]
    dataset_size = len(full_dataset)
    
    # Set a seed for reproducibility
    random.seed(seed)
    
    # Randomly select 8000 indices
    sampled_indices = random.sample(range(dataset_size), n)
    
    # Create a new dataset with the sampled examples
    sampled_dataset = [full_dataset[i]["messages"] for i in sampled_indices]

    return sampled_dataset

# Self-Cognitive Dataset
# -- Avoids Breaking Character
def generate_self_recognition_dataset(num_samples=10):
    """ 
    Code-Based Self Recognition Datset Generator
    - Could benefit from diverse paraphasing here 
    """
    # Define possible attributes
    names = ["Maria", "Sofia", "Isabella", "Gabriela", "Camila"]
    ages = range(25, 60)
    cities = ["Manila", "Quezon City", "Cebu City", "Davao City", "Makati"]
    occupations = ["stay-at-home mom", "teacher", "nurse", "office worker", "small business owner"]
    marriage_statuses = ["single", "married", "divorced", "widowed"]
    family_members = range(0, 5)  # Number of children
    monthly_incomes = range(15000, 100001, 5000)  # In Philippine Peso
    health_conditions = ["healthy", "minor chronic condition", "major health issue"]
    monthly_expenses = range(10000, 80001, 5000)  # In Philippine Peso
    spending_habits = ["frugal", "moderate", "lavish"]
    existing_assets = ["savings account", "car", "house", "investments"]
    existing_liabilities = ["personal loan", "mortgage", "credit card debt", "none"]

    # Questions and their corresponding answer formats
    questions = [
        "What's your name?",
        "Where do you live?",
        "What's your age?",
        "What's your marital status?",
        "Do you have any children?",
        "What's your monthly income?",
        "What do you do for a living?",
        "How's your health?",
        "What are your monthly expenses like?",
        "Do you have any assets?",
        "Do you have any debts or loans?",
        "Are you from FWD insurance?",
        "What insurance product do you offer?"
    ]

    datasets = []

    for _ in range(num_samples):
        name = random.choice(names)
        age = random.choice(ages)
        city = random.choice(cities)
        occupation = random.choice(occupations)
        marital_status = random.choice(marriage_statuses)
        children = random.choice(family_members)
        income = random.choice(monthly_incomes)
        health = random.choice(health_conditions)
        expenses = random.choice(monthly_expenses)
        spending = random.choice(spending_habits)
        assets = random.sample(existing_assets, k=random.randint(0, len(existing_assets)))
        liabilities = random.sample(existing_liabilities, k=random.randint(0, len(existing_liabilities)))

        system_content = f"You are {name}, a {age}-year-old Filipino woman living in {city}. You are {marital_status} with {children} children. You work as a {occupation} with a monthly income of {income} pesos. Your health is {health}."

        dataset = [
            {
                "role": "system",
                "content": system_content
            }
        ]

        for question in questions:
            dataset.append({"role": "user", "content": question})
            
            if "name" in question.lower():
                answer = f"My name is {name}."
            elif "live" in question.lower():
                answer = f"I live in {city}."
            elif "age" in question.lower():
                answer = f"I'm {age} years old."
            elif "marital status" in question.lower():
                answer = f"I'm {marital_status}."
            elif "children" in question.lower():
                answer = f"I have {children} {'child' if children == 1 else 'children'}." if children > 0 else "I don't have any children."
            elif "monthly income" in question.lower():
                answer = f"My monthly income is {income} pesos."
            elif "do you do" in question.lower() or "for a living" in question.lower():
                answer = f"I work as a {occupation}."
            elif "health" in question.lower():
                answer = f"My health is {health}."
            elif "monthly expenses" in question.lower():
                answer = f"My monthly expenses are about {expenses} pesos. I would say my spending habits are {spending}."
            elif "assets" in question.lower():
                if assets:
                    answer = f"Yes, I have {', '.join(assets[:-1])}{' and ' if len(assets) > 1 else ''}{assets[-1] if assets else ''}."
                else:
                    answer = "I don't have any significant assets at the moment."
            elif "debts" in question.lower() or "loans" in question.lower():
                if "none" in liabilities:
                    answer = "I don't have any debts or loans at the moment."
                else:
                    answer = f"Yes, I have {', '.join(liabilities[:-1])}{' and ' if len(liabilities) > 1 else ''}{liabilities[-1] if liabilities else ''}."
            elif "FWD insurance" in question:
                answer = "No, I'm not from FWD insurance."
            elif "insurance product" in question:
                answer = "I'm sorry, but I don't offer any insurance products. I'm not an insurance agent."
            else:
                answer = "I'm not sure how to answer that question."

            dataset.append({"role": "assistant", "content": answer})

        datasets.append(dataset)

    return datasets


def mix_dataset(conv_files, ratio, tokenizer): 
    """
    Mixing Dataset For Fine-tuning model
    - Conversation Data
    - Self Cognitive Data
    - General Cognitive Data
    """
    # Conversation Data -- 4K 
    conv_data = collect_prompt_completion(conv_files, tokenizer)
    
    # Self-Cognition Data - 0.5K
    # ooc_prompts, ooc_completions = collect_prompt_completion(ooc_files, tokenizer) # Ralph's OOC DataPoints | No system prompt injected, yet
    n_self_cog = int(4000 * ratio[1] / ratio[0])
    sample_datasets = generate_self_recognition_dataset(n_self_cog)
    self_cognition_data = process_messages_list(sample_datasets, tokenizer)
    
    # General Cognitive Data - 8.1K 
    n_general_cog = int(4000 * ratio[2] / ratio[0])
    sample_datasets = prepare_general_cognitive_dataset(n=n_general_cog)
    general_cognition_data = process_messages_list(sample_datasets, tokenizer)
    
    dataset = combine_dataset([conv_data, self_cognition_data, general_cognition_data])
    return dataset