import random, datasets
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