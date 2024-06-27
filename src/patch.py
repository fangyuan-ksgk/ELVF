##### Patch Module #####
from ..database.pro.clean import clean_response 
import glob 
import datasets 
from datasets import Dataset
from sklearn.model_selection import train_test_split


label_map = {0: "Agent", 1: "Customer"}


# Issue Texts & Labels will be incrementally collected from actual conversations 
cases = [    
    ("Have you heard of FWD before?", 0),
    ("Have you heard about FWD before?", 0),
    ("I work at FWD insurance", 0),
    ("I am not interested in your insurance product", 1),
    ("So what is the different between FWD insurance and others?", 1),
    ("What is FWD again?", 1),
    ("I do not trust FWD.", 1),
    ("Hello Alex!", 1),
    ("Can you explain the benefits of FWD insurance?", 1),
    ("I'm looking for the best insurance coverage.", 1),
    ("FWD seems like a reliable company.", 1),
    ("What are the premiums for FWD insurance?", 1),
    ("I've had bad experiences with insurance companies before.", 1),
    ("Is FWD insurance available in my country?", 1),
    ("How does FWD compare to other major insurers?", 1),
    ("I'm satisfied with my current insurance provider.", 1),
    ("Tell me more about FWD's customer service.", 1)
]

def prepare_ooc_dataset(txt_files, cases):
    customer_responses = []
    agent_responses = []

    # Process customer and agent responses separately
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            txt = f.read()
            agent_rs, customer_rs = clean_response(txt)
            agent_responses.extend(agent_rs)
            customer_responses.extend(customer_rs)

    # We want to "mix" the customer responses with agent responses here
    all_responses = []
    labels = []
    from random import shuffle
    shuffle(agent_responses) # shuffle in-place
    shuffle(customer_responses) # shuffle in-place
    for i in range(min(len(agent_responses), len(customer_responses))):
        all_responses.append(agent_responses[i])
        all_responses.append(customer_responses[i])
        labels.append(0)
        labels.append(1)

    # Add in Issue Cases
    issue_texts = [case[0] for case in cases]
    issue_labels = [case[1] for case in cases]

    # Calculate the number of times to repeat the issue dataset
    num_repeats = int(len(all_responses) * 0.2 / len(issue_texts))

    # Repeat the issue dataset
    repeated_issue_texts = issue_texts * num_repeats
    repeated_issue_labels = issue_labels * num_repeats

    # Combine the original and issue datasets
    combined_texts = all_responses + repeated_issue_texts
    combined_labels = labels + repeated_issue_labels

    # Train & Test Split 
    from sklearn.model_selection import train_test_split

    train_dataset = Dataset.from_dict({"text": combined_texts, "label": combined_labels})
    train_dataset, test_dataset = train_test_split(train_dataset, train_size=0.9, stratify=train_dataset['label'], random_state=42)
    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)

    return train_dataset, test_dataset 

def train_nb(train_dataset, test_dataset):
    # Why not help me fit a NaiveBayes to classify the text to label
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, accuracy_score

    # Create a CountVectorizer to convert text to numerical features
    vectorizer = CountVectorizer()

    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train_dataset["text"])
    y_train = train_dataset["label"]

    # Transform the test data
    X_test = vectorizer.transform(test_dataset["text"])
    y_test = test_dataset["label"]

    # Initialize and train the Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = nb_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Agent", "Customer"]))

    return vectorizer, nb_classifier




from torch.utils.data import Dataset 
import torch 

class TextClassificationDataset_v2(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    


def train_em(train_dataset, test_dataset, num_epochs=2):
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    import torch 
    from sklearn.metrics import accuracy_score, classification_report

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Torch Dataset 
    train_dataset_torch = TextClassificationDataset_v2(train_dataset["text"], train_dataset["label"], tokenizer, max_length=128)
    test_dataset_torch = TextClassificationDataset_v2(test_dataset["text"], test_dataset["label"], tokenizer, max_length=128)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Set up training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            texts = batch['text']
            labels = batch['label'].to(device)

            # Tokenize the texts
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                texts = batch['text']
                labels = batch['label'].to(device)

                # Tokenize the texts
                encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)

                outputs = model(input_ids)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["Agent", "Customer"]))

    return tokenizer, model 



def predict_em(sample_text, model, tokenizer, threshold_agent=0, threshold_customer=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tokenize the sample text
    encoded = tokenizer(sample_text, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_confidence = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class].item()

    # Map the predicted class to the corresponding label
    label_map = {0: "Agent", 1: "Customer"}
    thres_map = {0: threshold_agent, 1: threshold_customer}
    if predicted_confidence < thres_map[predicted_class]:
        return "Not Sure", 0
    else:
        predicted_label = label_map[predicted_class]
        return predicted_label, predicted_confidence
    


def pred_nb(text, vectorizer, nb_classifier, threshold_agent=0, threshold_customer=0):
    X_new = vectorizer.transform([text])
    prediction = nb_classifier.predict(X_new)
    probabilities = nb_classifier.predict_proba(X_new)
    confidence = probabilities[0][prediction[0]]

    label_map = {0: "Agent", 1: "Customer"}
    thres_map = {0: threshold_agent, 1: threshold_customer}
    if confidence < thres_map[prediction[0]]:
        return "Not Sure", 0
    else:
        predicted_label = label_map[prediction[0]]
    return predicted_label, confidence
