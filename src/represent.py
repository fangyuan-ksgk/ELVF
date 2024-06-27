from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
from tqdm import tqdm as tqdm 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pyreft 
import transformers
from typing import Dict
import copy
from pyreft.dataset import ReftDataCollator
import datasets
from sklearn import preprocessing


# Function to extract hidden representation form a model
def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n

# tokenize and prepare the input
def prepare_prompt_reft(data, tokenizer):
    """ 
    Current Version looks only at the question, not the answer
    - We are checking what these question means to the model's mind
    """
    system_prompt = data.get('system_prompt', "Follow the instruction closely and provide your answer.")
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": data['prompt']}], 
        tokenize=False)
    return prompt

# def get_hidden_states(prompt, model, tokenizer):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model(**inputs, output_hidden_states=True)
#     return outputs.hidden_states

def get_hidden_states(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_hidden_states=True)
    return [h.cpu() for h in outputs.hidden_states]

def get_specific_layers(hidden_states, layers):
    return [hidden_states[i] for i in layers]

def get_average_of_layers(hidden_states, layers):
    selected_layers = get_specific_layers(hidden_states, layers)
    output = sum(selected_layers) / len(selected_layers)
    return output.squeeze()

def get_specific_positions(hidden_states, pos: str = "f1+l1"):
    assert len(hidden_states.shape) == 2, "Hidden states should be 2D tensor"
    positions = parse_positions(pos)
    return [hidden_states[i] for i in positions]

def get_average_of_positions(hidden_states, positions):
    selected_positions = get_specific_positions(hidden_states, positions)
    return sum(selected_positions) / len(selected_positions)


def get_hidden_embedding(f, trainset, feedback,  avg_layer=[1,2,3], positions="f1+l1"):
    pb = tqdm(total=len(trainset), desc="Running inference to get hidden embeddings...")
    embed_vecs = []
    for data in trainset:
        prompt = prepare_prompt_reft(data, f.tokenizer)
        hidden_states = get_hidden_states(prompt, f.model, f.tokenizer)
        avg_hidden_states = get_average_of_layers(hidden_states, avg_layer)
        embed_vec = get_average_of_positions(avg_hidden_states, positions)
        embed_vecs.append(embed_vec.to(torch.float32).detach().cpu().numpy().tolist())
        pb.update(1)
    pb.close()
    print("Inference done!")
    # Saving Embedding Vector into local file
    file_name = "embed_array_layer_" + ("-").join([str(l) for l in avg_layer]) + "_position_" + positions
    embed_array = np.array(embed_vecs)
    file_path = f"database/{feedback.file_name}/{file_name}.npy"
    np.save(file_path, embed_array)
    return embed_array, file_path


#######################
##  Cluster Analysis ##
#######################

def renormalize_std(embed_vecs, target_std = 10.):
    """ 
    Increase Contrast of Embedding Vectors
    - Target a bigger standard deviation
    """
    return (embed_vecs - embed_vecs.mean()) / embed_vecs.std() * target_std + embed_vecs.mean()

def pca_compress(embed_vecs, n_components=3):
    """ 
    Compress Embedding Vectors using PCA
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(embed_vecs)

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    return pca_data


def visualize_3d(pca_data, labels=None):
    """ 
    Visualize 3D PCA Projection with Cluster Labels
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=labels)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title("Projection of Embedding Vectors to 3D PCA Space")
    plt.show()


def visualize_pca(embed_vecs, n_components=3):
    """ 
    Quick PCA Visualization on 3D compressed vectors
    """
    pca_data = pca_compress(embed_vecs, n_components=n_components)

    # Visualize the PCA projection
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title("Projection of Embedding Vectors to 3D PCA Space")
    plt.show()

def get_cluster_labels(embed_vecs, n_clusters):
    """
    Perform K-means clustering on the embedding vectors and return the cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    res = kmeans.fit(embed_vecs)
    return res.labels_


def get_cluster_labels_v2(embed_vecs, n_clusters):
    """
    Perform K-means clustering on the embedding vectors and return the cluster labels.
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    embed_vecs = min_max_scaler.fit_transform(embed_vecs)
    kcluster = KMeans(n_clusters=n_clusters).fit(embed_vecs) #for 2 Clusters
    return kcluster.labels_

def present_corner(corner, idx):
    try: 
        p_str = f"Prompt: {corner['prompt'][idx]} \n Response: {corner['pred'][idx]} \n Accept: {corner['accept'][idx]} \n Rationale: {corner['rationale'][idx]} \n GT: {corner['gt'][idx]} \n"
    except:
        p_str = f"Prompt: {corner['prompt'][idx]} \n Response: {corner['pred'][idx]} \n Score: {corner['score'][idx]} \n Rationale: {corner['feedback'][idx]} \n GT: {corner['gt'][idx]} \n"
    print(p_str)


########################
### Patch for pyreft ###
########################
IGNORE_INDEX = -100

def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(pad_amount)]
        intervention_locations = [position_list]*num_interventions
    else:
        left_pad_amount = (_first_n - first_n)
        right_pad_amount = (_last_n - last_n)
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
        intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
            [right_intervention_locations]*(num_interventions//2)
    
    return intervention_locations
    
def make_multiple_position_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model, inputs, outputs, 
    positions="f1+l1", num_interventions=1, nonstop=False, share_weights=False
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    first_n, last_n = parse_positions(positions)
    
    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    for i in range(len(inputs)):
        _input = inputs[i]
        _output = outputs[i]
    
        base_prompt = _input
        base_input = base_prompt + _output
        if not nonstop:
            base_input += tokenizer.eos_token
    
        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX

        intervention_locations = get_intervention_locations(
            last_position=base_prompt_length, 
            first_n=first_n, 
            last_n=last_n,
            pad_mode="last",
            num_interventions=num_interventions,
            share_weights=share_weights,
        )

        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append(intervention_locations)
        all_output_ids.append(output_ids)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
    })
        
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)