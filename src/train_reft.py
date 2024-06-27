from transformers import HfArgumentParser
import json, pyreft, transformers, torch, argparse
from huggingface_hub import login, create_repo
from os import getenv

from src.dataset.feedback_utils_v2 import Feedback
from src.dataset.format_v2 import to_distill_sft
from src.utils_v2 import ModelArguments, ReftArguments, arg_to_repo
from src.represent import make_multiple_position_supervised_data_module

login(
  token=getenv("HF_TOKEN"), # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_reft(arg_file, dataset, repo_id):
    """ 
    Function to train a REFT model and save it to the Hugging Face Hub
    """
    with open(arg_file, "r") as f:
        arg_dict = json.load(f)

    ##############
    # Load Model # 
    ##############
    model_arg_parser = HfArgumentParser((ModelArguments,))
    model_args: ModelArguments = model_arg_parser.parse_dict(arg_dict["model_args"])[0]
    model, tokenizer = model_args.make()

    ###################### 
    # Load ReFT Argument #
    ######################
    reft_args = HfArgumentParser((ReftArguments,)).parse_dict(arg_dict["reft_args"])[0]
    reft_config = reft_args.make_config(model)

    ###################
    # Form ReFT Model #
    ###################
    import pyreft 
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device("cuda")
    reft_model.print_trainable_parameters()

    ###############
    # Data Module # 
    ###############

    system_prompt = "Follow the instruction closely and provide your answer."

    query_list = [tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": data['prompt']}
            ], tokenize=False
    ) for data in dataset]

    answer_list = [
            tokenizer.apply_chat_template(
                [{"role": "assistant", "content": data['completion']}], tokenize=False,
            )[len(tokenizer.bos_token):] for data in dataset
    ]

    data_module = make_multiple_position_supervised_data_module(
        tokenizer, model, query_list, answer_list, 
        positions=reft_args.intervention_positions, num_interventions=len(reft_config.representations), share_weights=reft_args.share_weights, nonstop=False)

    ################
    # Train & Save #
    ################

    training_args = transformers.TrainingArguments(
        num_train_epochs=50.0, output_dir="./tmp", 
        per_device_train_batch_size=10, 
        learning_rate=4e-3, report_to=[], logging_steps=20)

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model, tokenizer=tokenizer,
        args=training_args, **data_module)
    _ = trainer.train()

    # Name of the new repository
    create_repo(repo_id, private=False)

    reft_model.set_device("cpu") # send back to cpu before saving.
    reft_model.save(
        save_directory="./reft_to_share", 
        save_to_hf_hub=True, 
        hf_repo_name=f"Ksgk-fy/{repo_id}"
    )

    return 



# def run_reft_logit_prediction(dataset, tokenizer, reft_model):
#     """ 
#     Obtain Logit prediction from REFT adapted model
#     """
#     pd = tqdm(total=len(dataset), position=0, leave=False)
#     prompts = []
#     preds = []
#     for data in dataset:
#         # tokenize and prepare the input
#         prompt = tokenizer.apply_chat_template(
#             [{"role": "system", "content": system_prompt}, {"role": "user", "content": data['prompt']}], 
#             tokenize=False)
#         prompt = tokenizer(prompt, return_tensors="pt").to(device)
        
#         unit_locations = torch.IntTensor([pyreft.get_intervention_locations(
#             last_position=prompt["input_ids"].shape[-1], 
#             first_n=first_n, 
#             last_n=last_n,
#             pad_mode="last",
#             num_interventions=len(reft_config.representations),
#             share_weights=share_weights
#         )]).permute(1, 0, 2).tolist()

#         _, reft_prediction = reft_model(prompt, unit_locations={"sources->base": (None, unit_locations)})
#         pred_logits = reft_prediction.logits

#         prompts.append(prompt)
#         preds.append(pred_logits.detach().cpu().tolist())
#         pd.update(1)
#     return prompts, preds


# Save the prompts and predictions in a DataFrame and CSV
# prompts, preds = run_reft_inference(dataset, tokenizer, reft_model)
# import pandas as pd
# df = pd.DataFrame({"prompt": prompts, "prediction": preds})
# df.to_csv("inference_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg_file", type=str, default="configs/config_reft_v1.json")
    repo_id = parser.arg_file.split("/config_")[-1].replace(".json", "_elvf")
    args = parser.parse_args()

    feedback = Feedback(content = "Do not talk about elephant")
    dataset = to_distill_sft(feedback)["train"]

    train_reft(args.arg_file, dataset, repo_id)