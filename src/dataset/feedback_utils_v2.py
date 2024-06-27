import re
import os
import json
from enum import Enum
from uuid import uuid5, UUID
from typing import Optional, Any, Callable
import datasets

# Used to generate deterministic UUIDs for feedback
NAMESPACE_UUID = UUID("00000000-0000-0000-0000-000000000000")

# First Principle: ICL response is already good enough | If ICL is good enough, it's all about prompt-completion and error cases
# Self-supervision loss based on a verbal feedback: Loss(Prompted response, FineTuned Response)
# -- Question is how to compress the prompt into the model --> REFT
# -- Question is how to compress REFT into the mode --> Fine-Tuning 

class Feedback:
    content: str
    prompts: list # Places where feedback apply
    search_infos: dict # Search Information

    def __init__(self, content: str):
        self.content = content
        try:
            self.load_info()
            print("Loaded {} prompts".format(len(self.prompts)))
            print("Loaded {} search infos".format(len(self.search_infos)))
        except:
            print("Completion Information not found.")

    @property
    def id(self):
        return uuid5(NAMESPACE_UUID, self.content)
    
    @property
    def file_name(self):
        assert self.id is not None, "Feedback must have an ID to have a file name"
        content = self.content.lower()[:30]
        content = re.sub(r"[^a-z0-9 ]", " ", content)
        content = re.sub(r" +", " ", content)
        content = content.replace(" ", "_")
        content = content.strip()
        return f"{content}_{self.id}"
    
    def load_info(self):
        with open(f"database/{self.file_name}/prompts.json", "r") as f:
            prompts = json.load(f)

        with open(f"database/{self.file_name}/search_infos.json", "r") as f:  
            search_infos = json.load(f)

        with open(f"database/{self.file_name}/test_dataset.json", "r") as f:  
            test_cases = json.load(f)

        self.prompts = prompts
        self.search_infos = search_infos
        self.test_cases = test_cases
        return

    def save_info(self):
        os.makedirs(f"database/{self.file_name}", exist_ok=True)
        
        with open(f"database/{self.file_name}/prompts.json", "w") as f:
            json.dump(self.prompts, f)

        with open(f"database/{self.file_name}/search_infos.json", "w") as f:  
            json.dump(self.search_infos, f)
        return

    def update_feedback_search_completion(self):
        search_infos = {}
        for prompt in self.prompts:
            # Get completion file -- There are bunch of rejected completions, and accepted completions
            get_prompt_complete_id = lambda prompt: "search_info_"+prompt.replace(" ","-").replace(".","")
            file_name = get_prompt_complete_id(prompt)
            file_path = f"database/{self.file_name}/{file_name}.json"
            with open(file_path, "r") as f:
                search_info = json.load(f)
            search_infos[prompt] = search_info
            os.remove(file_path) # Remove File
        self.search_infos = search_infos

    def boostrap_augment(self, aug_name="aug_r1"):
        """ 
        Concatenate Extra augmented prompts into the feedback object
        """
        try:
            dest_dir = f'database/{self.file_name}/{aug_name}'
            import glob, json
            aug_prompts = []
            aug_search_infos = {}
            for file in glob.glob(dest_dir + "/*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                prompt = data[0]['prompt']
                prompt_id = file.split(".json")[0].split("search_info_")[-1]
                aug_search_infos[prompt_id] = data
                aug_prompts.append(prompt)

            self.prompts = self.prompts + aug_prompts
            self.search_infos.update(aug_search_infos) # Update the search info as well
            print("Augmenting {} prompts".format(len(aug_prompts)))
            print("Database Updated!")
        except:
            print(f"Augmentation file {aug_name} not found.")