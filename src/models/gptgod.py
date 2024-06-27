# Potentially use GPTGOD to call Chinese LLMs
import requests
from os import getenv

class GGModel:
    BASEURL = "https://api.gptgod.online/v1/chat/completions"
    MODELS = [
        "glm-4",
        "dbrx-instruct"
    ]
    KEY_ENV_VAR = "GPTGOD_API_KEY"
    MAX_TOKENS = 9999

    def __init__(self, model: str = "glm-4"):
        self.model = model
        self.api_key = getenv(self.KEY_ENV_VAR)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    def get_completion(self, system_prompt: str, prompt: str) -> str:
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        response = requests.post(self.BASEURL, headers=self.headers, json=data)
        return response.json()