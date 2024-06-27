# Simplified Version
from openai import OpenAI
from os import getenv

class OpenAIModel:
    MODELS = [
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-3.5-turbo-1106",
        "gpt-4-0125-preview",
        "gpt-4-turbo",
    ]
    KEY_ENV_VAR = "OPENAI_API_KEY"

    def __init__(self, model_name: str):
        api_key = getenv(self.KEY_ENV_VAR)
        self.client = OpenAI(
            api_key=api_key,
            base_url=getattr(self, "BASEURL", None),
            max_retries=2)
        self.model_name = model_name

    # This is not that general | one system prompt + one user input
    def get_completion(self, system_prompt: str, prompt: str, stop=[]) -> str:
        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stop=stop
        )
        return completion.choices[0].message.content
