from os import getenv
from openai import OpenAI

# Simpler Version
class OpenRouterModel:
    BASEURL = "https://openrouter.ai/api/v1"
    MODELS = [
        "qwen/qwen-110b-chat", 
        "mistralai/mistral-large", 
        "meta-llama/llama-3-70b-instruct:nitro", 
        "01-ai/yi-34b-chat", 
        "cohere/command-r-plus", 
        "anthropic/claude-3-opus", 
        "microsoft/wizardlm-2-8x22b"
    ]
    KEY_ENV_VAR = "OPENROUTER_API_KEY"
    MAX_TOKENS = 200
    def __init__(self):
        self.client = OpenAI(
            base_url=self.BASEURL,
            api_key=getenv(self.KEY_ENV_VAR),
        )
    def get_completion(self, system_prompt: str, prompt: str, idx: int = 1) -> str:
        completion = self.client.chat.completions.create(
            model = self.MODELS[idx],
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
        )
        return completion.choices[0].message.content



