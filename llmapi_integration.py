# Integrating API interface of GPT and DeepSeek to switch models conveniently

import time
import openai
import json


# Integrated API interface
class LLMAPI:

    # Initialize the API with the given model and API key
    def __init__(self, model: str, api_key: str):
        self.model = model.lower()
        self.api_key = api_key
        self.client = None
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com" if self.model == 'deepseek' else None
        )

    # Call the API with the given dialogue
    def call(self, dialogue: list[str], first_is_system: bool = False) -> dict:
        # Convert the dialogue to the format required by the API
        messages: list[dict] = []
        if first_is_system:
            messages.append({"role": "system", "content": dialogue[0]})
            dialogue = dialogue[1:]
        for i, utterance in enumerate(dialogue):
            role = 'assistant' if i % 2 == len(dialogue) % 2 else 'user'
            messages.append({"role": role, "content": utterance})
        print(messages)

        # Call the API and return the response
        if self.model == 'gpt':
            return self._call_openai(messages)
        elif self.model == 'deepseek':
            return self._call_deepseek(messages)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    # Call the OpenAI API with the given messages
    def _call_openai(self, messages) -> dict:
        start = time.time()
        response = self.client.responses.create(
            model="gpt-4.1",
            input=messages
        )
        end = time.time()
        return {
            "response_text": response.output[0].content[0].text,
            "tokens": response.usage.total_tokens,
            "duration": end - start
        }

    # Call the DeepSeek API with the given messages
    def _call_deepseek(self, messages) -> dict:
        start = time.time()
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        end = time.time()
        return {
            "response_text": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
            "duration": end - start
        }


if __name__ == "__main__":
    with open('api_key.json', 'r', encoding='utf-8') as f:
        api_keys = json.load(f)
    model_name = 'deepseek'
    gpt1 = LLMAPI(model_name, api_keys.get(model_name))
    print(gpt1.call(["What is the capital of France?"]))
