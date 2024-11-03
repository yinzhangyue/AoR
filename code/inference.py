# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2023/9/10
from openai import OpenAI
from openai import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import unicodedata

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = OpenAI(api_key="")

# this is also the default, it can be omitted
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class Inference_Model:
    def __init__(self, default_model: str) -> None:
        self.model_name = default_model

    def get_info(self, query: str, System_Prompt: str = "You are a helpful assistant.", messages_list: list = [], n: int = 1):
        if messages_list == []:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": System_Prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                n=n,
            )
        else:
            completion = client.chat.completions.create(model=self.model_name, messages=messages_list)
        message = [completion.choices[_].message for _ in range(n)]
        content = [unicodedata.normalize("NFKC", _message.content) for _message in message]
        return content

    def get_info_openai(self, query: str, System_Prompt: str = "You are a helpful assistant.", messages_list: list = [], n: int = 1):
        if messages_list == []:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": System_Prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                n=n,
            )
        else:
            completion = client.chat.completions.create(model=self.model_name, messages=messages_list)
        message = [completion.choices[_].message for _ in range(n)]
        content = [unicodedata.normalize("NFKC", _message.content) for _message in message]
        return content
