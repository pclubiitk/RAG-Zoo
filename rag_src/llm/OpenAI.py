from typing import List, Union
from llama_index.llms.openai import OpenAI
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4"):
        self.llm = OpenAI(model=model)

    def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text
