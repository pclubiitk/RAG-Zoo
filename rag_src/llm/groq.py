from typing import List, Union
from llama_index.llms.groq import Groq
from .base import BaseLLM


class GroqLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "llama3-8b-8192"):
        self.llm = Groq(model=model, api_key=api_key)

    def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text
