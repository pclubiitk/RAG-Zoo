from typing import List, Union
from llama_index.llms.groq import Groq
from .base import BaseLLM

class GroqLLM(BaseLLM):
    def __init__(self, model: str = "mixtral-8x7b-32768", api_key: str = None):
        self.llm = Groq(model=model, api_key=api_key)

    def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text
