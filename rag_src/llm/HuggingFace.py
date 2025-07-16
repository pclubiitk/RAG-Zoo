from typing import List, Union
from llama_index.llms.huggingface import HuggingFaceLLM
from .base import BaseLLM


class HuggingFaceLLMWrapper(BaseLLM):
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-alpha"):
        self.llm = HuggingFaceLLM(model_name=model_name)

    def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text
