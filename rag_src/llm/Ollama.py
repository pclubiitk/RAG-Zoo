from typing import List, Union
from llama_index.llms.ollama import Ollama
from rag_src.llm.base import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "mistral"):
        self.llm = Ollama(model=model)

    def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text