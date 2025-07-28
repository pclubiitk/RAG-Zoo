from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Union
import asyncio
from .base import BaseLLM

class DefaultLLM(BaseLLM):
    """
    Async Default LLM implementation using Hugging Face Causal LM (e.g., GPT-2).
    It prepends context to the query before generation.
    """
    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

        if torch.cuda.is_available():
            self.model.to("cuda")

    def _generate_sync(self, query: str, contexts: List[str]) -> str:
        combined_context = "\n".join(contexts)
        prompt = f"Context:\n{combined_context}\n\nQuestion:\n{query}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output[len(prompt):].strip()
        stop_tokens = ["Question:"]
        for token in stop_tokens:
            if token in answer:
                answer = answer.split(token)[0].strip()
                break

        return answer

    async def generate(self, query: str, contexts: List[str]) -> str:
        return await asyncio.to_thread(self._generate_sync, query, contexts)