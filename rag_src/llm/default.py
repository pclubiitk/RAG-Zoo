from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Union
from .base import BaseLLM

class DefaultLLM(BaseLLM):
    """
    Default LLM implementation using Hugging Face Causal LM (e.g., GPT-2).
    It prepends context to the query before generation.
    """
    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

        if torch.cuda.is_available():
            self.model.to("cuda")

    def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        # Combine all context chunks
        combined_context = "\n".join(contexts)
        prompt = f"Context:\n{combined_context}\n\nQuestion:\n{query}\n\nAnswer:"

        # Tokenize and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

        # Decode and strip prompt portion
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output[len(prompt):].strip()

        return answer