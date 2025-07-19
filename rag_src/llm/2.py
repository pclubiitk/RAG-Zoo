from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
from base import BaseLLM
import asyncio
import time

class AsyncDefaultLLM(BaseLLM):
    """
    Async Default LLM implementation using Hugging Face Causal LM (e.g., GPT-2).
    It prepends context to the query before generation.
    """
    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 100, max_workers: int = 4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        if torch.cuda.is_available():
            self.model.to("cuda")

    def _generate_sync(self, prompt: str) -> str:
        """Synchronous generation method to be run in thread pool"""
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
        stop_tokens = ["Question:"]
        for token in stop_tokens:
            if token in answer:
                answer = answer.split(token)[0].strip()
                break

        return answer

    async def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        # Combine all context chunks
        combined_context = "\n".join(contexts)
        prompt = f"Context:\n{combined_context}\n\nQuestion:\n{query}\n\nAnswer:"

        # Run the synchronous generation in a thread pool
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(self.executor, self._generate_sync, prompt)
        
        return answer

    def __del__(self):
        """Clean up thread pool on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
            
async def async_main():
    model = AsyncDefaultLLM(model_name="gpt2", max_new_tokens=50)

    query = "What is the capital of France?"
    context = ["France is a country in Europe. It is famous for its culture, food, and landmarks."]

    start_time = time.time()
    answer = await model.generate(query, context)
    end_time = time.time()

    print("Generated Answer:", answer)
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(async_main())