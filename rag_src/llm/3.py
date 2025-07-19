# Comparison of AsyncDefaultLLM (with ThreadPoolExecutor) vs DefaultLLM (with asyncio.to_thread)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Union
import asyncio
import time

# ------------------------
# Variant 1: ThreadPoolExecutor
# ------------------------
from concurrent.futures import ThreadPoolExecutor

class AsyncDefaultLLM:
    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 100, max_workers: int = 4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        if torch.cuda.is_available():
            self.model.to("cuda")

    def _generate_sync(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output[len(prompt):].strip()
        for token in ["Question:"]:
            if token in answer:
                answer = answer.split(token)[0].strip()
                break
        return answer

    async def generate(self, query: str, contexts: List[str]) -> str:
        combined_context = "\n".join(contexts)
        prompt = f"Context:\n{combined_context}\n\nQuestion:\n{query}\n\nAnswer:"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._generate_sync, prompt)

# ------------------------
# Variant 2: asyncio.to_thread
# ------------------------
class DefaultLLM:
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
        for token in ["Question:"]:
            if token in answer:
                answer = answer.split(token)[0].strip()
                break
        return answer

    async def generate(self, query: str, contexts: List[str]) -> str:
        return await asyncio.to_thread(self._generate_sync, query, contexts)

# ------------------------
# Test Benchmark
# ------------------------
async def benchmark():
    queries = [
        ("What is the capital of France?", ["France is a country in Europe. It is famous for its culture."]*3),
        ("Why do birds migrate?", ["Birds migrate to find food and better climate."]*3),
    ]

    model1 = AsyncDefaultLLM()
    model2 = DefaultLLM()

    # Time for model 1
    start = time.perf_counter()
    results1 = await asyncio.gather(*(model1.generate(q, c) for q, c in queries))
    print("\n[ThreadPoolExecutor Variant]")
    for r in results1: print(r)
    print("Time:", time.perf_counter() - start)

    # Time for model 2
    start = time.perf_counter()
    results2 = await asyncio.gather(*(model2.generate(q, c) for q, c in queries))
    print("\n[asyncio.to_thread Variant]")
    for r in results2: print(r)
    print("Time:", time.perf_counter() - start)

if __name__ == "__main__":
    asyncio.run(benchmark())
