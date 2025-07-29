import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List
class CatcheRAG:
    def __init__(
        self,
        context_documents: Optional[List[str]] = None,
        context_path: Optional[str] = None,
        model_name: str = "gpt2",
        device: str = "cpu",
        max_context_length: int = 1024
    ):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        #context corpus
        if context_documents:
            self.context = "\n".join(context_documents)
        elif context_path and os.path.exists(context_path):
            with open(context_path, "r", encoding="utf-8") as f:
                self.context = f.read()
        else:
            raise ValueError("Provide either context_documents or a valid context_path.")

        # Preload
        self.kv_cache = self._encode_context_to_cache(max_context_length)
    def _encode_context_to_cache(self, max_length: int):
        """
        CKV ← KV‑Encode(D)
        Preload document corpus into model’s attention cache.
        """
        inputs = self.tokenizer(
            self.context,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llm(
                **inputs,
                use_cache=True,
                return_dict=True
            )

        return outputs.past_key_values
    def run(self, query: str, max_new_tokens: int = 128) -> str:
        """
        Run CAG: Use cached context KV and generate response using full query by feeding tokens step-by-step.
        """
        # Tokenize the query
        query_inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        input_ids = query_inputs["input_ids"]

        past_key_values = self.kv_cache
        all_generated = []

        with torch.no_grad():
            for i in range(input_ids.shape[1]):
                # Feed one token at a time
                input_token = input_ids[:, i].unsqueeze(-1)
                outputs = self.llm(
                    input_ids=input_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
                all_generated.append(input_token)

            # Now generate new tokens autoregressively
            for _ in range(max_new_tokens):
                outputs = self.llm(
                    input_ids=all_generated[-1],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values
                all_generated.append(next_token)

        # Stack all tokens together
        output_ids = torch.cat(all_generated, dim=-1)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def reset_cache(self, new_documents: Optional[List[str]] = None, new_path: Optional[str] = None):
        """
        Optional Stage 3: Reset CKV with new documents.
        """
        if new_documents:
            self.context = "\n".join(new_documents)
        elif new_path and os.path.exists(new_path):
            with open(new_path, "r", encoding="utf-8") as f:
                self.context = f.read()
        else:
            raise ValueError("Provide new_documents or valid new_path to reset.")

        self.kv_cache = self._encode_context_to_cache(max_length=1024)