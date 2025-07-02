from .base import BaseLLM
from llama_index.core.llms import LLM
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
)
from llama_index.core.prompts import PromptTemplate
from typing import List, AsyncGenerator

import os

from .OpenAI import OpenAILLM
from .groq import GroqLLM
from .gemini import GeminiLLM


class SmartLLM(BaseLLM, LLM):
    def __init__(self):
        super().__init__()  # ✅ important for Pydantic
        object.__setattr__(self, "llm", self._init_llm())  # ✅ bypass Pydantic restrictions

    def _init_llm(self):
        if os.getenv("GEMINI_API_KEY"):
            print("[SmartLLM] Using Gemini LLM")
            return GeminiLLM(os.getenv("GEMINI_API_KEY"))
        elif os.getenv("GROQ_API_KEY"):
            print("[SmartLLM] Using Groq LLM")
            return GroqLLM(os.getenv("GROQ_API_KEY"))
        elif os.getenv("OPENAI_API_KEY"):
            print("[SmartLLM] Using OpenAI LLM")
            return OpenAILLM(os.getenv("OPENAI_API_KEY"))
        else:
            raise EnvironmentError(
                "No valid LLM API key found. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY."
            )

    def _extract_text(self, output) -> str:
        return output.text if hasattr(output, "text") else str(output)

    def generate(self, query: str, contexts: List[str]) -> str:
        return self._extract_text(self.llm.generate(query, contexts))

    def predict(self, prompt_template: PromptTemplate, **kwargs) -> str:
        prompt = prompt_template.format(**kwargs)
        return self._extract_text(self.llm.generate(prompt, contexts=[]))

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        text = self._extract_text(self.llm.generate(prompt, contexts=[]))
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        raise NotImplementedError("SmartLLM does not support stream_complete().")

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        raise NotImplementedError("SmartLLM does not support chat().")

    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponseGen:
        raise NotImplementedError("SmartLLM does not support stream_chat().")

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        raise NotImplementedError("SmartLLM does not support acomplete().")

    async def astream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[CompletionResponse, None]:
        raise NotImplementedError("SmartLLM does not support astream_complete().")

    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        raise NotImplementedError("SmartLLM does not support achat().")

    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError("SmartLLM does not support astream_chat().")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=512,
            is_chat_model=False,
            is_function_calling_model=False,
            model_name="SmartLLM"
        )

    # ✅ Allow arbitrary attribute assignment for llm
    model_config = {
        "extra": "allow"
    }
