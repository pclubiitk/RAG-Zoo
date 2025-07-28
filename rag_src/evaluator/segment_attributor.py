# rag_src/evaluator/segment_attributor.py

from typing import List, Dict, Any
from rag_src.evaluator.base import BaseEvaluator
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
import asyncio


class SegmentAttributor(BaseEvaluator):
    """
    Identifies exact segments from documents that support the generated answer.
    """

    def __init__(self, llm):
        self.llm = llm

    async def locate_segments(
        self,
        query: str,
        response: str,
        docs: List[TextNode]
    ) -> List[Dict[str, Any]]:
        """
        Extract exact document segments that support the generated answer.
        """
        format_docs = lambda docs: "\n".join(
            f"<doc{i + 1}>:\n{doc.text}\n</doc{i + 1}>" for i, doc in enumerate(docs)
        )

        prompt_template = PromptTemplate(
            "You are a helpful assistant tasked with extracting exact segments from provided documents "
            "that directly support a given answer to a user question.\n\n"
            "User Question:\n{query}\n\n"
            "LLM-Generated Answer:\n{response}\n\n"
            "Documents:\n{documents}\n\n"
            "Return a list of exact text segments from the documents that support the answer."
        )

        prompt = prompt_template.format(
            query=query,
            response=response,
            documents=format_docs(docs),
        )

        if asyncio.iscoroutinefunction(self.llm.generate):
            output = await self.llm.generate(prompt, contexts=[])
        else:
            output = await asyncio.to_thread(self.llm.generate, prompt, contexts=[])
        return {"segments": output}

    # Optional: Make it compatible with BaseEvaluator
    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        return await self.locate_segments(
            query=query,
            response=response,
            docs=[TextNode(text=c) for c in contexts]
        )
