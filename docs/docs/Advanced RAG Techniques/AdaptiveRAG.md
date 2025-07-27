# AdaptiveRAG

Adaptive RAG is a modular Retrieval-Augmented Generation system that enhances query responses by dynamically adjusting its retrieval strategy based on query complexity and context. It integrates a powerful language model for answer generation and uses a semantic embedding model to retrieve relevant information from a local document directory. Adaptive RAG is able to deliver context-rich and accurate answers to diverse and open-ended questions.

```python title="Adaptive RAG" linenums="1"
from rag_src.Complete_RAG_Pipeline import AdaptiveRAG
from rag_src.llm import OpenAILLM
from rag_src.embedder import OpenAIEmbedder
llm = OpenAILLM(model="gpt-4")
embedder = OpenAIEmbedder(model_name="text-embedding-3-small")

adaptive_rag = AdaptiveRAG(
    llm=llm,
    embeddor=embedder,
    docdir="./documents"
)

response = adaptive_rag.run("What are different perspectives on AI ethics?")
print(f"Adaptive RAG Response: {response}")
```
