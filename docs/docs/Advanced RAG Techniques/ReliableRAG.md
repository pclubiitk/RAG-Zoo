# ReliableRAG

Reliable RAG focuses on producing trustworthy, fact based responses by combining semantic retrieval with robust generation. It retrieves high-relevance content from a local document directory using a semantic embedder and then formulates answers through a language model. Designed with reliability in mind, it incorporates consistency checks to minimize hallucinations, making it ideal for use cases that demand factual accuracy, such as scientific and technical domains.

```python title="Reliable RAG" linenums="1"
from rag_src.Complete_RAG_Pipeline import ReliableRAG
from rag_src.llm import OpenAILLM
from rag_src.embedder import OpenAIEmbedder

llm = OpenAILLM(model="gpt-4")
embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
reliable_rag = ReliableRAG(
    llm=llm,
    embeddor=embedder,
    docdir="./documents"
)

response = reliable_rag.run("Explain the principles of quantum entanglement")
print(f"Reliable RAG Response: {response}")
```
