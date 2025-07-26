# GraphRAG

Graph RAG is a RAG system that structures retrieved information as a graph to better capture relationships between concepts. It uses a language model for generation and a semantic embedder to retrieve relevant documents from a local directory. By organizing information into a knowledge graph, Graph RAG enables more coherent, connection-rich responses, especially for queries involving complex interrelated topics. It also supports graph visualization, helping users explore how concepts are linked within the retrieved context.

```python title="GraphRAG" linenums="1"
from rag_src.Complete_RAG_Pipeline import GraphRAG
from rag_src.llm import SmartLLM
from rag_src.embedder import OpenAIEmbedder

llm = SmartLLM()
embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
graph_rag = GraphRAG(
    llm=llm,
    embedder=embedder,
    docdir="./documents"
)
response = graph_rag.run("What is the relationship between quantum computing and cryptography?")
print(f"Graph RAG Response: {response}")

graph_rag.visualize_graph(output_file="knowledge_graph.html")
```
