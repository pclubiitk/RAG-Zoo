# Retrieval

So this is us getting the necessary data from the pipeline to be fed into the LLM as extra context. 
Retrieval fetches relevant data either from vector databases using similarity search or from the web using keyword-based search
We use the web_retriever and [retriever](../Module-By-Module%20Deep%20Dive/retriever.md) here for the same.

```python title="Retrieving Data from the Web" linenums="1"
    from rag_src.web_retriever import DuckDuckGoWebRetriever

    query = "What is Retrieval-Augmented Generation?"

    retriever = DuckDuckGoWebRetriever(max_results=5)
    results = retriever.retrieve(query)
    text = "\n\n".join(textnode.text for textnode in results)

    print("DuckDuckGo Web Results")
    print(text)
```
```python title="Reranking data in the index" linenums="1"
from rag_src.retriever import ReRankingRetriever

query = "What are the components of a RAG system?"
reranking_retriever = ReRankingRetriever(
    index_path="./chroma_index",
    initial_top_n=20
)
reranked_docs = reranking_retriever.retrieve(query, k=5)
print("Showing reranked docs-")
for doc in reranked_docs:
    print(f"Text:\n{doc['text']}\n")
```

