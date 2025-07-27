# Generation

The response generator takes the query and enriched retrieved data to produce a coherent and contextually accurate final answer. The Post Retrieval Enricher helps enrich the data retrieved.

We'll see about using the [post_retrieval_enricher](../Module-By-Module%20Deep%20Dive/post_retrieval_enricher.md) and [llm](../Module-By-Module%20Deep%20Dive/llm.md) modules here.

```python title="PostRetrieval Enrichnment and Generation with LLM" linenums="1"
from rag_src.post_retrieval_enricher import DocSummarizer
from rag_src.llm import OpenAILLM, GroqLLM

retrieved_texts = [doc['text'] for doc in reranked_docs]
query = "What are the components of a RAG system?"
llm = OpenAILLM()
groq_llm = GroqLLM(model="llama3-8b-8192")

summarizer = DocSummarizer(llm=llm)
summarized_docs = summarizer.enrich(retrieved_texts)
print("Summarized Context:")
print(summarized_docs[0], end="\n\n")

response = groq_llm.generate(query=query, contexts=summarized_docs)
print("Final Generated Answer: ")
print(response)
```
