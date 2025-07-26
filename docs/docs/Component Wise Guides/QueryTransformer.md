# Query Transformer

In case of RAG techniques having good queries can make our results MUCH BETTER. 
Query transformation rewrites or reformulates the user’s query to improve retrieval accuracy and relevance
That's exactly what the [query_transformer](../Module-By-Module%20Deep%20Dive/query_transformer.md) module does

```python title="Query Transformation" linenums="1"
from rag_src.query_transformer import LLMWebQueryTransformer
from rag_src.llm import OpenAILLM

llm = OpenAILLM()

web_transformer = LLMWebQueryTransformer(llm = llm)
web_search_query = web_transformer.transform("What were the latest advancements in AI in 2024?")
print("\nWeb Search Optimized Query - ")
print(web_search_query[0])
```
