# Evaluation

An evaluator assesses the quality of retrieved documents or final answers, using certain metrics

The single component dedicated to measuring the system's effectiveness - [evaluator](../Module-By-Module%20Deep%20Dive/evaluator.md)
```python title="LLM as a Judge" linenums="1"
from rag_src.evaluator import RelevanceEvaluator
from rag_src.llm import OpenAILLM

query = "What are the components of a RAG system?"
response = "A RAG system has three main components: a data preparation pipeline, a retrieval mechanism, and a generation model." #Example

eval_llm = OpenAILLM()
relevance_evaluator = RelevanceEvaluator(llm=eval_llm, threshold=0.7)
relevance_data = relevance_evaluator.evaluate(
    query=query,
    response=response,
    contexts=contexts
)

print("Relevance Evaluation- ")
print(f"Relevance Score: {relevance_data['relevance_score']}")
print(f"Above Threshold?: {relevance_data['above_threshold']}")
```
