# Quick Start Guide

## Your First RAG Pipeline!

This sets up a default linear pipeline: documents are automatically loaded, chunked, embedded, indexed, and retrieved.  
Then an LLM generates an answer based on the top results.

```python title="Set up and run your first RAG code!"
from rag_src.Complete_RAG_Pipeline import RunRAG
from rag_src.embedder import OpenAIEmbedder
from rag_src.retriever import HybridRetriever
from rag_src.llm import OllamaLLM

pipeline = RunRAG(docdir = "./docs_pdf",
  embedder = OpenAIEmbedder(),
  retriever = HybridRetriever(),
  llm = OllamaLLM()
)

answer = pipeline.run("What is the main topic of these documents?")
print(answer)
```

## Swap ANY Module!

One core idea of this package is modularity and ability to change and test specific components of our choice. 

eg:
```python title="Swapping Module" linenums="1"
from rag_src.doc_loader import UniversalDocLoader
from rag_src.doc_preprocessor import AdvancedPreprocessor
from rag_src.chunker import SemanticChunker
from rag_src.embedder import GeminiEmbedder

# Load & preprocess documents
loader = UniversalDocLoader(path = "./docs_txt")
loaded_docs = loader.load()

preprocessed_docs = AdvancedPreprocessor().preprocess(loaded_docs)

# Chunk text
chunker = SemanticChunker(chunk_size = 300)
chunks = chunker.chunk(preprocessed_docs)

# Embed using Gemini
embedder = GeminiEmbedder()
embeddings = embedder.embed(chunks)

print(f"First embedding: {embeddings[0]}")
```
You can use your OWN COMBINATION - any chunker, any embedder, any vectorstore, etc.

## Injecting Metadata and Context enrichment

Here’s how to attach metadata and apply post-retrieval reranking.

```python title="Adding Metadata" linenums="1"
from rag_src.pre_embedding_enricher import MetadataInjector
from rag_src.post_retrival_enricher import SelfRerank
from rag_src.llm import OpenAILLM

# Add metadata tags to docs
metadata = {0: {"title": "My Document Title", "author": "Alice"}}
enricher = MetadataInjector(metadata)
docs = enricher.enrich(loaded_docs) # load your docs before enriching it

# After retrieval, rerank top results
reranker = SelfRerank(llm=OpenAILLM())
reranked_docs = reranker.enrich(docs)

print(reranked_docs)
```

## Improving your Queries with our Query Transformation Module

Query Transformers boost your pipeline’s relevance by rewriting, decomposing, or expanding queries.

eg:

```python title="Query Transformation" linenums="1"
from rag_src.query_transformer import QueryDecomposer, MultiQuery
from rag_src.llm import OpenAILLM

# Decompose a complex query into simpler parts
decomposer = QueryDecomposer(llm=OpenAILLM())
sub_queries = decomposer.transform("Explain the pros and cons of quantum computing")
print(sub_queries)

# Or generate multiple rephrasings for robust retrieval
multi = MultiQuery(llm=OpenAILLM())
alt_queries = multi.transform("How does quantum computing work?", n=3)
print(alt_queries)
```

## Evaluators - check how grounded your answers are with your source

Our evaluator modules equip you with the tools to evaluate how relevant the generated response is to the original query using the provided contexts.

eg:
```python title="Judging Answer Quality" linenums="1"
from rag_src.evaluator import RelevanceEvaluator, SegmentAttributor
from rag_src.llm import OpenAILLM

query = "What is quantum computing?"
response = "Quantum computing uses qubits to..."
contexts = ["Quantum computing is based on qubits...", "Second context..."]

# Relevance check
evaluator = RelevanceEvaluator(llm=OpenAILLM())
result = evaluator.evaluate(query, response, contexts)
print(result)

# Find which document segments support the answer
segmenter = SegmentAttributor(llm=OpenAILLM())
segments = segmenter.locate_segments(query, response, contexts)
print(segments)
```

We'll next move on to component wise guides

