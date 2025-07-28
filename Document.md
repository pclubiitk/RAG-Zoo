# Comparative Latency and Performance Analysis: CRAG and CustomRAG Pipelines (Synchronous vs Asynchronous)

This report documents and compares the latency and performance of two retrieval-augmented generation pipelines — CRAG (Corrective Retrieval-Augmented Generation) and CustomRAG — under both asynchronous and synchronous execution modes. It highlights:

- Detailed time breakdowns for each stage in both pipelines

- Optimizations used to improve performance

- Instrumentation strategies for accurate measurement

---

## CRAG Pipeline Latency and Performance Documentation

This section provides a detailed breakdown of the CRAG (Corrective Retrieval-Augmented Generation) pipeline execution, including performance metrics for both synchronous and asynchronous runs.

###  Latency Breakdown (Asynchronous Version)

| Stage                            | Time Taken (s) |
|----------------------------------|----------------|
| Document Loading + Preprocessing | 0.21           |
| Enrichment (47 docs)             | 0.00           |
| Indexing (3 batches)             | 0.77           |
| Retriever Initialization         | 4.39           |
| Query Transformation             | 1.33           |
| Retrieval                        | 0.13           |
| Evaluation + Fallback (Web)      | 3.29           |
| Enrichment (Post Retrieval)      | 0.00           |
| Prompt Preparation               | 0.00           |
| LLM Generation                   | 0.96           |
| **Total Time**                   | **11.08 s**    |

###  Latency Breakdown (Synchronous Version)

| Stage                              | Time Taken (s) |
|------------------------------------|----------------|
| Document Loading + Preprocessing   | 0.25           |
| Enrichment                         | 0.00           |
| Indexing                           | 1.11           |
| Retriever Initialization           | 4.91           |
| Query Transformation               | 1.58           |
| Retrieval                          | 0.15           |
| Evaluation + Fallback (Web)        | 3.24           |
| Enrichment (Post Retrieval)        | 0.00           |
| Prompt Preparation + LLM           | 0.58           |
| **Total Time**                     | **11.82 s**    |

### Instrumentation Summary

**Instrumented Stages:**

- Document Loader  
- Preprocessor  
- Chunker  
- Enricher  
- Embedder  
- Indexer  
- Retriever  
- Evaluator  
- WebRetriever  
- LLM (Prompting + Generation)

### Optimization Techniques Used

- **Embedder**: Implementing GPU-based parallel batching substantially improved embedding computation and can reduce latency during bulk queries.
- **LLM Generation**: Improved LLM's usage efficiency by using async nature of LLMs to generate and execute function calls concurrently.
- **Async Pipeline Composition**: Orchestrated all I/O-bound stages (retrieval, LLM calls, web fallback) using asyncio.gather() to parallelize steps that don’t depend on each other.
-- **Asyncio.to_thread()**: Used asyncio.to_thread() to offload synchronous tasks asynchronously
- **Streaming Generation**: Enabled token-wise output streaming from the LLM to minimize first-token latency.

### Understanding Limited Efficiency Gains in Async Execution

While the CRAG pipeline is designed to take advantage of asynchronous execution, the performance improvement is less dramatic for single-query runs. This is primarily due to the sequential dependency between stages and the lack of parallelizable workload in a single-pass scenario.

- Many stages depend on the output of previous ones, which constrains opportunities for concurrent execution.
- Stages like Retriever Initialization, Web Retrieval, and LLM Generation contribute most to the total time. Since they run one after the other, they limit the overall pipeline speed regardless of async orchestration.
- Although asyncio.gather() enables concurrent execution, it can add minor overhead for tasks that are extremely fast or already sequential, especially when there's only one query in the pipeline.

---

## CustomRAG Pipeline Latency and Performance Documentation

This section presents a detailed breakdown of the CustomRAG pipeline execution, with performance metrics for both asynchronous and synchronous runs, along with instrumentation and optimization

###  Latency Breakdown (Asynchronous Version)

| Stage                            | Time Taken (s) |
|----------------------------------|----------------|
| Document Loading + Preprocessing | 0.19           |
| Enrichment (47 docs)             | 0.00           |
| Indexing (3 batches)             | 0.82           |
| Retriever Initialization         | 4.80           |
| Query Transformation             | 6.29           |
| Retrieval (6 queries)            | 0.20           |
| Generation (6 queries)           | 61.09          |
| **Total Time**                   | **73.39 s**    |

###  Latency Breakdown (Synchronous Version)

| Stage                            | Time Taken (s) |
|----------------------------------|----------------|
| Document Loading + Preprocessing | 0.23           |
| Enrichment (47 docs)             | 0.00           |
| Indexing                         | 1.79           |
| Retriever Initialization         | 5.53           |
| Query Transformation             | 3.02           |
| Retrieval (6 queries)            | 0.46           |
| Generation (6 queries)           | 94.81          |
| **Total Time**                   | **105.84 s**   |

### Instrumentation Summary

**Instrumented Stages:**

- Document Loader  
- Preprocessor  
- Chunker  
- Enricher  
- Embedder  
- Indexer  
- Retriever     
- LLM (Generation)

### Optimization Techniques Used

- **Embedder**: Implementing GPU-based parallel batching substantially improved embedding computation and can reduce latency during bulk queries.
- **LLM Generation**: Improved LLM's usage efficiency by using async nature of LLMs to generate and execute function calls concurrently.
- **Async Pipeline Composition**: Orchestrated all I/O-bound stages (retrieval and LLM calls) using asyncio.gather() to parallelize steps that don’t depend on each other.
- **Streaming Generation**: Enabled token-wise output streaming from the LLM to minimize first-token latency.

### Efficient Async Execution with Multiple Queries in CustomRAG

Unlike the CRAG pipeline, which handles a single query per pass and gains limited performance from async execution due to sequential stage dependencies, the CustomRAG pipeline is designed to handle multiple queries concurrently.

- Multiple queries are retrieved and embedded in parallel, taking full advantage of async-compatible retrievers and enrichers.
- LLM responses to multiple queries are handled in parallel, increasing overall throughput while maintaining low latency for each individual query.
- By leveraging Gemini LLM’s streaming output, responses are delivered token-by-token, enhancing the user experience by reducing the apparent response time.

