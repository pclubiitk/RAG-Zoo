# Embedding And Indexing

Next we'll see how about seeing how our modules help us go about embedding and indexing

Embedding and indexing involves converting documents into vector representations and organizing them for efficient retrieval

Our [embedder](../Module-By-Module%20Deep%20Dive/embedder.md) and [indexer](../Module-By-Module%20Deep%20Dive/indexer.md) modules help us here

```python title="Embedding" linenums="1"
from rag_src.embedder import OpenAIEmbedder

embedder = OpenAIEmbedder(model_name = "text-embedding-3-small") 
embeddings = embedder.embed(texts=chunks)

print(f"Generated {len(embeddings)} embeddings.")
```
```python title="Storing Embedded data with Vector Space" linenums="1"
from rag_src.indexer import ChromaDBIndexer

indexer = ChromaDBIndexer(collection_name="rag_documents", persist_directory="./chroma_index")
indexer.index(embeddings=embeddings, documents=chunks)
indexer.persist()

print(f"Successfully indexed {len(chunks)} chunks in ChromaDB.")
```
