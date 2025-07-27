# Data Preparation

We will be looking at the components related to preparing data to get raw data ready for the system.

Preparing to use raw data generally consists of 3 main processes - loading the data from the documents we have, processing the data to filter out unnecessary parts and optionally enriching the data.

Our [doc_loader](../Module-By-Module%20Deep%20Dive/doc_loader.md) , [doc_preprocessor](../Module-By-Module%20Deep%20Dive/doc_preprocessor.md) , [chunker](../Module-By-Module%20Deep%20Dive/chunker.md) and [pre_embedding_enricher](../Module-By-Module%20Deep%20Dive/pre_embedding_enricher.md) modules help us accomplish this.

```python title="Loading and Preprocessing Data" linenums="1"
from rag_src.doc_loader import UniversalDocLoader, UncommonDocLoader
from rag_src.doc_preprocessor import AdvancedPreprocessor

universal_loader = UniversalDocLoader(path="./sample.txt")
common_docs = universal_loader.load()

uncommon_loader = UncommonDocLoader(path="./products.csv")
uncommon_docs = uncommon_loader.load()

all_docs = common_docs + uncommon_docs
print(f"Got {len(all_docs)} docs.")

# Preprocessing all docs
preprocessor = AdvancedPreprocessor(remove_stopwords=True)
processed_docs = preprocessor.preprocess(all_docs)

print("Sample doc - ")
print(processed_docs[0])
```
```python title="PreEmbedding Enrichment" linenums="1"
from rag_src.pre_embedding_enricher import QAPairGenerator
from rag_src.llm import OpenAILLM

qa_generator = QAPairGenerator(llm=OpenAILLM())
qa_docs = qa_generator.enrich(processed_docs)
print("\n--- Enriched with QA Pairs ---")
print(qa_docs[0])
```
```python title="Chunking Data" linenums="1"
from rag_src.chunker import RecursiveChunker

final_docs = qa_docs

chunker = RecursiveChunker()
chunks = chunker.chunk(final_docs)
print(f"Created {len(chunks)} chunks using RecursiveChunker.")
```
