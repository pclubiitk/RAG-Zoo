# Layout of the Repo:

```title="Structure of the Repository"
RAG-Zoo/
├── rag_src/                     # core package
│   ├── Complete_RAG_Pipeline/   # pipelines
│   ├── chunker/                 # text splitting
│   ├── embedder/                # embedding 
│   ├── indexer/                 # vector stores
│   ├── retriever/               # retrieval 
│   ├── llm/                     # LLM adapters
│   ├── doc_loader/              # loads docs
│   ├── doc_preprocessor/        # preprocesses docs
│   ├── pre_embedding_enricher/  # Enriches query
│   ├── post_retrieval_enricher/ # Enricher prompt
│   ├── query_transformer/       # query transformer
│   ├── retriever/               # retrieves from vector store
│   ├── web_retriever/           # retrieves data from web
├── tests/                       # pytest suites
├── docs/                        # documentation site
├── pyproject.toml               # Poetry config
└── README.md                    # README
```
