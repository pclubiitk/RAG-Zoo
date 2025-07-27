# rag_src/complete_RAG_Pipeline/RAPTOR.py

from typing import List, Optional, Dict
from langchain.schema import AIMessage
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd


from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


from rag_src.llm import BaseLLM, DefaultLLM
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import BaseQueryTransformer, DefaultQueryTransformer
from rag_src.post_retrival_enricher import PostBaseEnricher, PostDefaultEnricher
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor
from rag_src.chunker import BaseChunker, DefaultChunker


class RAPTOR:
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embedder: Optional[BaseEmbedder] = None,
        query_transform: Optional[BaseQueryTransformer] = None,
        doc_loader: Optional[BaseDocLoader] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        chunker: Optional[BaseChunker] = None,
        doc_enricher: Optional[PostBaseEnricher] = None,
        docdir: str = "data",
    ):
        self.docdir = docdir
        self.llm = llm or DefaultLLM()
        self.embedder = embedder or DefaultEmbedder()
        self.query_transform = query_transform or DefaultQueryTransformer()
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()
        self.doc_enricher = doc_enricher or PostDefaultEnricher()

        self.texts = self.load_document()
        self.max_levels = 3
        self.tree_results = self.build_raptor_tree()

    def load_document(self):
        documents = self.doc_loader.load()
        documents = self.preprocessor.preprocess(documents)
        p_texts = [doc.page_content for doc in documents]
        chunk_p_texts = self.chunker.chunk(p_texts)
        return chunk_p_texts

    def extract_text(self, item):
        """Extract text content from either a string or an AIMessage object."""
        if isinstance(item, AIMessage):
            return item.content
        return item

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the configured embedder."""
        return self.embedder.embed([self.extract_text(text) for text in texts])

    def perform_clustering(
        self, embeddings: np.ndarray, n_clusters: int = 10
    ) -> np.ndarray:
        """Perform clustering on embeddings using Gaussian Mixture Model."""
        gm = GaussianMixture(n_components=n_clusters, random_state=42)
        return gm.fit_predict(embeddings)

    def summarize_texts(self, texts: List[str]) -> str:
        """Summarize a list of texts."""
        prompt = PromptTemplate.from_template(
            "Summarize the following text concisely:\n\n{text}"
        )
        combined_text = "\n\n".join(texts)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run({"text": combined_text})

    def create_retriever(self, vectorstore: FAISS) -> ContextualCompressionRetriever:
        """Create a retriever with contextual compression."""
        base_retriever = vectorstore.as_retriever()

        prompt = PromptTemplate.from_template(
            "Given the following context and question, extract only the relevant information for answering the question:\n\n"
            "Context: {context}\n"
            "Question: {question}\n\n"
            "Relevant Information:"
        )

        extractor = LLMChainExtractor.from_llm(self.llm, prompt=prompt)
        return ContextualCompressionRetriever(
            base_compressor=extractor, base_retriever=base_retriever
        )

    def build_vectorstore(self, tree_results: Dict[int, pd.DataFrame]) -> FAISS:
        """Build a FAISS vectorstore from all texts in the RAPTOR tree."""
        all_texts = []
        all_embeddings = []
        all_metadatas = []

        for level, df in tree_results.items():
            all_texts.extend([str(text) for text in df["text"].tolist()])
            all_embeddings.extend(
                [
                    (
                        embedding.tolist()
                        if isinstance(embedding, np.ndarray)
                        else embedding
                    )
                    for embedding in df["embedding"].tolist()
                ]
            )
            all_metadatas.extend(df["metadata"].tolist())

        documents = [
            Document(page_content=str(text), metadata=metadata)
            for text, metadata in zip(all_texts, all_metadatas)
        ]
        return FAISS.from_documents(documents, self.embedder)

    def build_raptor_tree(self) -> Dict[int, pd.DataFrame]:
        """Build the RAPTOR tree structure with level metadata and parent-child relationships."""
        results = {}
        current_texts = [self.extract_text(text) for text in self.texts]
        current_metadata = [
            {"level": 0, "origin": "original", "parent_id": None} for _ in self.texts
        ]

        for level in range(1, self.max_levels + 1):

            embeddings = self.embed_texts(current_texts)
            n_clusters = min(10, len(current_texts) // 2)
            cluster_labels = self.perform_clustering(np.array(embeddings), n_clusters)

            df = pd.DataFrame(
                {
                    "text": current_texts,
                    "embedding": embeddings,
                    "cluster": cluster_labels,
                    "metadata": current_metadata,
                }
            )

            results[level - 1] = df

            summaries = []
            new_metadata = []
            for cluster in df["cluster"].unique():
                cluster_docs = df[df["cluster"] == cluster]
                cluster_texts = cluster_docs["text"].tolist()
                cluster_metadata = cluster_docs["metadata"].tolist()
                summary = self.summarize_texts(cluster_texts)
                summaries.append(summary)
                new_metadata.append(
                    {
                        "level": level,
                        "origin": f"summary_of_cluster_{cluster}_level_{level - 1}",
                        "child_ids": [meta.get("id") for meta in cluster_metadata],
                        "id": f"summary_{level}_{cluster}",
                    }
                )

            current_texts = summaries
            current_metadata = new_metadata

            if len(current_texts) <= 1:
                results[level] = pd.DataFrame(
                    {
                        "text": current_texts,
                        "embedding": self.embed_texts(current_texts),
                        "cluster": [0],
                        "metadata": current_metadata,
                    }
                )
                break

        return results

    def run(self, query: str, k: int = 3) -> str:
        """Run the RAPTOR query pipeline."""
        vectorstore = self.build_vectorstore(self.tree_results)
        retriever = self.create_retriever(vectorstore)

        relevant_docs = []

        new_queries = self.query_transform.transform(query)

        for quer in new_queries:
            rel_doc = retriever.get_relevant_documents(quer)
            relevant_docs.extend(rel_doc)

        relevant_docs = self.doc_enricher.enrich(relevant_docs)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = PromptTemplate.from_template(
            "Given the following context, please answer the question:\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=query)

        return answer
