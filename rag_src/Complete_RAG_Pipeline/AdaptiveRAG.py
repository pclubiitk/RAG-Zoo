from typing import List, Optional
import os
from rag_src.llm import BaseLLM, DefaultLLM
from rag_src.retriever import BaseRetriever, DefaultRetriever
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import QueryDecomposer
from rag_src.post_retrival_enricher import PostBaseEnricher,PostDefaultEnricher
from rag_src.indexer import BaseIndexer, DefaultIndexer
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor
from rag_src.chunker import BaseChunker, DefaultChunker
from rag_src.evaluator.doc_relevance_evaluator import RelevanceEvaluator
from llama_index.core import PromptTemplate
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field

class SelectedIndices(BaseModel):
    indices: List[int] = Field(
        description="Indices of selected documents",
        json_schema_extra={"example": [0, 1, 2, 3]},
    )
class CategoriesOptions(BaseModel):
    category: str = Field(
        description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual",
        json_schema_extra={"example": "Factual"},
    )
class AdaptiveRAG:
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embeddor: Optional[BaseEmbedder] = None,
        indexer: Optional[BaseIndexer] = None,
        retriever: Optional[BaseRetriever] = None,
        doc_loader: Optional[BaseDocLoader] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        chunker: Optional[BaseChunker] = None,
        doc_enricher: Optional[PostBaseEnricher] = None,
        docdir: str = "data",
    ):
        self.docdir = docdir
        self.llm = llm or DefaultLLM()
        self.embeddor = embeddor or DefaultEmbedder()
        self.indexer = indexer or DefaultIndexer()
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()
        self.doc_enricher = doc_enricher or PostDefaultEnricher()
        self.relevance_grader = RelevanceEvaluator(llm=self.llm)

        index_path = getattr(self.indexer, "persist_path", "default_index")
        index_file = os.path.join(index_path, "index.faiss")
        if not os.path.exists(index_file):
            print("[INFO] FAISS index not found. Running ingestion.")
            self.load_and_ingest_documents()
            
        self.retriever = retriever or DefaultRetriever(index_path=index_path)
    
    # Factual queries: enhance the query for better retrieval
    def factual_retrieve(self, query):
        print("retrieving factual")
        enhanced_query_prompt = PromptTemplate("Enhance this factual query for better information retrieval: {query}")
        formatted_prompt = enhanced_query_prompt.format(query=query)
        enhanced_query = self.llm.generate(formatted_prompt,contexts=[])
        docs = self.retriever.retrieve(enhanced_query)
        return docs
    
    # Analytical queries: decompose into sub-questions, then rerank for diversity
    def analytical_retrieve(self, query, k=4):
        print("retrieving analytical")
        self.query_transform=QueryDecomposer(llm=self.llm.llm)
        sub_questions: list[str] = self.query_transform.transform(query=query)
        all_docs = [doc for sq in sub_questions for doc in self.retriever.retrieve(sq)]
            
        diversity_prompt = PromptTemplate(
                template= """Select the most diverse and relevant set of {k} documents for the query: '{query}'\nDocuments: {docs}\n.Return only the indices of selected documents as a list of integers.
                Return ONLY a JSON object in the following format:
                { "indices": [0, 1, 2, 3] }"""
            )
        diversity_program = LLMTextCompletionProgram.from_defaults(
            output_cls=SelectedIndices,
            llm=self.llm.llm, 
            prompt=diversity_prompt
        )
        docs_text = "\n".join([f"<doc{i+1}>:\n{doc.get("text","")[:100]}\n</doc{i+1}>" for i, doc in enumerate(all_docs)])
        result=diversity_program(query=query,docs=docs_text,k=k)
        selected_indices_result = result.indices
        return [all_docs[i] for i in selected_indices_result if i < len(all_docs)]

    # Opinion queries: extract perspectives, retrieve documents for each, then cluster
    def opinion_retrieve(self, query, k=4):
        viewpoints_prompt = PromptTemplate("Identify {k} distinct viewpoints or perspectives on the topic: {query}")
        formatted_prompt = viewpoints_prompt.format(query=query,k=k)
        viewpoints= self.llm.generate(formatted_prompt,contexts=[]).split('\n')
        all_docs = [doc for vp in viewpoints for doc in self.retriever.retrieve(vp)]
        opinion_prompt = PromptTemplate(
            template="""Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints:\nDocuments: {docs}\nSelected indices:
            Return ONLY a JSON object in the following format:
                { "indices": [0, 1, 2, 3] }"""
        )
        opinion_program = LLMTextCompletionProgram.from_defaults(
            output_cls=SelectedIndices,
            llm=self.llm.llm, 
            prompt=opinion_prompt
        )
        docs_text = "\n".join([f"<doc{i+1}>:\n{doc.get("text","")[:100]}\n</doc{i+1}>" for i, doc in enumerate(all_docs)])
        result=opinion_program(query=query,docs=docs_text,k=k)
        selected_indices=result.indices
        print(f'selected diverse and relevant documents')
        return [all_docs[i] for i in selected_indices if i < len(all_docs)]

    # Contextual queries: include user context in reformulating the query
    def context_retrieve(self, query, k=4):
            print("retrieving contextual")
            context_prompt = PromptTemplate(template="Given the user context: {context}\nReformulate the query to best address the user's needs: {query}")
            context_prompt_formatted=context_prompt.format(query=query,context="")
            contextualized_query = self.llm.generate(context_prompt_formatted,contexts=[])
            docs = self.retriever.retrieve(contextualized_query)
            return docs
        
    def run(self, query: str) -> str:
        print("=== RUNNING RELIABLE RAG PIPELINE ===")
        
        #Classifying the query type
        query_classifier_prompt = PromptTemplate(
            template="Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\nQuery: {query}",
        )
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=CategoriesOptions,
            llm=self.llm.llm, 
            prompt=query_classifier_prompt
        )
        result = program(query=query)
        print(f"Classifying query...  {result.category}")
        
        #Use the appropriate retrieval strategy
        self.strategies = {
            "Factual": self.factual_retrieve,
            "Analytical": self.analytical_retrieve,
            "Opinion": self.opinion_retrieve,
            "Contextual": self.context_retrieve
        }
        
        strategy = self.strategies[result.category]
        docs = strategy(query)
        
        #Enrich the documents with metadata/context
        enriched_docs = self.doc_enricher.enrich(docs)
        
        #Final Answer generation
        prompt_template = PromptTemplate("""Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        )
        FinalPrompt = prompt_template.format(question=query,context="\n".join([doc.get("text","") for doc in enriched_docs]))
        return self.llm.generate(FinalPrompt,contexts=[])
        
    # Index documents using embedding + indexing pipeline
    def ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        print("[INFO] Indexing documents...")
        embeddings = self.embeddor.embed(documents)
        self.indexer.index(embeddings, documents, metadata)
        self.indexer.persist()
        print("[INFO] Indexing complete.")

    # Load + preprocess + chunk + index documents (called if index is missing)
    def load_and_ingest_documents(self) -> None:
        print("[INFO] Loading and processing documents...")
        documents = self.doc_loader.load()
        if self.preprocessor:
            documents = self.preprocessor.preprocess(documents)
        if self.chunker:
            documents = self.chunker.chunk(documents)
        self.ingest_documents(documents)
