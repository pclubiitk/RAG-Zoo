from typing import List, Optional
import os
import asyncio
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
import asyncio
from typing import AsyncGenerator

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

        if doc_loader:
            if getattr(doc_loader, "path", None) is None:
                doc_loader.path = self.docdir
            self.doc_loader = doc_loader
        else:
            self.doc_loader = DefaultDocLoader(self.docdir)
        self.retriever = retriever
    
    async def create_retriever(self):
        await self.ensure_index_ready()
        # Initialize retriever (after index exists)
        index_path = getattr(self.indexer, "persist_path", "default_index")
        self.retriever = self.retriever or DefaultRetriever(index_path=index_path)
        return self
    
    async def ensure_index_ready(self):
        """
        Check if FAISS index exists. If not, run the ingestion pipeline.
        """
        index_path = getattr(self.indexer, "persist_path", "default_index")
        index_file = os.path.join(index_path, "index.faiss")

        if not os.path.exists(index_file):
            print(f"[INFO] FAISS index not found at {index_file}. Running ingestion pipeline.")
            docs = self.load_preprocess_chunk_documents()
            enriched_docs = self.doc_enricher.enrich(docs)
            print(f"Enriched documents count: {len(enriched_docs)}")
            await self.index_documents(enriched_docs)
        else:
            print(f"[INFO] Found existing index at {index_file}. Skipping ingestion.")

    def load_preprocess_chunk_documents(self) -> List[str]:
        print("=== LOADING DOCUMENTS ===")
        documents = self.doc_loader.load()
        print(f"Loaded {len(documents)} raw documents.")

        if not documents:
            raise RuntimeError("No documents found by doc_loader.")

        if self.preprocessor:
            documents = self.preprocessor.preprocess(documents)
            print(f"Preprocessed down to {len(documents)} documents.")

        if self.chunker:
            documents = self.chunker.chunk(documents)
            print(f"Chunked into {len(documents)} total chunks.")

        return documents
    
    async def embed_and_index(self, docs, meta):
        embeddings = await asyncio.to_thread(self.embeddor.embed, docs)
        await asyncio.to_thread(self.indexer.index, embeddings, docs, meta)

    async def index_documents(self, documents: List[str], metadata: Optional[List[dict]] = None, batch_size=16) -> None:
        print("=== INDEXING DOCUMENTS ===")
        batches = [
            (documents[i:i+batch_size], metadata[i:i+batch_size] if metadata else [{}]*batch_size)
            for i in range(0, len(documents), batch_size)
        ]
        tasks = [
            self.embed_and_index(docs, meta)
            for docs, meta in batches
        ]
        await asyncio.gather(*tasks)
        self.indexer.persist()
        print("[INFO] Index persisted.")
    
    # Factual queries: enhance the query for better retrieval
    async def factual_retrieve(self, query):
        print("retrieving factual")
        enhanced_query_prompt = PromptTemplate("Enhance this factual query for better information retrieval: {query}")
        formatted_prompt = enhanced_query_prompt.format(query=query)
        enhanced_query = await self.llm.generate(formatted_prompt,contexts=[])
        docs = await self.retriever.retrieve(enhanced_query)
        return docs
    
    # Analytical queries: decompose into sub-questions, then rerank for diversity
    async def analytical_retrieve(self, query, k=4):
        print("retrieving analytical")
        self.query_transform=QueryDecomposer(llm=self.llm.llm)
        sub_questions: list[str] = await asyncio.to_thread(self.query_transform.transform(query=query))
        all_docs = []
        retrieve_tasks = [asyncio.create_task(self.retriever.retrieve(sq)) for sq in sub_questions]
        results = await asyncio.gather(*retrieve_tasks)
        for result in results:
            all_docs.extend(result)
            
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
    async def opinion_retrieve(self, query, k=4):
        viewpoints_prompt = PromptTemplate("Identify {k} distinct viewpoints or perspectives on the topic: {query}")
        formatted_prompt = viewpoints_prompt.format(query=query,k=k)
        viewpoints= await self.llm.generate(formatted_prompt,contexts=[]).split('\n')
        all_docs = []
        retrieve_tasks = [asyncio.create_task(self.retriever.retrieve(vp)) for vp in viewpoints]
        results = await asyncio.gather(*retrieve_tasks)
        for result in results:
            all_docs.extend(result)
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
    async def context_retrieve(self, query, k=4):
            print("retrieving contextual")
            context_prompt = PromptTemplate(template="Given the user context: {context}\nReformulate the query to best address the user's needs: {query}")
            context_prompt_formatted=context_prompt.format(query=query,context="")
            contextualized_query = await self.llm.generate(context_prompt_formatted,contexts=[])
            docs = await self.retriever.retrieve(contextualized_query)
            return docs
       
    async def run(self, query: str) -> AsyncGenerator[str, None]:
        print("=== RUNNING RELIABLE RAG PIPELINE ===")
        
        await self.create_retriever()
        
        docs = self.load_preprocess_chunk_documents()
        
        #Enrich the documents with metadata/context
        task = asyncio.create_task(self.doc_enricher.enrich(docs))
        await self.index_documents(docs)
        
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
        docs = await strategy(query)
        
        #Final Answer generation
        prompt_template = PromptTemplate("""Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        )
        
        enriched_docs = await task
        FinalPrompt = prompt_template.format(question=query,context="\n".join(enriched_docs))
        if hasattr(self.llm,"generate_stream"):
            async for token in self.llm.generate_stream(FinalPrompt, contexts=[]):
                yield token
                yield "\n"
        else:
            yield self.llm.generate(FinalPrompt, contexts=[])