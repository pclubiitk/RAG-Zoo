from .default import DefaultRetriever
from typing import List, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ReRankingRetriever(DefaultRetriever):
    '''
    Reranker + Retriever
    '''
    def __init__(
        self, 
        index_path = "default_index", 
        model_name = "all-MiniLM-L6-v2", 
        reranker_model_name="BAAI/bge-reranker-large",
        initial_top_n:int = 20
    ):
        super().__init__(index_path=index_path, model_name=model_name, top_k=initial_top_n)
        
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.reranker_model =  AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
        self.reranker_model.eval()
        
    def rerank(self, query: str, docs:List[Dict[str, Any]], k:int =5) -> List[Dict[str, Any]]:
        '''
        Reranks a list of documents based on a query using a BAAI reranker model.
        '''
        
        if not docs:
            return []
        
        pairs = [[query, doc["text"]] for doc in docs]
        
        with torch.no_grad():
            inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1).float()
        
        doc_score_pairs = list(zip(docs, scores)) # Puts score back into docs
        sorted_doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in sorted_doc_score_pairs[:min(len(docs), k)]] # Removes score from the elements
        
        return reranked_docs
    
    def retrieve(self, query:str, k:int =5) -> List[Dict[str, Any]]:
        '''
        Retrieves initial documents and then reranks them.
        '''
        initial_docs = super().retrieve(query=query)
        return self.rerank(query, initial_docs, k)
