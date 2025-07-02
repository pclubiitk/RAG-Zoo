from rag_src.query_transformer.base import BaseQueryTransformer
from typing import List
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import PromptTemplate

class MultiQuery(BaseQueryTransformer):
    """
    Multi Query transformer that generates multiple reformulated queries from the original input query.
    Useful when multiple retrievals are needed.
    """
    def __init__(self, llm):
        self.llm = llm

    def transform(self, query: str, n: int = 5) -> List[str]:
        
        output_parser = StrOutputParser()
        
        QUERY_PROMPT = PromptTemplate(
        input_variables=["question", "num_queries"],
        template="""You are an AI language model assistant. Your task is to generate {num_queries} 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
        )
        
        llm_chain = QUERY_PROMPT | self.llm | output_parser
        
        queries = llm_chain.invoke({"question": query, "num_queries": n}).replace("\n\n", "\n").split("\n")
        
        return queries