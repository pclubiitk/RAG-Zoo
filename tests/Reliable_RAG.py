from dotenv import load_dotenv
from rag_src.llm import GroqLLM
import os

from rag_src.complete_RAG_Pipeline.ReliableRAG import ReliableRAG

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
def main():
    rag = ReliableRAG(llm=GroqLLM(api_key=GROQ_API_KEY),docdir=r"D:\data\final_draft.pdf")

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = rag.run(query)
        print("\n[ANSWER]:", answer)

if __name__ == "__main__":
    main()
