from dotenv import load_dotenv
from rag_src.llm import GroqLLM
import os
load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from rag_src.complete_RAG_Pipeline.CRAG import CRAG  # wherever you saved the CRAG class

# Initialize CRAG with defaults
crag = CRAG(llm=GroqLLM(api_key=GROQ_API_KEY),docdir=r"D:\data\final_draft.pdf")  # assumes you have documents in 'data/'

# Run query loop
while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = crag.run(query)
    print(f"\n[ANSWER]:\n{answer}\n")
