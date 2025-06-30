from dotenv import load_dotenv
load_dotenv()

from rag_src.complete_RAG_Pipeline.CRAG import CRAG  # wherever you saved the CRAG class

# Initialize CRAG with defaults
crag = CRAG(docdir=r"D:\data\final_draft.pdf")  # assumes you have documents in 'data/'

# Run query loop
while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = crag.run(query)
    print(f"\n[ANSWER]:\n{answer}\n")
