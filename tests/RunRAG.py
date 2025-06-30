from rag_src.complete_RAG_Pipeline.RunRAG import RunRAG
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader
from rag_src.llm.gemini import GeminiLLM
from rag_src.chunker import DefaultChunker
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access the key
gemini_api_key = os.getenv('GEMINI_API_KEY')
# Initialize with just docdir — everything else uses default classes
doc_dir=r"D:\data\final_draft.pdf"
rag = RunRAG(
    llm=GeminiLLM(gemini_api_key),
    embeddor=None,
    indexer=None,
    retriever=None,
    query_transform=None,
    doc_enricher=None,
    doc_loader=UniversalDocLoader(doc_dir),
    preprocessor=None,
    docdir=r"D:\data\final_draft.pdf",  # directory where sample.txt is saved
    chunker= DefaultChunker(chunk_size=512, chunk_overlap=50), 
)

# Step 1: Ingest the documents
rag.load_and_ingest_documents()

# Step 2: Ask a question
query = "What are RAG?"
answer = rag.run(query)

print("\n💡 Answer:")
print(answer)