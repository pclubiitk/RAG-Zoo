import os
import pytest
# from rag_src.complete_RAG_Pipeline.CatcheRAG import CatcheRAG
from CacheRag import CatcheRAG
@pytest.mark.skipif(
    not os.path.exists("/Users/abhinavchhabra/Documents/Coding/Eiffel.txt"),
    reason="Context file missing at D:/data/mini_docs.txt"
)
def test_catche_rag_exact_paper():
    rag = CatcheRAG(
        context_path="/Users/abhinavchhabra/Documents/Coding/Eiffel.txt",
        model_name="gpt2",
        device="cpu",
        max_context_length=512
    )

    query = "Where is the Eiffel Tower?"
    answer = rag.run(query)
    assert isinstance(answer, str), "Answer must be a string"
    assert len(answer.strip()) > 0, "Answer cannot be empty"

    print("\nCatcheRAG Output:\n", answer)

