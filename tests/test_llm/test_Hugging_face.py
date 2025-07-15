from rag_src.llm import HuggingFaceLLMWrapper


def test_huggingface_llm_wrapper_generate():
    llm = HuggingFaceLLMWrapper(model_name="sshleifer/tiny-gpt2")
    query = "What is the capital of Germany?"
    contexts = ["Geography facts:", "Europe has many countries."]

    result = llm.generate(query, contexts)

    assert isinstance(result, str)
    assert len(result.strip()) > 0
