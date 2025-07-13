import pytest
from rag_src.evaluator.default import DefaultEvaluator


class MockLLM:
    def __init__(self, response_text):
        self.response_text = response_text

    def generate(self, prompt, contexts=None):
        return {"text": self.response_text}


@pytest.mark.parametrize(
    "llm_output, expected_hallucination",
    [
        ("Yes. The answer is supported by the context.", False),
        ("No. The answer adds facts not present in the context.", True),
        ("no. It includes unrelated information.", True),
        ("yes", False),
    ]
)
def test_evaluate_hallucination_detection(llm_output, expected_hallucination):
    mock_llm = MockLLM(response_text=llm_output)
    evaluator = DefaultEvaluator(llm=mock_llm)

    result = evaluator.evaluate(
        query="What is the capital of France?",
        response="The capital of France is Paris.",
        contexts=["Paris is the capital of France."]
    )

    assert result["hallucination_detected"] == expected_hallucination
    assert isinstance(result["verdict"], str)
    assert len(result["verdict"]) > 0


def test_evaluate_output_structure():
    mock_llm = MockLLM("Yes. It is fully supported.")
    evaluator = DefaultEvaluator(llm=mock_llm)

    result = evaluator.evaluate(
        query="What is 2 + 2?",
        response="2 + 2 is 4.",
        contexts=["Basic arithmetic operations like 2 + 2 equal 4."]
    )

    assert "hallucination_detected" in result
    assert "verdict" in result
