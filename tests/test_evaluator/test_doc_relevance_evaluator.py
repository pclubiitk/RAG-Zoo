import pytest
from rag_src.evaluator import RelevanceEvaluator


class MockLLM:
    def __init__(self, output: str):
        self.output = output

    def generate(self, query, contexts=None):
        return self.output


@pytest.mark.parametrize(
    "mock_output, expected_score, expected_above_threshold",
    [
        ("0.9", 0.9, True),
        ("0.7", 0.7, True),
        ("0.69", 0.69, False),
        ("0.0", 0.0, False),
        ("invalid", 0.0, False),  # simulating LLM failure
    ]
)
def test_evaluate_score_parsing_and_threshold(mock_output, expected_score, expected_above_threshold):
    mock_llm = MockLLM(output=mock_output)
    evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)

    result = evaluator.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        contexts=["Paris is the capital city of France."]
    )

    assert isinstance(result, dict)
    assert result["relevance_score"] == expected_score
    assert result["above_threshold"] == expected_above_threshold


def test_structure_of_output():
    mock_llm = MockLLM("0.85")
    evaluator = RelevanceEvaluator(llm=mock_llm)
    
    result = evaluator.evaluate(
        query="What is 2 + 2?",
        response="The answer is 4.",
        contexts=["Simple arithmetic operations like 2 + 2 equal 4."]
    )

    assert "relevance_score" in result
    assert "above_threshold" in result
    assert isinstance(result["relevance_score"], float)
    assert isinstance(result["above_threshold"], bool)
