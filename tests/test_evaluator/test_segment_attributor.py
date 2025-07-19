import pytest
from rag_src.evaluator.segment_attributor import SegmentAttributor
from llama_index.core.schema import TextNode


class MockLLM:
    def __init__(self, output: str):
        self.output = output

    def generate(self, prompt: str, contexts=None):
        return self.output


@pytest.fixture
def mock_llm():
    return MockLLM(output="Segment 1.\nSegment 2.")


@pytest.fixture
def attributor(mock_llm):
    return SegmentAttributor(llm=mock_llm)


@pytest.fixture
def sample_query():
    return "What is photosynthesis?"


@pytest.fixture
def sample_response():
    return "Photosynthesis is the process by which plants make food using sunlight."


@pytest.fixture
def sample_nodes():
    return [
        TextNode(
            text="Photosynthesis is the process by which green plants use sunlight to make food."
        ),
        TextNode(text="It typically occurs in the chloroplasts of plant cells."),
        TextNode(text="This process converts light energy into chemical energy."),
    ]


def test_locate_segments_output_structure(
    attributor, sample_query, sample_response, sample_nodes
):
    result = attributor.locate_segments(
        query=sample_query, response=sample_response, docs=sample_nodes
    )

    assert isinstance(result, dict)
    assert "segments" in result
    assert isinstance(result["segments"], str)
    assert "Segment 1." in result["segments"]


def test_evaluate_compatibility(
    attributor, sample_query, sample_response, sample_nodes
):
    contexts = [node.text for node in sample_nodes]
    result = attributor.evaluate(
        query=sample_query, response=sample_response, contexts=contexts
    )

    assert isinstance(result, dict)
    assert "segments" in result
    assert isinstance(result["segments"], str)
