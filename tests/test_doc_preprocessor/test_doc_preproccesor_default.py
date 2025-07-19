import pytest
from rag_src.doc_preprocessor import DefaultPreprocessor


@pytest.fixture
def sample_docs():
    return [
        "  This is a TEST.  ",
        "\n\nNewlines\tand   tabs\n\n",
        "   Multiple    spaces    between   words.   ",
    ]


def test_preprocessing_behavior(sample_docs):
    preprocessor = DefaultPreprocessor()
    processed = preprocessor.preprocess(sample_docs)

    assert processed[0] == "this is a test."
    assert processed[1] == "newlines and tabs"
    assert processed[2] == "multiple spaces between words."


def test_empty_input():
    preprocessor = DefaultPreprocessor()
    assert preprocessor.preprocess([]) == []


def test_whitespace_only_strings():
    preprocessor = DefaultPreprocessor()
    input_docs = ["   ", "\n\t ", "     \n"]
    output = preprocessor.preprocess(input_docs)
    assert output == ["", "", ""]
