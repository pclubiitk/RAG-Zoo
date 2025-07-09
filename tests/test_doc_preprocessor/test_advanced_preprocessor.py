import pytest
from rag_src.doc_preprocessor import AdvancedPreprocessor

@pytest.fixture
def html_text():
    return [
        "<p>Hello <b>world!</b></p>   ",
        "This is <i>clean</i> & <u>formatted</u>.",
        "Remove    extra   spaces!!! 😃"
    ]

@pytest.fixture
def raw_text():
    return [
        "Éléphant très fâché.",
        "He didn't say anything; he was silent.",
        "Running, jumping, and swimming - all fun."
    ]

def test_basic_cleaning(html_text):
    preprocessor = AdvancedPreprocessor()
    processed = preprocessor.preprocess(html_text)

    assert processed[0] == "hello world!"
    assert processed[1] == "this is clean formatted."
    assert "spaces! ! !" in processed[2].lower()
    assert "😃" not in processed[2]  # emoji removed

def test_unicode_normalization(raw_text):
    preprocessor = AdvancedPreprocessor()
    result = preprocessor.preprocess(raw_text)

    assert "elephant tres fache." in result[0]  # é, â → e, a
    assert "he didn't say anything;" in result[1].lower()

def test_stopword_removal(raw_text):
    preprocessor = AdvancedPreprocessor(remove_stopwords=True)
    result = preprocessor.preprocess(raw_text)

    # Confirm common English stopwords like 'he', 'was', 'and' are removed
    assert "he" not in result[1]
    assert "was" not in result[1]
    assert "and" not in result[2]

def test_empty_and_whitespace_only():
    preprocessor = AdvancedPreprocessor()
    output = preprocessor.preprocess(["   ", "\n\t", ""])
    assert output == ["", "", ""]
