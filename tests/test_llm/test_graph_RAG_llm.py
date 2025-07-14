import pytest
from unittest.mock import MagicMock
from rag_src.llm import SmartLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import CompletionResponse


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = MagicMock(text="mocked response")
    return llm


def test_generate(mock_llm):
    sllm = SmartLLM(llm_instance=mock_llm)
    result = sllm.generate("What's up?", ["context"])
    assert result == "mocked response"
    mock_llm.generate.assert_called_once_with("What's up?", ["context"])


def test_predict(mock_llm):
    sllm = SmartLLM(llm_instance=mock_llm)
    template = PromptTemplate("Hello {name}")
    result = sllm.predict(template, name="Shubham")
    assert result == "mocked response"
    mock_llm.generate.assert_called_once_with("Hello Shubham", contexts=[])


def test_complete(mock_llm):
    sllm = SmartLLM(llm_instance=mock_llm)
    response = sllm.complete("Tell me something interesting.")
    assert isinstance(response, CompletionResponse)
    assert response.text == "mocked response"
    mock_llm.generate.assert_called_once_with("Tell me something interesting.", contexts=[])


def test_extract_text_string_fallback():
    sllm = SmartLLM(llm_instance=MagicMock())
    output = 42  # No `.text` attribute
    result = sllm._extract_text(output)
    assert result == "42"


def test_metadata_values(mock_llm):
    sllm = SmartLLM(llm_instance=mock_llm)
    metadata = sllm.metadata
    assert metadata.context_window == 4096
    assert metadata.num_output == 512
    assert metadata.model_name == "SmartLLM"
    assert metadata.is_chat_model is False
