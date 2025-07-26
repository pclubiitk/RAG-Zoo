# llm/

## `SmartLLM` 

GraphRAG LLM

- Parameters (with default vals and types) for class object:
    ```python
    * No parameters
    ```

- Functions:
  ```py
  # Prioritized environment variable-based selection
  * _init_llm(self)
  * _extract_text(self, output) -> str
  * predict(self, prompt_template: PromptTemplate, **kwargs) -> str
  * complete(self, prompt: str, **kwargs) -> CompletionResponse
  * stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen
  * chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse
  * stream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponseGen
  * generate(self, query: str, contexts: List[str]) -> str

  Below are async methods:

  * acomplete(self, prompt: str, **kwargs) -> CompletionResponse
  * astream_complete(
        self, prompt: str, **kwargs
    ) -> AsyncGenerator[CompletionResponse, None]
  * achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse
  * astream_chat(
        self, messages: List[ChatMessage], **kwargs
    ) -> AsyncGenerator[ChatResponse, None]
  ```

## `HuggingFaceLLMWrapper` 

HuggingFace LLM

- Parameters (with default vals and types) for class object:
    ```python
    * model_name: str = "distilbert"
    ```

- Functions:
  ```py
  * generate(self, query: str, contexts: List[str], temperature: float = 0.7) -> str
  ```

## `OpenAILLM` 

GPT-3.5-Turbo / GPT-4 / GPT-4o

- Parameters (with default vals and types) for class object:
    ```python
    * model: str = "gpt-4"
    ```

- Functions:
  ```py
  * generate(self, query: str, contexts: List[str]) -> Union[str, dict] 
  ```

## `GroqLLM` 

- Parameters (with default vals and types) for class object:
    ```python
    * api_key: str = None
    * model: str = "llama3-8b-8192"
    ```

- Functions:
  ```py
  * generate(self, query: str, contexts: List[str]) -> Union[str, dict]
  ```

## `OllamaLLM` 

Local Llama 3, Mistral

- Parameters (with default vals and types) for class object:
    ```python
    * model: str = "mistral"
    ```
  
- Functions:
  ```py
  * generate(self, query: str, contexts: List[str]) -> Union[str, dict]
  ```

## `GeminiLLM` 

Google Gemini Pro / Flash

- Parameters (with default vals and types) for class object:
    ```python
    * api_key: str = None
    * model: str = "gemini-1.5-flash"
    ```
    
- Functions:
  ```py
  * generate(self, query: str, contexts: List[str]) -> Union[str, dict]
  ```