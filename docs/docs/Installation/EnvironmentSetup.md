# Environment Setup

Like standard practices to use our API Keys, we store them away in a .env file (typically in the form `GEMINI_API_KEY = ****`) 

To use them in your program:

```python title="Loading environment variables" linenums="1"
import os
import dotenv
dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

Then use these in your program now passing it under the api_key parameter 
eg:
```python title="Using API Keys" linenums="6"
from rag_src.llm import GeminiLLM 
llm = GeminiLLM(api_key=GEMINI_API_KEY)
print(llm.generate("What is Retrieval Augmented Generation?"))
```
