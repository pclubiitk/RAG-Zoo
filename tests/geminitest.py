from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-1.5-flash", api_key="AIzaSyDcWZvNjpQdzpv-XQ7Nz2ZzJKN4aVSEuug")
resp = llm.complete("HI")
print(str(resp))