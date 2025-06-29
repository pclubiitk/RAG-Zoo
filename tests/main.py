from rag_src.doc_preprocessor.Advanced_preprocessor import AdvancedPreprocessor

docs = [
    "  This is <b>Example</b> TEXT!   😊   ",
    "Here’s another\tone… with      spaces & weird chars!"
]

pre = AdvancedPreprocessor(remove_stopwords=False)
cleaned = pre.preprocess(docs)

print(cleaned)
