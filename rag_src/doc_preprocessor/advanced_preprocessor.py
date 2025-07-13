from typing import List
from bs4 import BeautifulSoup
from unidecode import unidecode
import re
from nltk.corpus import stopwords
from .base import BasePreprocessor
import nltk
nltk.download("stopwords")


class AdvancedPreprocessor(BasePreprocessor):
    """
    Advanced preprocessor that performs:
    - Lowercasing
    - Stripping extra whitespace and newlines
    - Unicode normalization
    - HTML tag removal
    - Special character cleanup
    - Stopword removal (optional)
    """

    def __init__(self, remove_stopwords: bool = False, language: str = "english"):
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()

    def preprocess(self, docs: List[str]) -> List[str]:
        cleaned_docs = []

        for doc in docs:
            # Lowercase
            text = doc.lower()

            # Strip HTML if present
            text = BeautifulSoup(text, "html.parser").get_text()

            # Normalize unicode characters (e.g., é → e)
            text = unidecode(text)

            # Remove emojis and special symbols
            text = re.sub(r'[^\w\s.,!?;:()\[\]\'\"-]', '', text)

            # Normalize punctuation spacing (e.g., "hello   , world" → "hello, world")
            text = re.sub(r'\s*([.,!?;:()\[\]\"-])\s*', r'\1 ', text)

            # Collapse multiple spaces and newlines
            text = re.sub(r'\s+', ' ', text).strip()

            # Optionally remove stopwords
            if self.remove_stopwords:
                words = text.split()
                words = [w for w in words if w not in self.stop_words]
                text = ' '.join(words)

            cleaned_docs.append(text)

        return cleaned_docs
