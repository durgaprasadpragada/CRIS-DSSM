"""
Text Preprocessing Module
Clean and prepare text for FinBERT sentiment analysis
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and normalize text data"""

    def __init__(self):
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        # HTML pattern
        self.html_pattern = re.compile(r'<[^>]+>')

        # Special characters pattern
        self.special_pattern = re.compile(r'[^a-zA-Z0-9\s\.\,\!\?\-]')

    def clean(self, text: str) -> str:
        """
        Clean text by:
        1. Removing URLs
        2. Removing HTML tags
        3. Removing special characters
        4. Lowercasing
        5. Removing extra whitespace
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = self.url_pattern.sub('', text)

        # Remove HTML tags
        text = self.html_pattern.sub('', text)

        # Remove special characters (keep basic punctuation)
        text = self.special_pattern.sub('', text)

        # Lowercase
        text = text.lower()

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts"""
        return [self.clean(text) for text in texts]


class TextTokenizer:
    """Tokenize text for FinBERT (512 token limit)"""

    MAX_TOKENS = 512

    @staticmethod
    def truncate(text: str, max_tokens: int = MAX_TOKENS) -> str:
        """
        Truncate text to fit FinBERT's token limit
        Rough estimate: ~4 chars per token
        """
        if not isinstance(text, str):
            return ""

        max_chars = max_tokens * 4
        return text[:max_chars]

    @staticmethod
    def prepare(text: str, max_tokens: int = MAX_TOKENS) -> str:
        """Prepare text for FinBERT"""
        return TextTokenizer.truncate(text, max_tokens)


class TextPreprocessor:
    """Complete preprocessing pipeline"""

    def __init__(self):
        self.cleaner = TextCleaner()
        self.tokenizer = TextTokenizer()

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline:
        1. Clean
        2. Tokenize/truncate
        """
        text = self.cleaner.clean(text)
        text = self.tokenizer.prepare(text)
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]

    def preprocess_article(self, title: str, text: str) -> str:
        """Preprocess combined title and text"""
        # Combine title and text with separator
        combined = f"{title} | {text}" if title and text else (title or text)
        return self.preprocess(combined)


def create_preprocessor() -> TextPreprocessor:
    """Factory function to create preprocessor"""
    return TextPreprocessor()
