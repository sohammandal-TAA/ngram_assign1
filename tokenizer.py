import re
import string
from typing import List


class Tokenizer:
    """
    Flexible tokenizer for NLP preprocessing.

    Supports:
    - lowercasing
    - punctuation removal
    - special character removal
    - number removal
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_special_chars: bool = True,
        remove_numbers: bool = False,
        remove_apostrophes: bool = False,   # default = KEEP apostrophes
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_apostrophes = remove_apostrophes

        self._build_patterns()

    # ---------------------------
    # Regex pattern builder
    # ---------------------------
    def _build_patterns(self):
        punct = string.punctuation

        # If we want to KEEP apostrophes, exclude them from punctuation
        if not self.remove_apostrophes:
            punct = punct.replace("'", "")

        self.punctuation_pattern = re.compile(f"[{re.escape(punct)}]")

        # Special characters pattern
        if self.remove_apostrophes:
            self.special_char_pattern = re.compile(r"[^a-zA-Z0-9\s]")
        else:
            self.special_char_pattern = re.compile(r"[^a-zA-Z0-9\s']")

        self.number_pattern = re.compile(r"\d+")

    # ---------------------------
    # Main tokenize function
    # ---------------------------
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        if self.lowercase:
            text = text.lower()

        if self.remove_numbers:
            text = self.number_pattern.sub(" ", text)

        if self.remove_punctuation:
            text = self.punctuation_pattern.sub(" ", text)

        if self.remove_special_chars:
            text = self.special_char_pattern.sub(" ", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    # ---------------------------
    # Sentence-level tokenize
    # ---------------------------
    def tokenize_sentences(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenize(text) for text in texts]

    # ---------------------------
    # Debug helper
    # ---------------------------
    def explain(self) -> dict:
        return {
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_special_chars": self.remove_special_chars,
            "remove_numbers": self.remove_numbers,
            "remove_apostrophes": self.remove_apostrophes,
        }
