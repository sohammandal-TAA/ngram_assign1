import string
from typing import List


class Tokenizer:
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        """
        Initializes the Tokenizer.

        Args:
            lowercase: If True, convert text to lowercase.
            remove_punctuation: If True, remove symbols defined in string.punctuation.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

        # Translation table for removing punctuation characters
        self._punct_table = str.maketrans("", "", string.punctuation)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text based on initialization settings.
        """
        # Handle empty or whitespace-only input
        if not text or text.strip() == "":
            return []

        # 1. Lowercase handling
        if self.lowercase:
            text = text.lower()

        # 2. Punctuation removal using str.translate
        if self.remove_punctuation:
            text = text.translate(self._punct_table)

        # 3. Split on any whitespace (spaces, tabs, newlines)
        tokens = text.split()

        return tokens
