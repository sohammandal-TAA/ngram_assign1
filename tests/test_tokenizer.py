import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from tokenizer import Tokenizer

class TestTokenizer:
    def test_tokenize_with_numbers(self):
        tokenizer = Tokenizer(lowercase=True, remove_punctuation=True)
        text = "The price is $19.99."
        expected_tokens = ["the", "price", "is", "1999"]
        assert tokenizer.tokenize(text) == expected_tokens

    def test_tokenize_no_lowercase_or_punctuation(self):
        tokenizer = Tokenizer(lowercase=False, remove_punctuation=False)
        text = "This is a Test!"
        expected_tokens = ["This", "is", "a", "Test!"]
        assert tokenizer.tokenize(text) == expected_tokens

    def test_tokenize_lowercase(self):
        tokenizer = Tokenizer(lowercase=True, remove_punctuation=False)
        assert tokenizer.tokenize("HELLO WORLD") == ["hello", "world"]

    def test_tokenize_with_punctuation(self):
        tokenizer = Tokenizer(lowercase=True, remove_punctuation=True)
        text = "Wait... what? (This is fun!)"
        expected = ["wait", "what", "this", "is", "fun"]
        assert tokenizer.tokenize(text) == expected

    def test_tokenize_empty_text(self):
        tokenizer = Tokenizer()
        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize("   ") == []

    def test_whitespace_handling(self):
        tokenizer = Tokenizer()
        text = "Words\twith\nnewlines  and   tabs."
        assert len(tokenizer.tokenize(text)) == 5