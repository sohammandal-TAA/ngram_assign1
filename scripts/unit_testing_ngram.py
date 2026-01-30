import math
import random
import sys
import os
import tempfile
import pytest
from pathlib import Path

# allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import (
    NGramLanguageModel,
    BOS,
    EOS,
    UNK,
    build_vocabulary,
    replace_unk,
    add_sentence_markers,
    load_txt_corpus,
    split_80_20,
)
from tokenizer import Tokenizer

TOKENIZER = Tokenizer()

# ---------------------------
# Small deterministic corpus
# ---------------------------

TRAIN_TEXTS = [
    "i love natural language processing",
    "i love machine learning",
    "natural language processing is fun",
]

TEST_TEXTS = [
    "i love learning",
    "nlp is cool",
]


# ---------------------------
# Tokenizer
# ---------------------------

def test_tokenizer_basic():
    # basic tokenizer correctness
    tokens = TOKENIZER.tokenize("I Love NLP!")
    assert tokens == ["i", "love", "nlp"]


# ---------------------------
# Vocabulary
# ---------------------------

def test_build_vocabulary():
    # vocab creation + counts
    tokens = ["i", "love", "nlp", "i", "love"]
    vocab, counter = build_vocabulary(tokens, vocab_size=10, unk_cutoff=0)

    assert "i" in vocab
    assert counter["i"] == 2
    assert UNK in vocab
    assert BOS in vocab
    assert EOS in vocab


# ---------------------------
# UNK replacement
# ---------------------------

def test_replace_unk():
    # unseen words should map to UNK
    vocab = {"i", "love", UNK}
    tokens = ["i", "love", "deep"]
    out = replace_unk(tokens, vocab)

    assert out == ["i", "love", UNK]


# ---------------------------
# Sentence markers
# ---------------------------

def test_add_sentence_markers():
    # BOS and EOS placement
    tokens = ["i", "love", "nlp"]
    out = add_sentence_markers(tokens, n=3)

    assert out[:2] == [BOS, BOS]
    assert out[-1] == EOS


# ---------------------------
# Training
# ---------------------------

def test_training_creates_counts():
    # ngram and context counts created
    lm = NGramLanguageModel(n=2, tokenizer=TOKENIZER, unk_cutoff=0)
    lm.train(["i love nlp"])

    assert ("i", "love") in lm.ngram_counts
    assert ("love", "nlp") in lm.ngram_counts
    assert (BOS,) in lm.context_counts


# ---------------------------
# Probability (MLE)
# ---------------------------

def test_mle_probability_zero_for_unseen():
    # MLE gives zero for unseen ngrams
    lm = NGramLanguageModel(n=2, tokenizer=TOKENIZER, smoothing="mle")
    lm.train(["a a a"])

    prob = lm.probability(("a", "b"))
    assert prob == 0.0


# ---------------------------
# Add-k smoothing
# ---------------------------

def test_addk_probability_nonzero():
    # add-k gives non-zero prob
    lm = NGramLanguageModel(n=2, tokenizer=TOKENIZER, smoothing="add_k", k=0.5)
    lm.train(["a a a"])

    prob = lm.probability(("a", "b"))
    assert prob > 0.0


# ---------------------------
# Interpolation
# ---------------------------

def test_interpolation_probability_nonzero():
    # interpolation backs off correctly
    lm = NGramLanguageModel(
        n=2,
        tokenizer=TOKENIZER,
        smoothing="interpolation",
        lambdas=[0.5, 0.5],
        k=0.5,
    )
    lm.train(["a a a"])

    prob = lm.probability(("a", "b"))
    assert prob > 0.0


# ---------------------------
# Sentence log-probability
# ---------------------------

def test_sentence_log_probability():
    # sentence log-probability finite
    lm = NGramLanguageModel(n=2, tokenizer=TOKENIZER, smoothing="add_k", k=0.5)
    lm.train(TRAIN_TEXTS)

    logp, count = lm.sentence_log_probability("i love nlp")
    assert count > 0
    assert not math.isinf(logp)


# ---------------------------
# Perplexity (evaluation)
# ---------------------------

def test_perplexity_mle_vs_smoothing():
    # smoothing should not be worse than MLE
    lm_mle = NGramLanguageModel(n=2, tokenizer=TOKENIZER, smoothing="mle")
    lm_addk = NGramLanguageModel(n=2, tokenizer=TOKENIZER, smoothing="add_k", k=0.5)

    lm_mle.train(TRAIN_TEXTS)
    lm_addk.train(TRAIN_TEXTS)

    ppl_mle = lm_mle.perplexity(TEST_TEXTS)
    ppl_addk = lm_addk.perplexity(TEST_TEXTS)

    assert ppl_addk > 0
    assert not math.isinf(ppl_addk)

    if not math.isinf(ppl_mle):
        assert ppl_addk <= ppl_mle


# ---------------------------
# Text generation
# ---------------------------

def test_generate_text():
    # generation returns clean text
    random.seed(0)

    lm = NGramLanguageModel(n=2, tokenizer=TOKENIZER, smoothing="add_k", k=0.5)
    lm.train(TRAIN_TEXTS)

    text = lm.generate(max_length=10, seed=["i"])
    assert isinstance(text, str)
    assert len(text.split()) <= 10
    assert BOS not in text
    assert UNK not in text


# ---------------------------
# Generation with unseen context
# ---------------------------

def test_generation_works_with_unseen_context():
    # smoothing used during generation
    lm = NGramLanguageModel(n=3, tokenizer=TOKENIZER, smoothing="add_k", k=0.5)
    lm.train(["hello world"])

    out = lm.generate(max_length=5, seed=["completely", "new"])
    assert isinstance(out, str)


# ---------------------------
# Load txt corpus
# ---------------------------

def test_load_txt_corpus_basic():
    # file loading and line cleanup
    content = "line one\n\nline two\nline three\n"

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(content)
        path = f.name

    lines = load_txt_corpus(path)
    os.remove(path)

    assert lines == ["line one", "line two", "line three"]


def test_load_txt_corpus_empty_file():
    # empty file should raise error
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        path = f.name

    with pytest.raises(ValueError):
        load_txt_corpus(path)

    os.remove(path)


# ---------------------------
# Train / test split
# ---------------------------

def test_split_80_20_normal():
    # correct 80/20 split
    lines = [f"line {i}" for i in range(10)]

    train, test = split_80_20(lines)

    assert len(train) == 8
    assert len(test) == 2
    assert train == lines[:8]
    assert test == lines[8:]


def test_split_80_20_small():
    # small corpus fallback
    lines = ["only one line"]

    train, test = split_80_20(lines)

    assert train == lines
    assert test == lines
