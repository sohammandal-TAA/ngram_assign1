import random
import re
from typing import List
import math
from collections import defaultdict, Counter
from typing import List, Tuple
from model import NGramLanguageModel, TOKENIZER ,build_vocabulary,add_sentence_markers,replace_unk
from tokenizer import  Tokenizer

if __name__ == "__main__":

    # ---------------------------
    # Sample data
    # ---------------------------
    texts = [
        "I love natural language processing",
        "I love machine learning",
        "Natural language processing is fun",
        "I enjoy learning new things"
    ]

    print("=== TOKENIZER TEST (GLOBAL DEFAULT) ===")
    for t in texts:
        print("Original :", t)
        print("Tokens   :", TOKENIZER.tokenize(t))
        print()

    # ---------------------------
    # build_vocabulary()
    # ---------------------------
    print("=== build_vocabulary() ===")
    all_tokens = []
    for t in texts:
        all_tokens.extend(TOKENIZER.tokenize(t))

    vocab, counter = build_vocabulary(all_tokens, vocab_size=20, unk_cutoff=0)
    print("Vocabulary:", vocab)
    print("Token counts:", counter)
    print()

    # ---------------------------
    # replace_unk()
    # ---------------------------
    print("=== replace_unk() ===")
    test_tokens = ["i", "love", "deep", "learning"]
    print("Before:", test_tokens)
    print("After :", replace_unk(test_tokens, vocab))
    print()

    # ---------------------------
    # add_sentence_markers()
    # ---------------------------
    print("=== add_sentence_markers() ===")
    tokens = ["i", "love", "nlp"]
    print("Before:", tokens)
    print("After :", add_sentence_markers(tokens, n=3))
    print()

    # ---------------------------
    # Initialize NGramLanguageModel
    # ---------------------------
    print("=== Initialize NGramLanguageModel ===")
    model = NGramLanguageModel(n=3, vocab_size=10000, unk_cutoff=0)
    print("n:", model.n)
    print()

    # ---------------------------
    # train()
    # ---------------------------
    print("=== train() ===")
    model.train(texts)
    print("Vocabulary size:", len(model.vocab))
    print("Sample vocab:", list(model.vocab)[:10])
    print("=== N-GRAM COUNTS ===")
    for ngram, count in sorted(model.ngram_counts.items()):
        print(f"{ngram} -> {count}")
    print()

    print("=== CONTEXT COUNTS ===")
    for context, count in sorted(model.context_counts.items()):
        print(f"{context} -> {count}")
    print()
    print("Total ngrams:", len(model.ngram_counts))
    print("Total contexts:", len(model.context_counts))
    print()

    # ---------------------------
    # probability()
    # ---------------------------
    print("=== probability() ===")
    ngram = ("i", "love", "machine")
    print("Ngram:", ngram)
    print("Probability:", model.probability(ngram))
    print()

    # ---------------------------
    # sentence_log_probability()
    # ---------------------------
    print("=== sentence_log_probability() ===")
    sentence = "I love natural language processing"
    log_prob, count = model.sentence_log_probability(sentence)
    print("Sentence :", sentence)
    print("Log prob :", log_prob)
    print("Ngrams   :", count)
    print()

    # ---------------------------
    # perplexity()
    # ---------------------------
    print("=== perplexity() ===")
    test_texts = [
        "I love learning",
        "natural language processing"
    ]
    print("Texts:", test_texts)
    print("Perplexity:", model.perplexity(test_texts))
    print()

    # ---------------------------
    # generate() without seed
    # ---------------------------
    print("=== generate() (no seed) ===")
    print(model.generate(max_length=15))
    print()

    # ---------------------------
    # generate() with seed
    # ---------------------------
    print("=== generate() (with seed) ===")
    seed = ["i", "love"]
    print("Seed:", seed)
    print(model.generate(max_length=15, seed=seed))
    print()
