import pytest
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import NGramLanguageModel, ModelConfig
from tokenizer import Tokenizer

# ---------------------------
# TEST DATA
# ---------------------------
CORPUS_PARAGRAPH = [
    "The quick brown fox jumps over the lazy dog.",
    "The lazy dog sleeps under the quick brown sun.",
    "A quick brown fox is very quick and very brown."
]

class TestNgram:
    
    # 1. Vocabulary and Training Logic
    def test_build_vocab(self):
        config = ModelConfig(n=2, unk_cutoff=1)
        tokenizer = Tokenizer(lowercase=True, remove_punctuation=True)
        model = NGramLanguageModel(config, tokenizer)
        
        tokenized = [tokenizer.tokenize(s) for s in CORPUS_PARAGRAPH]
        model._build_vocab(tokenized)
        
        # 'fox' appears 2x, 'quick' 4x, 'brown' 3x. All should be in.
        assert "fox" in model.vocab
        assert "quick" in model.vocab
        assert model.config.UNK in model.vocab
        assert model.config.BOS in model.vocab

    def test_variable_n_context(self):
        """Verify that context window shifts correctly for different n values."""
        data = ["a b c d"]
        
        # Bigram
        model2 = NGramLanguageModel(ModelConfig(n=2, smoothing="mle"), Tokenizer())
        model2.train(data)
        assert ("a",) in model2.ngram_model
        assert "b" in model2.ngram_model[("a",)]
        
        # Trigram
        model3 = NGramLanguageModel(ModelConfig(n=3, smoothing="mle"), Tokenizer())
        model3.train(data)
        assert ("a", "b") in model3.ngram_model
        assert "c" in model3.ngram_model[("a", "b")]

    # 2. Probability and Smoothing
    def test_mle_probabilities(self):
        config = ModelConfig(n=2, smoothing="mle")
        model = NGramLanguageModel(config, Tokenizer())
        # 'the' is followed by 'cat' 2/3 of the time, 'dog' 1/3
        model.train(["the cat", "the cat", "the dog"])
        
        p_cat = model.probability(("the", "cat"))
        p_dog = model.probability(("the", "dog"))
        
        assert pytest.approx(p_cat) == 2/3
        assert pytest.approx(p_dog) == 1/3

    def test_add_k_smoothing(self):
        # Using a tiny k to check if unseen words get mass
        config = ModelConfig(n=2, smoothing="add_k", k=0.1)
        model = NGramLanguageModel(config, Tokenizer())
        model.train(["the cat"])
        
        # Transition that never happened
        p_unseen = model.probability(("the", "unseen_token"))
        assert p_unseen > 0
        assert p_unseen < 0.1 # Should be a small share of the mass

    def test_interpolation_logic(self):
        # 0.5 Bigram + 0.5 Unigram
        config = ModelConfig(n=2, smoothing="interpolation", lambdas=[0.5, 0.5])
        model = NGramLanguageModel(config, Tokenizer())
        model.train(["a b", "c b"])
        
        # P(b|a) = 0.5 * P_mle(b|a) + 0.5 * P_mle(b)
        prob = model.probability(("a", "b"))
        assert 0.0 < prob <= 1.0
        
    # 3. Sampling (Top-K and Top-P)
    def test_sampling_top_k(self):
        config = ModelConfig(n=2)
        model = NGramLanguageModel(config, Tokenizer())
        # 'a' can be followed by 5 different tokens
        model.train(["a 1", "a 2", "a 3", "a 4", "a 5"])
        
        # With top_k=1, it should ONLY pick the single most likely token (determinism)
        samples = {model._sample_next(("a",), top_k=1) for _ in range(10)}
        assert len(samples) == 1

    def test_sampling_top_p_nucleus(self):
        config = ModelConfig(n=2)
        model = NGramLanguageModel(config, Tokenizer())
        # Distrubution: 'high' (90%), 'low' (10%)
        model.train(["x high"] * 9 + ["x low"])
        
        # With top_p=0.5, 'low' should be truncated because 'high' already satisfies the 50% mass
        samples = {model._sample_next(("x",), top_p=0.5) for _ in range(20)}
        assert "low" not in samples
        assert "high" in samples

    # 4. Evaluation and Persistence
    def test_perplexity_logic(self):
        config = ModelConfig(n=2, smoothing="add_k", k=1.0)
        model = NGramLanguageModel(config, Tokenizer())
        model.train(CORPUS_PARAGRAPH)
        
        # Perplexity on training data should be lower than on random noise
        ppl_train = model.perplexity(CORPUS_PARAGRAPH)
        ppl_noise = model.perplexity(["xyz abc qrs lmn"])
        assert ppl_train < ppl_noise

    def test_save_and_load_model(self, tmp_path):
        config = ModelConfig(n=3, smoothing="add_k", k=0.5)
        tokenizer = Tokenizer(lowercase=True, remove_punctuation=True)
        model = NGramLanguageModel(config, tokenizer)
        model.train(CORPUS_PARAGRAPH)

        save_path = tmp_path / "ngram_model.pkl"
        model.save(str(save_path))

        # Reload
        loaded_model = NGramLanguageModel.load(str(save_path), tokenizer)

        # Assert parity
        assert loaded_model.config.n == model.config.n
        assert loaded_model.config.k == model.config.k
        assert loaded_model.vocab == model.vocab
        # Check a specific context count in the nested dict
        ctx = (model.config.BOS, model.config.BOS)
        assert dict(loaded_model.ngram_model[ctx]) == dict(model.ngram_model[ctx])