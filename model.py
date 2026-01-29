import random
import math
from collections import defaultdict, Counter
from typing import List, Tuple
from tokenizer import Tokenizer


# =========================
# GLOBAL SPECIAL TOKENS
# =========================
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


# =========================
# GLOBAL TOKENIZER CONFIG
# =========================
TOKENIZER = Tokenizer(
    lowercase=True,
    remove_punctuation=True,
    remove_special_chars=True,
    remove_numbers=False,
    remove_apostrophes=True
)


# =========================
# UTILITY FUNCTIONS
# =========================
def build_vocabulary(
    tokens: List[str],
    vocab_size: int = 10000,
    unk_cutoff: int = 1
) -> Tuple[set, Counter]:
    """
    Build vocabulary using frequency cutoff and max size.
    """
    counter = Counter(tokens)

    # Remove rare words
    filtered = {w: c for w, c in counter.items() if c > unk_cutoff}

    # Keep top-k most frequent words
    most_common = Counter(filtered).most_common(vocab_size)

    vocab = set(w for w, _ in most_common)
    vocab.update({UNK, BOS, EOS})

    return vocab, counter


def replace_unk(tokens: List[str], vocab: set) -> List[str]:
    """
    Replace tokens not in vocabulary with <UNK>.
    """
    return [t if t in vocab else UNK for t in tokens]


def add_sentence_markers(tokens: List[str], n: int) -> List[str]:
    """
    Add BOS and EOS markers to a token sequence.
    """
    return [BOS] * (n - 1) + tokens + [EOS]


# =========================
# N-GRAM LANGUAGE MODEL
# =========================
class NGramLanguageModel:
    """
    N-gram Language Model using Maximum Likelihood Estimation (MLE)
    """

    def __init__(
        self,
        n: int = 3,
        vocab_size: int = 10000,
        unk_cutoff: int = 1,
        smoothing: str = "mle",
        k: float = 1.0,
        lambdas: List[float] = None
    ):
        if n < 1:
            raise ValueError("n must be >= 1")

        self.n = n
        self.vocab_size = vocab_size
        self.unk_cutoff = unk_cutoff
        self.smoothing = smoothing
        self.k = k
        # --- lambdas: if provided, must match `n`; otherwise default to uniform weights ---
        if lambdas:
            if len(lambdas) != n:
                raise ValueError(f"Interpolation lambdas must have length n={n} (got {len(lambdas)})")
            if any(x < 0 for x in lambdas):
                raise ValueError("Interpolation lambdas must be non-negative")
            total = sum(lambdas)
            if total == 0:
                raise ValueError("Interpolation lambdas must sum to a positive value")
            # normalize to sum to 1.0
            self.lambdas = [float(x) / total for x in lambdas]
        else:
            self.lambdas = [1.0 / n] * n

        self.vocab = set()
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)

    # ---------------------------
    # Training
    # ---------------------------
    def train(self, texts: List[str]) -> None:
        all_tokens = []
        tokenized_texts = []

        for text in texts:
            tokens = TOKENIZER.tokenize(text)
            tokenized_texts.append(tokens)
            all_tokens.extend(tokens)

        self.vocab, _ = build_vocabulary(
            all_tokens,
            vocab_size=self.vocab_size,
            unk_cutoff=self.unk_cutoff
        )

        for tokens in tokenized_texts:
            tokens = replace_unk(tokens, self.vocab)
            tokens = add_sentence_markers(tokens, self.n)

            # populate counts for all orders (1..n) so interpolation/backoff can use them
            for i in range(len(tokens)):
                for k in range(1, self.n + 1):
                    if i + k > len(tokens):
                        break
                    gram = tuple(tokens[i:i + k])
                    context = gram[:-1]
                    self.ngram_counts[gram] += 1
                    self.context_counts[context] += 1

    # ---------------------------
    # Probability (MLE)
    # ---------------------------
    def probability(self, ngram: Tuple[str, ...]) -> float:
        if len(ngram) != self.n:
            raise ValueError("Invalid ngram length")

        context = ngram[:-1]
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        # ---- MLE ----
        if self.smoothing == "mle":
            if context_count == 0:
                return 0.0
            return ngram_count / context_count

        # ---- Add-k smoothing ----
        if self.smoothing == "add_k":
            V = len(self.vocab)
            k = self.k if self.k is not None else 1.0
            return (ngram_count + k) / (context_count + k * V)

        # ---- Interpolation ----
        if self.smoothing == "interpolation":
            if not self.lambdas:
                raise ValueError("Interpolation requires lambdas")

            prob = 0.0
            V = len(self.vocab)

            # interpolate over orders 1..n (unigram..n-gram)
            for i in range(1, self.n + 1):
                sub_ngram = ngram[-i:]
                sub_context = sub_ngram[:-1]

                num = self.ngram_counts.get(tuple(sub_ngram), 0)
                den = self.context_counts.get(tuple(sub_context), 0)

                if i == 1:
                    # unigram with add-k smoothing (den may be 0; add-k handles it)
                    k = self.k if self.k is not None else 1.0
                    p = (num + k) / (den + k * V)
                else:
                    # higher-order: MLE (guard division)
                    p = (num / den) if den > 0 else 0.0

                prob += self.lambdas[i - 1] * p

            return prob

        raise ValueError("Unknown smoothing method")

    # ---------------------------
    # Sampling with backoff
    # ---------------------------
    def _sample_next(self, context: Tuple[str, ...]) -> str:
        """
        Backoff sampling: try shorter contexts if unseen.
        """
        for k in range(len(context)):
            sub_context = context[k:]

            candidates = []
            weights = []

            for ngram, count in self.ngram_counts.items():
                if ngram[:-1] == sub_context:
                    candidates.append(ngram[-1])
                    weights.append(count)

            if candidates:
                return random.choices(candidates, weights=weights, k=1)[0]

        return EOS

    # ---------------------------
    # Text Generation
    # ---------------------------
    def generate(
        self,
        max_length: int = 50,
        seed: List[str] = None
    ) -> str:

        SENT_END_PUNCT = {".", "!", "?"}

        # ---- Initialize context ----
        if seed is None:
            context = [BOS] * (self.n - 1)
            generated = []
        else:
            seed = replace_unk(seed, self.vocab)
            context = ([BOS] * (self.n - 1) + seed)[-self.n + 1:]
            generated = seed.copy()

        # ---- Generate tokens ----
        for _ in range(max_length):
            next_word = self._sample_next(tuple(context))

            if next_word == EOS:
                # ✔ Add period only if last token isn't punctuation
                if generated and generated[-1] not in SENT_END_PUNCT:
                    generated.append(".")
                # ✔ Reset context for next sentence
                context = [BOS] * (self.n - 1)
                continue

            generated.append(next_word)
            context.append(next_word)
            context = context[-self.n + 1:]

        # ---- fallback: if generation produced nothing or only punctuation, try a unigram/BOS-based fallback ----
        cleaned = [w for w in generated if w != UNK and w not in SENT_END_PUNCT]
        if not cleaned:
            # try sampling once from BOS context
            unigram_candidates = {}
            for g, c in self.ngram_counts.items():
                if len(g) == 1:
                    tok = g[0]
                    if tok not in {BOS, EOS, UNK} and tok.isalnum():
                        unigram_candidates[tok] = unigram_candidates.get(tok, 0) + c

            if unigram_candidates:
                # pick most frequent unigram (deterministic fallback)
                tok = max(unigram_candidates.items(), key=lambda x: x[1])[0]
                return tok

            # final fallback: try sampling from BOS context (may return EOS)
            fb = self._sample_next(tuple([BOS] * (self.n - 1)))
            return "" if fb in {EOS, UNK, None} else fb

        return " ".join(w for w in generated if w != UNK)



    # ---------------------------
    # Sentence log-probability
    # ---------------------------
    def sentence_log_probability(self, sentence: str) -> Tuple[float, int]:
        tokens = TOKENIZER.tokenize(sentence)
        tokens = replace_unk(tokens, self.vocab)
        tokens = add_sentence_markers(tokens, self.n)

        log_prob = 0.0
        count = 0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            prob = self.probability(ngram)

            if prob == 0:
                return float("-inf"), 0

            log_prob += math.log(prob)
            count += 1

        return log_prob, count

    # ---------------------------
    # Perplexity
    # ---------------------------
    def perplexity(self, texts: List[str]) -> float:
        total_log_prob = 0.0
        total_ngrams = 0

        for text in texts:
            log_prob, count = self.sentence_log_probability(text)

            if log_prob == float("-inf"):
                return float("inf")

            total_log_prob += log_prob
            total_ngrams += count

        if total_ngrams == 0:
            return float("inf")

        return math.exp(-total_log_prob / total_ngrams)