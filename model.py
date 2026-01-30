import random
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Optional

# =========================
# SPECIAL TOKENS
# =========================
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


# =========================
# UTILITY FUNCTIONS
# =========================
def build_vocabulary(
    tokens: List[str],
    vocab_size: int,
    unk_cutoff: int
) -> Tuple[set, Counter]:
    counter = Counter(tokens)

    # Keep words strictly above cutoff
    filtered = {w: c for w, c in counter.items() if c > unk_cutoff}

    most_common = Counter(filtered).most_common(vocab_size)
    vocab = set(w for w, _ in most_common)
    vocab.update({BOS, EOS, UNK})

    return vocab, counter


def replace_unk(tokens: List[str], vocab: set) -> List[str]:
    return [t if t in vocab else UNK for t in tokens]


def add_sentence_markers(tokens: List[str], n: int) -> List[str]:
    return [BOS] * (n - 1) + tokens + [EOS]

def load_txt_corpus(path: str) -> List[str]:
    """Load .txt file as a corpus"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError("Text file is empty.")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines


def split_80_20(lines: List[str]):
    """ Split the corpus intwo 80 % lines for traning and 20 % for testing"""
    if len(lines) < 2:
        return lines, lines

    split_idx = int(len(lines) * 0.8)
    return lines[:split_idx], lines[split_idx:]

# =========================
# PROBABILITY FUNCTIONS
# =========================
def mle_probability(
    ngram: Tuple[str, ...],
    ngram_counts: dict,
    context_counts: dict
) -> float:
    context = ngram[:-1]
    den = context_counts.get(context, 0)
    if den == 0:
        return 0.0
    return ngram_counts.get(ngram, 0) / den


def add_k_probability(
    ngram: Tuple[str, ...],
    ngram_counts: dict,
    context_counts: dict,
    vocab_size: int,
    k: float
) -> float:
    context = ngram[:-1]
    num = ngram_counts.get(ngram, 0)
    den = context_counts.get(context, 0)
    return (num + k) / (den + k * vocab_size)


def interpolated_probability(
    ngram: Tuple[str, ...],
    ngram_counts: dict,
    context_counts: dict,
    vocab_size: int,
    lambdas: List[float],
    k: float
) -> float:
    n = len(ngram)
    prob = 0.0

    for i in range(1, n + 1):
        sub = ngram[-i:]
        context = sub[:-1]

        if i == 1:
            p = add_k_probability(sub, ngram_counts, context_counts, vocab_size, k)
        else:
            den = context_counts.get(context, 0)
            p = (ngram_counts.get(sub, 0) / den) if den > 0 else 0.0

        prob += lambdas[i - 1] * p

    return prob


# =========================
# N-GRAM LANGUAGE MODEL
# =========================
class NGramLanguageModel:
    def __init__(
        self,
        n: int,
        tokenizer,
        vocab_size: int = 10000,
        unk_cutoff: int = 1,
        smoothing: str = "mle",
        k: float = 1.0,
        lambdas: Optional[List[float]] = None,
        rng: Optional[random.Random] = None
    ):
        if n < 1:
            raise ValueError("n must be >= 1")

        self.n = n
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.unk_cutoff = unk_cutoff
        self.smoothing = smoothing
        self.k = k
        self.rng = rng or random.Random()

        if smoothing == "interpolation":
            if not lambdas or len(lambdas) != n:
                raise ValueError("Interpolation lambdas must match n")
            s = sum(lambdas)
            self.lambdas = [x / s for x in lambdas]
        else:
            self.lambdas = None

        self.vocab = set()
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)

    # ---------------------------
    # TRAINING
    # ---------------------------
    def train(self, texts: List[str]) -> None:
        all_tokens = []
        tokenized = []

        for text in texts:
            toks = self.tokenizer.tokenize(text)
            tokenized.append(toks)
            all_tokens.extend(toks)

        self.vocab, _ = build_vocabulary(
            all_tokens, self.vocab_size, self.unk_cutoff
        )

        for toks in tokenized:
            toks = replace_unk(toks, self.vocab)
            toks = add_sentence_markers(toks, self.n)

            for i in range(len(toks)):
                for k in range(1, self.n + 1):
                    if i + k > len(toks):
                        break
                    gram = tuple(toks[i:i + k])
                    self.ngram_counts[gram] += 1
                    self.context_counts[gram[:-1]] += 1

    # ---------------------------
    # PROBABILITY
    # ---------------------------
    def probability(self, ngram: Tuple[str, ...]) -> float:
        if self.smoothing == "mle":
            return mle_probability(ngram, self.ngram_counts, self.context_counts)

        if self.smoothing == "add_k":
            return add_k_probability(
                ngram, self.ngram_counts, self.context_counts, len(self.vocab), self.k
            )

        if self.smoothing == "interpolation":
            return interpolated_probability(
                ngram,
                self.ngram_counts,
                self.context_counts,
                len(self.vocab),
                self.lambdas,
                self.k
            )

        raise ValueError("Unknown smoothing")

    # ---------------------------
    # SAMPLING (TESTABLE)
    # ---------------------------
    def _sample_next(self, context: Tuple[str, ...]) -> str:
        """
        Sample next word using model probability()
        (respects MLE / add-k / interpolation)
        """
        context = tuple(context[-(self.n - 1):])

        candidates = []
        probs = []

        for word in self.vocab:
            if word == BOS:
                continue

            ngram = context + (word,)
            p = self.probability(ngram)

            if p > 0:
                candidates.append(word)
                probs.append(p)

        if not candidates:
            # fallback: pick random word (excluding BOS)
            candidates = [w for w in self.vocab if w != BOS]
            probs = [1.0 / len(candidates)] * len(candidates)

        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # fallback uniform if all probs are zero
            probs = [1.0 / len(candidates)] * len(candidates)

        return self.rng.choices(candidates, weights=probs, k=1)[0]

    # ---------------------------
    # GENERATION
    # ---------------------------
    def generate(self, max_length: int = 50, seed: Optional[List[str]] = None) -> str:
        if seed is None:
            context = [BOS] * (self.n - 1)
            output = []
        else:
            seed = replace_unk(seed, self.vocab)
            context = ([BOS] * (self.n - 1) + seed)[-self.n + 1:]
            output = seed.copy()

        for _ in range(max_length):
            word = self._sample_next(tuple(context))
            if word == EOS:
                break
            output.append(word)
            context = (context + [word])[-self.n + 1:]

        return " ".join(w for w in output if w != UNK)

    # ---------------------------
    # PERPLEXITY
    # ---------------------------
    def sentence_log_probability(self, sentence: str) -> Tuple[float, int]:
        toks = self.tokenizer.tokenize(sentence)
        toks = replace_unk(toks, self.vocab)
        toks = add_sentence_markers(toks, self.n)

        log_p = 0.0
        count = 0

        for i in range(len(toks) - self.n + 1):
            p = self.probability(tuple(toks[i:i + self.n]))
            if p == 0:
                return float("-inf"), count
            log_p += math.log(p)
            count += 1

        return log_p, count

    def perplexity(self, texts: List[str]) -> float:
        total_lp = 0.0
        total_n = 0

        for t in texts:
            lp, c = self.sentence_log_probability(t)
            if lp == float("-inf"):
                return float("inf")
            total_lp += lp
            total_n += c

        return math.exp(-total_lp / total_n) if total_n > 0 else float("inf")
