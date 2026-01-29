import random
import math
from collections import defaultdict, Counter
from typing import List, Tuple
from .tokenizer import Tokenizer


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
        unk_cutoff: int = 1
    ):
        if n < 1:
            raise ValueError("n must be >= 1")

        self.n = n
        self.vocab_size = vocab_size
        self.unk_cutoff = unk_cutoff

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

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = ngram[:-1]

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    # ---------------------------
    # Probability (MLE)
    # ---------------------------
    def probability(self, ngram: Tuple[str, ...]) -> float:
        if len(ngram) != self.n:
            raise ValueError("Invalid ngram length")

        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)

        if context_count == 0:
            return 0.0

        return self.ngram_counts.get(ngram, 0) / context_count

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
