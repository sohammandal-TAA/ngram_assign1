import random
import math
import re
import sys
import csv
import json
from datetime import datetime
from pathlib import Path

from typing import List, Tuple

# allow running this test directly from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_n_gram.model import NGramLanguageModel, BOS, EOS, UNK
from model_n_gram.tokenizer import Tokenizer


def load_txt_file(path: str) -> list[str]:
    paragraphs = []
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line:
                current.append(line)
            else:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
        if current:
            paragraphs.append(" ".join(current))
    return paragraphs


def sent_tokenize_simple(para: str) -> list[str]:
    if not para or not para.strip():
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]


def test_generate_sample_and_perplexity_on_unseen_data(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "training_data.txt"
    test_path = project_root / "test_data.txt"

    train_paras = load_txt_file(str(train_path)) if train_path.exists() else []
    test_paras = load_txt_file(str(test_path)) if test_path.exists() else []

    # small training for quick tests
    if len(train_paras) > 200:
        train_paras = train_paras[:200]

    # build a small list of test sentences (fallback to synthetic)
    all_test_sents = []
    for p in test_paras:
        all_test_sents.extend(sent_tokenize_simple(p))

    tokenizer = Tokenizer()
    test_sentences = [s for s in all_test_sents if len(tokenizer.tokenize(s)) >= 2]
    if not test_sentences:
        test_sentences = ["a a a a", "the quick brown fox"]

    test_sample = test_sentences[:8]

    # Train MLE (no smoothing)
    lm_mle = NGramLanguageModel(n=3, smoothing="mle")
    lm_mle.train(train_paras if train_paras else ["a a a a"])

    # Train add-k smoothing
    lm_addk = NGramLanguageModel(n=3, smoothing="add_k", k=0.5)
    lm_addk.train(train_paras if train_paras else ["a a a a"])

    # ---- generate() smoke test ----
    random.seed(0)
    out = lm_addk.generate(max_length=12, seed=["the"])
    assert isinstance(out, str)
    assert out != ""
    # should not expose internal markers
    assert BOS not in out and UNK not in out
    # obey max token constraint
    assert len(out.split()) <= 12

    # ---- _sample_next() backoff behavior ----
    # unseen (unlikely) context: a long improbable tuple -> should return EOS or a token
    ctx = tuple(["__not_in_vocab__"] * (lm_addk.n - 1))
    nxt = lm_addk._sample_next(ctx)
    assert isinstance(nxt, str)

    # also test sampling from BOS context
    bos_ctx = tuple([BOS] * (lm_addk.n - 1))
    nxt2 = lm_addk._sample_next(bos_ctx)
    assert isinstance(nxt2, str)

    # ---- perplexity on unseen test sentences ----
    ppl_mle = lm_mle.perplexity(test_sample)
    ppl_addk = lm_addk.perplexity(test_sample)

    # MLE often gives infinite perplexity on unseen data (acceptable); add-k should be finite
    assert ppl_addk > 0 and not math.isinf(ppl_addk)
    # if MLE is finite (rare for small train), add-k should not be worse
    if not math.isinf(ppl_mle):
        assert ppl_addk <= ppl_mle + 1e-6
    else:
        assert math.isinf(ppl_mle)

    # sentence_log_probability should return finite values for add-k
    lp, cnt = lm_addk.sentence_log_probability(test_sample[0])
    assert cnt > 0
    assert not math.isinf(lp)

def train_dev_test_split(sentences: List[str], train_ratio: float = 0.8, dev_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits sentences into train / dev / test.
    Default: 80 / 10 / 10
    """
    assert train_ratio + dev_ratio < 1.0, "Ratios must sum to < 1.0"

    sentences = sentences[:]  # copy
    random.seed(seed)
    random.shuffle(sentences)

    n = len(sentences)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train = sentences[:n_train]
    dev = sentences[n_train:n_train + n_dev]
    test = sentences[n_train + n_dev:]

    return train, dev, test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="test_generation_perplexity.py",
        description="Interactive runner: train models, generate text, and print perplexity scores."
    )
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--unk_cutoff", type=int, default=None)
    parser.add_argument("--k", type=float, default=None, help="add-k smoothing value")
    parser.add_argument("--lambdas", type=str, default=None, help="comma-separated lambdas for interpolation")
    parser.add_argument("--out_csv", type=str, default=None, help="If set, save the n | MLE | Add-k | Interpolation table to CSV")
    args = parser.parse_args()

    def ask(name: str, cur, cast, default):
        if cur is not None:
            return cur
        try:
            val = input(f"{name} (default={default}): ")
        except EOFError:
            # non-interactive environment: use default
            return default
        if not val.strip():
            return default
        return cast(val)

    vocab_size = ask("vocab size", args.vocab_size, int, 10000)
    n = ask("n (ngram order)", args.n, int, 3)
    unk_cutoff = ask("unk_cutoff", args.unk_cutoff, int, 1)
    k = ask("add-k k value", args.k, float, 0.5)

    if args.lambdas:
        lambdas = [float(x) for x in args.lambdas.split(",")]
    else:
        # default: equal weights
        lambdas = None

    # validate lambdas vs n early (if both provided)
    if lambdas is not None and args.n is not None:
        if len(lambdas) != args.n:
            print(f"Error: provided --lambdas length={len(lambdas)} does not match --n={args.n}")
            raise SystemExit(2)

    # --- prepare data ---
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "training_data.txt"
    test_path = project_root / "test_data.txt"

    train_paras = load_txt_file(str(train_path)) if train_path.exists() else []
    test_paras = load_txt_file(str(test_path)) if test_path.exists() else []

    # If training file is empty, prefer test data for quick local runs; otherwise use a small synthetic fallback
    if not train_paras and test_paras:
        print("\nWarning: training_data.txt is empty — using test_data.txt for training (local fallback).")
        all_paras = test_paras
    elif not train_paras and not test_paras:
        print("\nWarning: no training or test data found — using synthetic fallback corpus for demo.")
        all_paras = [
            "the quick brown fox jumps over the lazy dog",
            "the cat sat on the mat",
            "a a a a",
            "hello world this is a tiny corpus for tests"
        ]
    else:
        all_paras = train_paras

    train_paras, dev_paras, test_paras = train_dev_test_split(
        all_paras, train_ratio=0.8, dev_ratio=0.1
    )

    # If the corpus is tiny the split may yield an empty training set — use the
    # full corpus for training in that case so generation/perplexity are meaningful.
    if not train_paras:
        print("\nWarning: corpus too small for standard train/dev/test split — using full corpus for training.")
        train_paras = all_paras
        dev_paras = []
        test_paras = []

    print("\nData split:")
    print(f"  Train: {len(train_paras)} paragraphs")
    print(f"  Dev:   {len(dev_paras)} paragraphs")
    print(f"  Test:  {len(test_paras)} paragraphs")


    tokenizer = Tokenizer()

    # show a short training-token sample (after tokenization, before UNK replacement)
    all_train_tokens = []
    for p in train_paras:
        all_train_tokens.extend(tokenizer.tokenize(p))

    sample_tokens = all_train_tokens[:40]

    print("\nTraining token sample:")
    print(" ".join(sample_tokens) if sample_tokens else "(no training tokens)")

    # Train models with the requested hyperparameters
    lm_mle = NGramLanguageModel(n=n, vocab_size=vocab_size, unk_cutoff=unk_cutoff, smoothing="mle")
    lm_mle.train(train_paras)

    lm_addk = NGramLanguageModel(n=n, vocab_size=vocab_size, unk_cutoff=unk_cutoff, smoothing="add_k", k=k)
    lm_addk.train(train_paras)

    interp_lambdas = lambdas if lambdas is not None else ([1.0 / n] * n)
    lm_interp = NGramLanguageModel(n=n, vocab_size=vocab_size, unk_cutoff=unk_cutoff, smoothing="interpolation", k=k, lambdas=interp_lambdas)
    lm_interp.train(train_paras)

    # vocab diagnostics
    from collections import Counter
    counter = Counter(all_train_tokens)
    vocab_members = sorted(list(lm_addk.vocab))
    effective_vocab_size = len(lm_addk.vocab)
    top10 = [w for w, _ in counter.most_common(10) if w in lm_addk.vocab][:10]

    print("\nHyperparameters:")
    print(f"  n (ngram order): {n}")
    print(f"  vocab_size (requested): {vocab_size}")
    print(f"  unk_cutoff: {unk_cutoff}")
    print(f"  add-k k: {k}")
    print(f"  interpolation lambdas: {interp_lambdas}")

    print("\nVocabulary:")
    print(f"  effective vocab size: {effective_vocab_size}")
    print(f"  top-10 vocab tokens: {top10}")

    # generation
    random.seed(0)
    gen = lm_addk.generate(max_length=200, seed=["oh my"])

    print("\nText generated (seed=\"oh my\"):")
    print(f"  {gen}\n")

    # prepare test sentences 
    all_test_sents = []
    for p in test_paras:
        all_test_sents.extend(sent_tokenize_simple(p))
    test_sentences = [s for s in all_test_sents if len(tokenizer.tokenize(s)) >= 2]
    if not test_sentences:
        test_sentences = ["a a a a", "the quick brown fox"]

    sample_for_eval = test_sentences[:50]

    # scores
    ppl_mle = lm_mle.perplexity(sample_for_eval)
    ppl_addk = lm_addk.perplexity(sample_for_eval)
    ppl_interp = lm_interp.perplexity(sample_for_eval)

    print("Scores:")
    print(f"  Perplexity (MLE / no smoothing): {ppl_mle}")
    print(f"  Perplexity (Add-k, k={k}): {ppl_addk}")
    print(f"  Perplexity (Interpolation, lambdas={interp_lambdas}): {ppl_interp}\n")

    # short interpretation
    print("Interpretation:")
    if math.isinf(ppl_mle):
        print("  - MLE produced infinite perplexity on unseen data (expected when unseen n-grams exist).")
    else:
        print(f"  - MLE perplexity: {ppl_mle:.2f}")

    print(f"  - Add-k smoothing produced {'finite' if not math.isinf(ppl_addk) else 'infinite'} perplexity")
    print(f"  - Interpolation produced {'finite' if not math.isinf(ppl_interp) else 'infinite'} perplexity")

    # comparative note
    if not math.isinf(ppl_addk) and (math.isinf(ppl_mle) or ppl_addk <= ppl_mle):
        print("  => add-k helps on unseen data (lower or finite perplexity).")
    if not math.isinf(ppl_interp) and (math.isinf(ppl_mle) or ppl_interp <= ppl_mle):
        print("  => interpolation helps by mixing lower-order estimates.")


    print("\nPerplexity Table (n = 1,2,3):")
    print("n | MLE | Add-k | Interpolation")
    print("--|-----|-------|--------------")

    table_rows = []
    for n_val in [1, 2, 3]:
        lm_mle = NGramLanguageModel(
            n=n_val,
            vocab_size=vocab_size,
            unk_cutoff=unk_cutoff,
            smoothing="mle"
        )
        lm_mle.train(train_paras)

        lm_addk = NGramLanguageModel(
            n=n_val,
            vocab_size=vocab_size,
            unk_cutoff=unk_cutoff,
            smoothing="add_k",
            k=k
        )
        lm_addk.train(train_paras)

        lambdas_n = [1.0 / n_val] * n_val
        lm_interp = NGramLanguageModel(
            n=n_val,
            vocab_size=vocab_size,
            unk_cutoff=unk_cutoff,
            smoothing="interpolation",
            lambdas=lambdas_n
        )
        lm_interp.train(train_paras)

        ppl_mle = lm_mle.perplexity(sample_for_eval)
        ppl_addk = lm_addk.perplexity(sample_for_eval)
        ppl_interp = lm_interp.perplexity(sample_for_eval)

        mle_disp = 'inf' if math.isinf(ppl_mle) else round(ppl_mle, 2)
        addk_disp = round(ppl_addk, 2)
        interp_disp = round(ppl_interp, 2)

        print(
            f"{n_val} | "
            f"{mle_disp} | "
            f"{addk_disp} | "
            f"{interp_disp}"
        )

        table_rows.append({
            "n": n_val,
            "mle": None if math.isinf(ppl_mle) else float(ppl_mle),
            "add_k": float(ppl_addk),
            "interpolation": float(ppl_interp),
        })

    # --- optionally save table to CSV + metadata JSON ---
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # write CSV
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "mle", "add_k", "interpolation"])
            for r in table_rows:
                mle_val = "inf" if r["mle"] is None else f"{r['mle']:.6f}"
                writer.writerow([r["n"], mle_val, f"{r['add_k']:.6f}", f"{r['interpolation']:.6f}"])

        # write metadata sidecar
        meta = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "vocab_size": vocab_size,
            "n": n,
            "unk_cutoff": unk_cutoff,
            "k": k,
            "lambdas": interp_lambdas,
            "rows": len(table_rows),
        }
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

        print(f"\nSaved perplexity table to: {out_path}\n  (metadata: {meta_path})")

    print("\nDone.")

