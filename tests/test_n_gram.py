import random
import math
import re
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import List, Tuple

# allow running this test directly from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import NGramLanguageModel, BOS, EOS, UNK
from tokenizer import Tokenizer
import os
import matplotlib.pyplot as plt

from model import (
    NGramLanguageModel,
    load_txt_corpus,
    split_80_20,
)

# =========================
# CONFIG
# =========================
TRAIN_FILE = "/Users/ritik/Code/ngram-assign/ngram_assign1/training_data.txt"
TEST_FILE = "/Users/ritik/Code/ngram-assign/ngram_assign1/test_data.txt"

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

INTERP_CSV = os.path.join(RESULT_DIR, "interpolation_results.csv")
NGRAM_CSV = os.path.join(RESULT_DIR, "ngram_results.csv")

INTERP_PNG = os.path.join(RESULT_DIR, "interpolation_results.png")
NGRAM_PNG = os.path.join(RESULT_DIR, "ngram_results.png")
GEN_FILE = os.path.join(RESULT_DIR, "generated_text.txt")

# Fixed RNG (used ONLY via model)
RNG = random.Random(0)

# =========================
# LOAD DATA
# =========================
train_lines = load_txt_corpus(TRAIN_FILE)
test_lines = load_txt_corpus(TEST_FILE)

train_data, dev_data = split_80_20(train_lines)

# =========================
# INTERPOLATION EXPERIMENT
# =========================
lambda_sets = [
    [0.2, 0.2, 0.6],
    [0.4, 0.2, 0.4],
    [0.6, 0.2, 0.2],
    [0.2, 0.6, 0.2],
    [0.4, 0.4, 0.2],
    [0.1, 0.3, 0.6],
]

interp_rows = []

for lambdas in lambda_sets:
    model = NGramLanguageModel(
        n=3,
        smoothing="interpolation",
        lambdas=lambdas,
        rng=RNG,
    )
    model.train(train_data)

    interp_rows.append({
        "lambda_1": lambdas[0],
        "lambda_2": lambdas[1],
        "lambda_3": lambdas[2],
        "train_ppl": model.perplexity(train_data),
        "dev_ppl": model.perplexity(dev_data),
        "test_ppl": model.perplexity(test_lines),
    })

df_interp = pd.DataFrame(interp_rows)

# Append mode CSV
if os.path.exists(INTERP_CSV):
    df_interp.to_csv(INTERP_CSV, mode="a", header=False, index=False)
else:
    df_interp.to_csv(INTERP_CSV, index=False)

# =========================
# N-GRAM ORDER EXPERIMENT
# =========================
ngram_rows = []

for n in [1, 2, 3]:
    model = NGramLanguageModel(
        n=n,
        smoothing="add_k",
        k=0.2,
        rng=RNG,
    )
    model.train(train_data)

    ngram_rows.append({
        "n": n,
        "train_ppl": model.perplexity(train_data),
        "dev_ppl": model.perplexity(dev_data),
        "test_ppl": model.perplexity(test_lines),
    })

df_ngram = pd.DataFrame(ngram_rows)

# Append mode CSV
if os.path.exists(NGRAM_CSV):
    df_ngram.to_csv(NGRAM_CSV, mode="a", header=False, index=False)
else:
    df_ngram.to_csv(NGRAM_CSV, index=False)

# =========================
# SAVE TABLES AS IMAGES
# =========================
def save_table_image(df, path, title):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

save_table_image(df_interp, INTERP_PNG, "Interpolation Smoothing Results")
save_table_image(df_ngram, NGRAM_PNG, "N-gram Order Results")

# =========================
# TEXT GENERATION
# =========================
gen_model = NGramLanguageModel(
    n=3,
    smoothing="interpolation",
    lambdas=[0.2, 0.2, 0.6],
    rng=RNG,
)
gen_model.train(train_data)

generated = gen_model.generate(max_length=200, seed=["In", "the", "beginning"])

with open(GEN_FILE, "w", encoding="utf-8") as f:
    f.write(generated)

print("All tests completed.")
print("Results saved in:", RESULT_DIR)

