import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# allow running from tests/


from model import NGramLanguageModel, load_txt_corpus, split_80_20

# =========================
# ARGUMENT PARSING
# =========================
parser = argparse.ArgumentParser(description="N-gram Language Model Experiments")

parser.add_argument(
    "--train_file",
    type=str,
    default="../ngram_assign1/AllCombined.txt"
)

parser.add_argument(
    "--test_file",
    type=str,
    default="../ngram_assign1/AllCombined.txt"
)


parser.add_argument("--n_values", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument("--k", type=float, default=0.2)

parser.add_argument(
    "--lambda_sets",
    type=str,
    nargs="+",
    default=["0.2,0.2,0.6", "0.4,0.2,0.4", "0.6,0.2,0.2"],
)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--results_dir", type=str, default="results")

args = parser.parse_args()

# =========================
# CONFIG
# =========================
TRAIN_FILE = args.train_file
TEST_FILE = args.test_file
RESULT_DIR = args.results_dir

os.makedirs(RESULT_DIR, exist_ok=True)

INTERP_CSV = os.path.join(RESULT_DIR, "interpolation_results.csv")
NGRAM_CSV = os.path.join(RESULT_DIR, "ngram_results.csv")
INTERP_PNG = os.path.join(RESULT_DIR, "interpolation_results.png")
NGRAM_PNG = os.path.join(RESULT_DIR, "ngram_results.png")

RNG = random.Random(args.seed)

# =========================
# HELPERS
# =========================
def ppl(x):
    return int(round(x))

def parse_lambdas(lambda_strings):
    return [list(map(float, s.split(","))) for s in lambda_strings]

lambda_sets = parse_lambdas(args.lambda_sets)

# =========================
# LOAD DATA
# =========================
train_lines = load_txt_corpus(TRAIN_FILE)
test_lines = load_txt_corpus(TEST_FILE)
train_data, dev_data = split_80_20(train_lines)
train_data, rest_data = split_80_20(train_lines)
dev_data, test_data = split_80_20(rest_data)
print(f"Training lines: {len(train_data)}")
print(f"Development lines: {len(dev_data)}")
print(f"Testing lines: {len(test_data)}")

# =========================
# INTERPOLATION EXPERIMENT
# =========================
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
        "train_ppl": ppl(model.perplexity(train_data)),
        "dev_ppl": ppl(model.perplexity(dev_data)),
        "test_ppl": ppl(model.perplexity(test_lines)),
    })

df_interp = pd.DataFrame(interp_rows)
df_interp.to_csv(INTERP_CSV, index=False)

# =========================
# N-GRAM ORDER EXPERIMENT
# =========================
ngram_rows = []

for n in args.n_values:
    model = NGramLanguageModel(
        n=n,
        smoothing="add_k",
        k=args.k,
        rng=RNG,
    )
    model.train(train_data)

    ngram_rows.append({
        "n": n,
        "train_ppl": ppl(model.perplexity(train_data)),
        "dev_ppl": ppl(model.perplexity(dev_data)),
        "test_ppl": ppl(model.perplexity(test_lines)),
    })

df_ngram = pd.DataFrame(ngram_rows)
df_ngram.to_csv(NGRAM_CSV, index=False)

# =========================
# SAVE TABLE IMAGES
# =========================
def save_table_image(df, path, title):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.set_title(title)
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

save_table_image(df_interp, INTERP_PNG, "Interpolation Results")
save_table_image(df_ngram, NGRAM_PNG, "N-gram Results")

print("✅ Experiments completed")
print("📁 Results saved to:", RESULT_DIR)
