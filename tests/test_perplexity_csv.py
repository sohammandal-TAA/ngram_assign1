import subprocess
import csv
from pathlib import Path


def test_runner_writes_csv(tmp_path):
    out_csv = tmp_path / "ppl_report.csv"

    cmd = [
        "python",
        "tests/test_generation_perplexity.py",
        "--vocab_size", "10000",
        "--n", "3",
        "--unk_cutoff", "1",
        "--k", "0.2",
        "--lambdas", "0.33,0.33,0.34",
        "--out_csv", str(out_csv),
    ]

    # run the script (should exit 0)
    res = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # header + 3 rows (n=1,2,3)
    assert rows[0] == ["n", "mle", "add_k", "interpolation"]
    assert len(rows) == 4
    # ensure n column contains 1,2,3
    assert [int(rows[i][0]) for i in range(1, 4)] == [1, 2, 3]
