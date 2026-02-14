from __future__ import annotations

import glob
from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"


def latest_csv(mode: str) -> Path:
    files = sorted(glob.glob(str(REPORTS_DIR / f"eval_per_query_{mode}_*.csv")))
    if not files:
        raise SystemExit(f"No eval csv found for mode={mode} in {REPORTS_DIR}")
    return Path(files[-1])


def main() -> None:
    # assumes you just ran: python -m app.eval.run_eval --mode hybrid_rrf
    csv_path = latest_csv("hybrid_rrf")
    df = pd.read_csv(csv_path)

    # hard guardrails
    cov = float(df["answer_coverage"].mean())
    bad = df[df["failure_reason"] != "ok"]

    print("latest:", csv_path.name)
    print("mean_answer_coverage:", cov)
    print("non_ok_rows:", len(bad))

    if cov < 0.95:
        raise SystemExit(f"FAIL: mean answer_coverage too low: {cov}")

    if len(bad) > 0:
        print(bad[["q", "answer_coverage", "failure_reason"]].to_string(index=False))
        raise SystemExit("FAIL: found non-ok rows")

    print("SMOKE PASS")


if __name__ == "__main__":
    main()
