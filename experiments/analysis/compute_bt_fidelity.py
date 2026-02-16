"""
B_t fidelity: violation rate (post burn-in) per run. Output: summaries/bt_fidelity.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def compute_violation_rate(iter_metrics_path: Path, B: int) -> float:
    """Post burn-in: fraction of steps with inside_bt == 0."""
    inside = []
    with open(iter_metrics_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["step"] > B:
                inside.append(rec["inside_bt"])
    if not inside:
        return float("nan")
    return 1.0 - (sum(inside) / len(inside))


def compute_bt_fidelity(
    run_dirs: list[Path],
    B: int,
    width: float | None = None,
    h: float | None = None,
    chain_id: int | None = None,
) -> pd.DataFrame:
    """One row per run: run_dir, width, h, chain_id, violation_rate."""
    rows = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "iter_metrics.jsonl"
        if not path.exists():
            continue
        rate = compute_violation_rate(path, B)
        rows.append({
            "run_dir": str(run_dir),
            "width": width,
            "h": h,
            "chain_id": chain_id,
            "violation_rate": rate,
        })
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", help="Run directories (each has iter_metrics.jsonl)")
    p.add_argument("--B", type=int, default=50_000, help="Burn-in steps")
    p.add_argument("-o", "--out", default="experiments/summaries/bt_fidelity.csv")
    args = p.parse_args()
    run_dirs = [Path(d) for d in args.run_dirs]
    df = compute_bt_fidelity(run_dirs, args.B)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
