#!/usr/bin/env python3
"""
Compare R2 vs R3 using second half of each run.
Computes mean/std and ESS for nll_probe_mean, U_train, U_data, dist_to_ref.
f_nll and nll_probe_mean in logs are on the TRAIN set (1024 subset), not probe set.

Usage: python scripts/compare_r2_r3.py [r2_dir] [r3_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ess_bulk(trace: np.ndarray, max_lag: int | None = None) -> float:
    """Bulk ESS from autocorrelation."""
    n = len(trace)
    if n < 2:
        return 0.0
    trace = trace - trace.mean()
    if trace.var() == 0:
        return float("nan")
    if max_lag is None:
        max_lag = min(n // 2, 500)
    ac = np.correlate(trace, trace, mode="full")[len(trace) - 1 :]
    ac = ac[: max_lag + 1] / (ac[0] + 1e-12)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return n / tau if tau > 0 else float("nan")


def load_iter_metrics(path: Path) -> list[dict]:
    """Load iter_metrics.jsonl."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        for k, v in rec.items():
            if isinstance(v, str) and v.lower() == "nan":
                rec[k] = np.nan
        records.append(rec)
    return records


def compare_run(recs: list[dict], name: str) -> dict:
    """Second-half stats: mean/std of nll_probe_mean, U_train, U_data; dist_to_ref trend."""
    if not recs:
        return {"name": name, "n": 0}
    steps = [r["step"] for r in recs if r.get("step") is not None]
    if not steps:
        return {"name": name, "n": 0}
    t_max = max(steps)
    half = t_max / 2
    second_half = [r for r in recs if r.get("step", 0) > half]
    if not second_half:
        second_half = recs[-max(1, len(recs) // 2) :]  # fallback: last half of records

    def vals(key: str):
        return [r[key] for r in second_half if key in r and r[key] is not None and np.isfinite(r[key])]

    nll = vals("nll_probe_mean")
    u_train = vals("U_train")
    u_data = vals("U_data")
    dist = vals("dist_to_ref")

    out = {"name": name, "n": len(second_half), "steps": [r["step"] for r in second_half]}
    if nll:
        out["nll_probe_mean_mean"] = float(np.mean(nll))
        out["nll_probe_mean_std"] = float(np.std(nll))
    if u_train:
        out["U_train_mean"] = float(np.mean(u_train))
        out["U_train_std"] = float(np.std(u_train))
    if u_data:
        out["U_data_mean"] = float(np.mean(u_data))
        out["U_data_std"] = float(np.std(u_data))
    if dist:
        out["dist_to_ref_min"] = float(np.min(dist))
        out["dist_to_ref_max"] = float(np.max(dist))
        out["dist_to_ref_mean"] = float(np.mean(dist))
        by_step = sorted((r["step"], r["dist_to_ref"]) for r in second_half if r.get("dist_to_ref") is not None and np.isfinite(r.get("dist_to_ref", np.nan)))
        if len(by_step) >= 2:
            out["dist_to_ref_first"] = by_step[0][1]
            out["dist_to_ref_last"] = by_step[-1][1]

    # ESS (on second-half traces in step order)
    by_step = sorted(second_half, key=lambda r: r.get("step", 0))
    for key in ("nll_probe_mean", "U_train", "U_data", "dist_to_ref"):
        arr = np.array([r[key] for r in by_step if key in r and r[key] is not None and np.isfinite(r.get(key, np.nan))])
        if len(arr) >= 2:
            out[f"{key}_ESS"] = float(ess_bulk(arr))
    return out


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    r2_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "experiments/runs/w0.1_n1024_h5e-06_a0.1_chain0"
    r3_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "experiments/runs/w0.1_n1024_h2.5e-06_a0.1_chain0"

    r2_recs = load_iter_metrics(r2_dir / "iter_metrics.jsonl")
    r3_recs = load_iter_metrics(r3_dir / "iter_metrics.jsonl")

    r2 = compare_run(r2_recs, "R2")
    r3 = compare_run(r3_recs, "R3")

    print("=" * 60)
    print("R2 vs R3 (second half of each run)")
    print("Note: f_nll and nll_probe_mean are on TRAIN set (1024), not probe set.")
    print("=" * 60)
    for label, keys in [
        ("nll_probe_mean", ("nll_probe_mean_mean", "nll_probe_mean_std", "nll_probe_mean_ESS")),
        ("U_train", ("U_train_mean", "U_train_std", "U_train_ESS")),
        ("U_data", ("U_data_mean", "U_data_std", "U_data_ESS")),
        ("dist_to_ref", ("dist_to_ref_mean", "dist_to_ref_min", "dist_to_ref_max", "dist_to_ref_first", "dist_to_ref_last", "dist_to_ref_ESS")),
    ]:
        print(f"\n{label}:")
        for run, data in [("R2", r2), ("R3", r3)]:
            parts = [f"  {run}: n={data.get('n', 0)}"]
            for k in keys:
                if k in data:
                    if "ESS" in k:
                        parts.append(f"{k}={data[k]:.1f}")
                    else:
                        parts.append(f"{k}={data[k]:.4f}")
            print(" ".join(parts))
    print()


if __name__ == "__main__":
    main()
