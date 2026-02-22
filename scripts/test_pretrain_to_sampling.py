#!/usr/bin/env python3
"""
Quick E2E test: pretrain -> eval_checkpoint -> run_single_chain with pretrain-path.
Uses minimal loads (n=64, pretrain 20 steps, T=2, h=0) to verify the pipeline.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rsc_e2e_"))
    ckpt = tmp / "pretrain.pt"

    print("1. Pretrain (n=64, 20 steps)...")
    r = run([
        sys.executable, "scripts/pretrain.py",
        "--width", "0.1",
        "--n_train", "64",
        "--pretrain-steps", "20",
        "-o", str(ckpt),
        "--data_dir", "experiments/data",
    ])
    if r.returncode != 0:
        print("Pretrain failed:", r.stderr)
        return 1
    print("   ", r.stdout.strip())

    print("2. Eval checkpoint...")
    r = run([
        sys.executable, "scripts/eval_checkpoint.py",
        str(ckpt),
        "--n_train", "64",
        "--probe_size", "32",
        "--data_dir", "experiments/data",
    ])
    if r.returncode != 0:
        print("Eval failed:", r.stderr)
        return 1
    print("   ", r.stdout.strip())

    run_dir = tmp / "chain_run"
    run_dir.mkdir(parents=True)

    print("3. Run chain with pretrain-path (T=2, h=0, B=0)...")
    r = run([
        sys.executable, "scripts/run_single_chain.py",
        "--width", "0.1",
        "--h", "0",
        "--noise-scale", "0",
        "--alpha", "0.1",
        "--chain", "0",
        "--n_train", "64",
        "--probe_size", "32",
        "--T", "2",
        "--B", "0",
        "--S", "2",
        "--log-every", "1",
        "--pretrain-path", str(ckpt),
        "--runs_dir", str(tmp),
    ])
    if r.returncode != 0:
        print("Chain failed:", r.stderr)
        return 1

    # Find run dir (name includes h=0, alpha, etc.)
    runs = list(tmp.glob("w0.1_n64_*_chain0"))
    if not runs:
        print("No run dir found")
        return 1
    run_dir = runs[0]
    iter_path = run_dir / "iter_metrics.jsonl"
    if not iter_path.exists():
        print("iter_metrics.jsonl not found")
        return 1

    import json
    lines = [json.loads(l) for l in iter_path.read_text().strip().splitlines() if l.strip()]
    if not lines:
        print("No iter_metrics lines")
        return 1

    first = lines[0]
    ce_mean = first.get("ce_mean_train")
    if ce_mean is None:
        print("ce_mean_train missing")
        return 1
    if not (0 < ce_mean < 20):
        print(f"ce_mean_train={ce_mean} out of expected range [0,20]")
        return 1

    print("4. Step 1 ce_mean_train =", ce_mean)
    print("OK: pretrain -> eval -> sampling pipeline works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
