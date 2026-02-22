#!/usr/bin/env python3
"""
Compare mean vs sum CE: stability at same h over 2000 steps.
Runs both, reports which stays stable and has reasonable U/f_nll/grad_norm.
Usage: python scripts/test_mean_vs_sum_ce.py [--h 1e-9]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=float, default=1e-9, help="Step size (default 1e-9)")
    cli = parser.parse_args()
    h = cli.h
    tmp = Path(tempfile.mkdtemp(prefix="rsc_mean_sum_"))
    ckpt = tmp / "pretrain.pt"

    print("1. Pretrain (n=64, 50 steps)...")
    r = run([
        sys.executable, "scripts/pretrain.py",
        "--width", "0.1", "--n_train", "64", "--pretrain-steps", "50",
        "-o", str(ckpt), "--data_dir", "experiments/data",
    ])
    if r.returncode != 0:
        print("Pretrain failed:", r.stderr)
        return 1
    print("   OK")

    T = 2000
    results = {}

    for red in ["mean", "sum"]:
        print(f"\n2. Run chain with ce_reduction={red}, h={h}, T={T}...")
        r = run([
            sys.executable, "scripts/run_single_chain.py",
            "--width", "0.1", "--h", str(h), "--alpha", "0.1", "--chain", "0",
            "--n_train", "64", "--probe_size", "32",
            "--T", str(T), "--B", "0", "--S", "100", "--log-every", "500",
            "--bn-mode", "eval",
            "--ce-reduction", red,
            "--pretrain-path", str(ckpt),
            "--runs_dir", str(tmp),
        ])
        if r.returncode != 0:
            print(f"   FAILED: {r.stderr[:500]}")
            results[red] = {"ok": False, "error": r.stderr[:200]}
            continue

        runs = list(tmp.glob(f"w0.1_n64_*_a0.1_chain0"))
        if not runs:
            results[red] = {"ok": False, "error": "no run dir"}
            continue

        path = runs[0] / "iter_metrics.jsonl"
        if not path.exists():
            results[red] = {"ok": False, "error": "no iter_metrics"}
            continue

        lines = [json.loads(l) for l in path.read_text().strip().splitlines() if l.strip()]
        if not lines:
            results[red] = {"ok": False, "error": "empty iter_metrics"}
            continue

        first, last = lines[0], lines[-1]
        u0, u1 = first.get("U_train"), last.get("U_train")
        f0, f1 = first.get("f_nll"), last.get("f_nll")
        g0, g1 = first.get("grad_norm"), last.get("grad_norm")
        finite = all(
            x is not None and (isinstance(x, bool) or (isinstance(x, (int, float)) and abs(x) < 1e30))
            for x in [u0, u1, f0, f1, g0, g1]
        )

        blowup = False
        if u1 is not None and u0 is not None and u1 > u0 * 3:
            blowup = True
        if g1 is not None and g0 is not None and g1 > g0 * 10:
            blowup = True

        results[red] = {
            "ok": r.returncode == 0 and finite,
            "blowup": blowup,
            "U": [u0, u1],
            "f_nll": [f0, f1],
            "grad_norm": [g0, g1],
        }
        status = "BLOWUP" if blowup else "OK"
        print(f"   {status}: U {u0:.2f}->{u1:.2f}, f_nll {f0:.4f}->{f1:.4f}, grad {g0:.1f}->{g1:.1f}")

    print("\n" + "=" * 50)
    print(f"Summary (h={h}, T={T}):")
    for red in ["mean", "sum"]:
        r = results.get(red, {})
        ok = r.get("ok", False)
        blowup = r.get("blowup", True)
        verdict = "STABLE" if (ok and not blowup) else "UNSTABLE"
        print(f"  ce_reduction={red}: {verdict}")
    print("=" * 50)

    mean_ok = results.get("mean", {}).get("ok") and not results.get("mean", {}).get("blowup")
    sum_ok = results.get("sum", {}).get("ok") and not results.get("sum", {}).get("blowup")

    if mean_ok and not sum_ok:
        print("\n-> Use ce_reduction=mean (default). Sum CE blows up at this h.")
    elif sum_ok and not mean_ok:
        print("\n-> Use ce_reduction=sum. Mean CE had issues.")
    elif mean_ok and sum_ok:
        print("\n-> Both stable. Prefer mean (allows larger h, standard in ML).")
    else:
        print("\n-> Both had issues. Check logs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
