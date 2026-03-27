#!/usr/bin/env python3
"""
Post-hoc escape-time diagnostics from completed runs.

For each chain (run dir): read iter_metrics.jsonl, compute first step τ where criteria fire.
Summarize mean/var of τ across chains in a group (same init / width, different chain id).

Optional: Gelman–Rubin R̂ on saved-sample probes after aligning traces to start at the save index
corresponding to each chain's τ (per criterion).

Usage:
  python scripts/analyze_escape_diagnostic.py \\
      experiments/runs/w1_n512_h5e-06_T40000_a0.3_b1p0_g3p0_ul_initI1_chain0 \\
      experiments/runs/w1_n512_h5e-06_T40000_a0.3_b1p0_g3p0_ul_initI1_chain1 \\
      ...

  python scripts/analyze_escape_diagnostic.py --parent-glob 'experiments/runs/w1_*initI1*' --auto-group
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_chain_convergence import gelman_rubin_rhat  # noqa: E402


def load_iter(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for k, v in list(rec.items()):
            if isinstance(v, str) and v.lower() == "nan":
                rec[k] = float("nan")
        out.append(rec)
    return out


def first_crossing(
    recs: list[dict[str, Any]],
    predicate,
) -> int | None:
    """First record step where predicate(rec) is True."""
    recs = sorted(recs, key=lambda r: int(r.get("step", 0)))
    for r in recs:
        step = int(r.get("step", 0))
        try:
            if predicate(r):
                return step
        except (TypeError, ValueError):
            continue
    return None


def chain_prefix(name: str) -> str:
    return re.sub(r"_chain\d+$", "", name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Escape times and escape-aligned R̂ from run dirs")
    ap.add_argument(
        "run_dirs",
        nargs="*",
        type=str,
        help="Run directories (each with iter_metrics.jsonl, run_config.yaml)",
    )
    ap.add_argument(
        "--parent-glob",
        type=str,
        default=None,
        help="Glob pattern under --runs-dir (e.g. 'w1_*initI1*'); use with --auto-group or alone",
    )
    ap.add_argument(
        "--runs-dir",
        type=str,
        default="experiments/runs",
        help="Base directory for --parent-glob (relative paths are project-rooted)",
    )
    ap.add_argument(
        "--auto-group",
        action="store_true",
        help="Group paths by stripping _chainN from directory name; one group per prefix",
    )
    ap.add_argument("--thresh-d-sqrt", type=float, default=0.05, help="dist_to_ref_over_sqrt_d > this")
    ap.add_argument("--thresh-ou", type=float, default=0.05, help="dist_to_ref_over_ou_radius > this")
    ap.add_argument(
        "--f-margin-max",
        type=float,
        default=None,
        help="First step with f_margin < this (probe margin from iter_metrics if present)",
    )
    ap.add_argument(
        "--nll-rise-frac",
        type=float,
        default=0.25,
        help="nll_probe_mean > (1+this) * first logged nll_probe_mean",
    )
    ap.add_argument(
        "--aligned-probe",
        type=str,
        default="f_nll",
        help="Key in samples_metrics.npz for escape-aligned R̂ (default f_nll)",
    )
    ap.add_argument("--out-csv", type=str, default=None, help="Write summary CSV path")
    args = ap.parse_args()

    runs_base = Path(args.runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base

    paths: list[Path] = []
    if args.run_dirs:
        paths = [Path(p) for p in args.run_dirs]
    elif args.parent_glob:
        paths = sorted(p for p in runs_base.glob(args.parent_glob) if p.is_dir())

    if not paths:
        print("No run directories given.", file=sys.stderr)
        sys.exit(1)

    groups: dict[str, list[Path]] = defaultdict(list)
    if args.auto_group:
        for p in paths:
            groups[chain_prefix(p.name)].append(p)
        for k in groups:
            groups[k] = sorted(groups[k], key=lambda x: x.name)
    else:
        groups["group"] = sorted(paths, key=lambda x: x.name)

    rows_out: list[dict[str, Any]] = []

    for gname, gpaths in groups.items():
        print(f"\n=== Group: {gname} ({len(gpaths)} runs) ===")

        taus: dict[str, list[int | None]] = {
            "d_sqrt": [],
            "ou": [],
            "f_margin": [],
            "nll_rise": [],
        }

        aligned_mats: dict[str, list[np.ndarray]] = defaultdict(list)

        for p in gpaths:
            recs = load_iter(p / "iter_metrics.jsonl")
            if not recs:
                print(f"  skip (no iter_metrics): {p}")
                for k in taus:
                    taus[k].append(None)
                for crit_name in ("d_sqrt", "ou", "f_margin", "nll_rise"):
                    rows_out.append(
                        {
                            "row_kind": "chain",
                            "group": gname,
                            "chain_run": p.name,
                            "criterion": crit_name,
                            "tau_escape": "",
                            "tau_mean": "",
                            "tau_std": "",
                            "n_chains": "",
                            "rhat_aligned": "",
                        }
                    )
                continue

            recs_sorted = sorted(recs, key=lambda r: int(r.get("step", 0)))
            first_nll = None
            for r in recs_sorted:
                v = r.get("nll_probe_mean")
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    first_nll = float(v)
                    break

            t_d = first_crossing(
                recs,
                lambda r: float(r.get("dist_to_ref_over_sqrt_d", float("nan"))) > args.thresh_d_sqrt
                if isinstance(r.get("dist_to_ref_over_sqrt_d"), (int, float))
                else False,
            )
            t_ou = first_crossing(
                recs,
                lambda r: float(r.get("dist_to_ref_over_ou_radius", float("nan"))) > args.thresh_ou
                if isinstance(r.get("dist_to_ref_over_ou_radius"), (int, float))
                else False,
            )
            t_fm = None
            if args.f_margin_max is not None:
                t_fm = first_crossing(
                    recs,
                    lambda r: float(r.get("f_margin", float("nan"))) < args.f_margin_max
                    if isinstance(r.get("f_margin"), (int, float))
                    else False,
                )

            t_nll = None
            if first_nll is not None and first_nll > 0:
                thr = first_nll * (1.0 + args.nll_rise_frac)

                def _nll_rise(r):
                    v = r.get("nll_probe_mean")
                    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                        return False
                    return float(v) > thr

                t_nll = first_crossing(recs, _nll_rise)

            taus["d_sqrt"].append(t_d)
            taus["ou"].append(t_ou)
            taus["f_margin"].append(t_fm)
            taus["nll_rise"].append(t_nll)

            print(
                f"  {p.name}: τ(d_sqrt)={t_d} τ(ou)={t_ou} τ(f_margin)={t_fm} τ(nll_rise)={t_nll}"
            )

            for crit_name, tau_val in (
                ("d_sqrt", t_d),
                ("ou", t_ou),
                ("f_margin", t_fm),
                ("nll_rise", t_nll),
            ):
                rows_out.append(
                    {
                        "row_kind": "chain",
                        "group": gname,
                        "chain_run": p.name,
                        "criterion": crit_name,
                        "tau_escape": tau_val if tau_val is not None else "",
                        "tau_mean": "",
                        "tau_std": "",
                        "n_chains": "",
                        "rhat_aligned": "",
                    }
                )

            # Escape-aligned sample traces
            npz_path = p / "samples_metrics.npz"
            if npz_path.exists() and args.aligned_probe:
                data = np.load(npz_path)
                if args.aligned_probe not in data:
                    continue
                trace = np.asarray(data[args.aligned_probe], dtype=np.float64)
                steps = np.asarray(data["step"], dtype=np.int64) if "step" in data.files else None
                for crit, t_esc in [
                    ("d_sqrt", t_d),
                    ("ou", t_ou),
                    ("f_margin", t_fm),
                    ("nll_rise", t_nll),
                ]:
                    if t_esc is None or steps is None or len(steps) != len(trace):
                        continue
                    # first save index with step >= t_esc
                    idxs = [i for i in range(len(steps)) if int(steps[i]) >= t_esc]
                    if not idxs:
                        continue
                    k0 = idxs[0]
                    aligned_mats[crit].append(trace[k0:])

        def _stat(vals: list[int | None]) -> tuple[float, float]:
            finite = [v for v in vals if v is not None]
            if not finite:
                return float("nan"), float("nan")
            a = np.array(finite, dtype=float)
            return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

        for crit in taus:
            m, s = _stat(taus[crit])
            print(f"  τ {crit}: mean={m:.4g} std={s:.4g} (finite {sum(1 for v in taus[crit] if v is not None)}/{len(taus[crit])})")
            rows_out.append(
                {
                    "row_kind": "summary",
                    "group": gname,
                    "chain_run": "",
                    "criterion": crit,
                    "tau_escape": "",
                    "tau_mean": m,
                    "tau_std": s,
                    "n_chains": len(taus[crit]),
                    "rhat_aligned": "",
                }
            )

        # R̂ aligned per criterion
        for crit, mats in aligned_mats.items():
            if len(mats) < 2:
                print(f"  R̂ aligned ({crit}, {args.aligned_probe}): need ≥2 chains with valid escape, got {len(mats)}")
                continue
            n_min = min(len(x) for x in mats)
            if n_min < 4:
                print(f"  R̂ aligned ({crit}): aligned length {n_min} < 4")
                continue
            mat = np.stack([x[:n_min] for x in mats], axis=0)
            rh = gelman_rubin_rhat(mat)
            print(f"  R̂ aligned ({crit}, {args.aligned_probe}, n={n_min}): {rh:.4f}")
            rows_out.append(
                {
                    "row_kind": "rhat",
                    "group": gname,
                    "chain_run": "",
                    "criterion": f"rhat_aligned_{crit}_{args.aligned_probe}",
                    "tau_escape": "",
                    "tau_mean": "",
                    "tau_std": "",
                    "n_chains": len(mats),
                    "rhat_aligned": rh,
                }
            )

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "row_kind",
            "group",
            "chain_run",
            "criterion",
            "tau_escape",
            "tau_mean",
            "tau_std",
            "n_chains",
            "rhat_aligned",
        ]
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows_out:
                w.writerow(row)
        print("\nWrote", outp)


if __name__ == "__main__":
    main()
