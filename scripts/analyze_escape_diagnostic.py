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
  # No cutoffs: τ from argmax/min of logged scalars (finite τ, no NaN group means)
  python scripts/analyze_escape_diagnostic.py ... --tau-from extremal
  # Threshold crossings, but fill any missing τ with extremal step (default --fill-missing-tau extremal)
  python scripts/analyze_escape_diagnostic.py ... --tau-from threshold --thresh-d-sqrt 0.05

  # Interpretable grid (per-chain nll_0, m_0 from first logged row; fixed geometry constants):
  #   d/sqrt(d) ∈ {0.05,0.10,0.15}, ou_radius ∈ {0.03,0.06,0.09},
  #   nll ≥ n0+{0.25,0.50}, f_margin ≤ m0-{0.5,1.0}
  python scripts/analyze_escape_diagnostic.py ... --auto-group --threshold-grid preset --fill-missing-tau none

  # Same preset grid plus absolute predictive levels (pick c from pooled iter range across inits):
  python scripts/analyze_escape_diagnostic.py ... --threshold-grid preset \\
      --abs-nll-ge=1.45,1.50,1.55 --abs-f-margin-le=-0.2,-0.25,-0.3
  # Note: use --opt=value when the list starts with '-' (argparse otherwise treats it as a flag).
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


def last_logged_step(recs: list[dict[str, Any]]) -> int | None:
    if not recs:
        return None
    return max(int(r.get("step", 0)) for r in recs)


def first_extremal_step(
    recs: list[dict[str, Any]], key: str, mode: str
) -> int | None:
    """First step attaining global min or max of `key` over logged rows (finite values only)."""
    recs = sorted(recs, key=lambda r: int(r.get("step", 0)))
    best_val: float | None = None
    best_step: int | None = None
    for r in recs:
        v = r.get(key)
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            continue
        fv = float(v)
        st = int(r.get("step", 0))
        if best_val is None:
            best_val, best_step = fv, st
        elif mode == "max" and fv > best_val:
            best_val, best_step = fv, st
        elif mode == "min" and fv < best_val:
            best_val, best_step = fv, st
    return best_step


def chain_prefix(name: str) -> str:
    return re.sub(r"_chain\d+$", "", name)


def _thr_label(x: float) -> str:
    """Stable id fragment: 0.05 -> 0p05, 1.0 -> 1."""
    return f"{x:g}".replace(".", "p")


def _thr_label_signed(x: float) -> str:
    """Stable id for signed levels: 1.4 -> 1p4, -0.25 -> m0p25."""
    neg = x < 0
    ax = abs(x)
    body = f"{ax:g}".replace(".", "p")
    return f"m{body}" if neg else body


def _unique_floats_preserve(xs: list[float]) -> tuple[float, ...]:
    seen: set[float] = set()
    out: list[float] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)


def parse_csv_floats(s: str | None) -> tuple[float, ...]:
    """Parse '1.5, -0.2' -> (1.5, -0.2); dedupe preserving order."""
    if s is None or not str(s).strip():
        return ()
    vals: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return _unique_floats_preserve(vals)


PRESET_GRID_D_SQRT = (0.05, 0.10, 0.15)
PRESET_GRID_OU = (0.03, 0.06, 0.09)
PRESET_GRID_NLL_PLUS = (0.25, 0.50)
PRESET_GRID_MARGIN_DROP = (0.5, 1.0)


def preset_criterion_ids() -> tuple[str, ...]:
    """Ordered criterion keys for --threshold-grid preset (must match compute_preset_taus_for_chain)."""
    ids: list[str] = []
    for c in PRESET_GRID_D_SQRT:
        ids.append(f"d_sqrt_ge_{_thr_label(c)}")
    for c in PRESET_GRID_OU:
        ids.append(f"ou_ge_{_thr_label(c)}")
    for d in PRESET_GRID_NLL_PLUS:
        ids.append(f"nll_ge_init_plus_{_thr_label(d)}")
    for m in PRESET_GRID_MARGIN_DROP:
        ids.append(f"f_margin_le_init_minus_{_thr_label(m)}")
    return tuple(ids)


def extended_preset_criterion_ids(
    abs_nll_ge: tuple[float, ...] = (),
    abs_f_margin_le: tuple[float, ...] = (),
) -> tuple[str, ...]:
    """Preset relative+geometry ids, then absolute predictive criteria (order matches compute_preset_taus_for_chain)."""
    ids = list(preset_criterion_ids())
    for c in abs_nll_ge:
        ids.append(f"nll_abs_ge_{_thr_label_signed(c)}")
    for c in abs_f_margin_le:
        ids.append(f"f_margin_abs_le_{_thr_label_signed(c)}")
    return tuple(ids)


def fill_missing_tau_single(
    t: int | None,
    policy: str,
    recs: list[dict[str, Any]],
    last_s: int | None,
    kind: str,
) -> int | None:
    """Apply --fill-missing-tau for one scalar criterion (preset or legacy)."""
    if t is not None:
        return t
    if policy == "none":
        return None
    if policy == "last":
        return last_s
    if kind == "d_sqrt":
        return first_extremal_step(recs, "dist_to_ref_over_sqrt_d", "max")
    if kind == "ou":
        return first_extremal_step(recs, "dist_to_ref_over_ou_radius", "max")
    if kind == "nll":
        return first_extremal_step(recs, "nll_probe_mean", "max")
    if kind == "f_margin":
        return first_extremal_step(recs, "f_margin", "min")
    return last_s


def compute_preset_taus_for_chain(
    recs: list[dict[str, Any]],
    *,
    fill_missing_tau: str,
    abs_nll_ge: tuple[float, ...] = (),
    abs_f_margin_le: tuple[float, ...] = (),
) -> dict[str, int | None]:
    """
    First-crossing τ per criterion. Geometry uses >= fixed c; predictive uses per-chain
    first logged nll_0 and m_0 with fixed additive offsets. Optional absolute predictive
    rules: nll_probe_mean >= c, f_margin <= c (same c for all chains).
    """
    recs_sorted = sorted(recs, key=lambda r: int(r.get("step", 0)))
    last_s = last_logged_step(recs_sorted)
    first_nll: float | None = None
    first_margin: float | None = None
    for r in recs_sorted:
        v = r.get("nll_probe_mean")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            first_nll = float(v)
            break
    for r in recs_sorted:
        v = r.get("f_margin")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            first_margin = float(v)
            break

    out: dict[str, int | None] = {}

    for c in PRESET_GRID_D_SQRT:
        cid = f"d_sqrt_ge_{_thr_label(c)}"
        t = first_crossing(
            recs,
            lambda r, cc=c: (
                isinstance(r.get("dist_to_ref_over_sqrt_d"), (int, float))
                and math.isfinite(float(r["dist_to_ref_over_sqrt_d"]))
                and float(r["dist_to_ref_over_sqrt_d"]) >= cc
            ),
        )
        out[cid] = fill_missing_tau_single(t, fill_missing_tau, recs, last_s, "d_sqrt")

    for c in PRESET_GRID_OU:
        cid = f"ou_ge_{_thr_label(c)}"
        t = first_crossing(
            recs,
            lambda r, cc=c: (
                isinstance(r.get("dist_to_ref_over_ou_radius"), (int, float))
                and math.isfinite(float(r["dist_to_ref_over_ou_radius"]))
                and float(r["dist_to_ref_over_ou_radius"]) >= cc
            ),
        )
        out[cid] = fill_missing_tau_single(t, fill_missing_tau, recs, last_s, "ou")

    for delta in PRESET_GRID_NLL_PLUS:
        cid = f"nll_ge_init_plus_{_thr_label(delta)}"
        if first_nll is None:
            t = None
        else:
            thr = first_nll + delta
            t = first_crossing(
                recs,
                lambda r, th=thr: (
                    isinstance(r.get("nll_probe_mean"), (int, float))
                    and math.isfinite(float(r["nll_probe_mean"]))
                    and float(r["nll_probe_mean"]) >= th
                ),
            )
        out[cid] = fill_missing_tau_single(t, fill_missing_tau, recs, last_s, "nll")

    for drop in PRESET_GRID_MARGIN_DROP:
        cid = f"f_margin_le_init_minus_{_thr_label(drop)}"
        if first_margin is None:
            t = None
        else:
            thr = first_margin - drop
            t = first_crossing(
                recs,
                lambda r, th=thr: (
                    isinstance(r.get("f_margin"), (int, float))
                    and math.isfinite(float(r["f_margin"]))
                    and float(r["f_margin"]) <= th
                ),
            )
        out[cid] = fill_missing_tau_single(t, fill_missing_tau, recs, last_s, "f_margin")

    for c in abs_nll_ge:
        cid = f"nll_abs_ge_{_thr_label_signed(c)}"
        t = first_crossing(
            recs,
            lambda r, th=c: (
                isinstance(r.get("nll_probe_mean"), (int, float))
                and math.isfinite(float(r["nll_probe_mean"]))
                and float(r["nll_probe_mean"]) >= th
            ),
        )
        out[cid] = fill_missing_tau_single(t, fill_missing_tau, recs, last_s, "nll")

    for c in abs_f_margin_le:
        cid = f"f_margin_abs_le_{_thr_label_signed(c)}"
        t = first_crossing(
            recs,
            lambda r, th=c: (
                isinstance(r.get("f_margin"), (int, float))
                and math.isfinite(float(r["f_margin"]))
                and float(r["f_margin"]) <= th
            ),
        )
        out[cid] = fill_missing_tau_single(t, fill_missing_tau, recs, last_s, "f_margin")

    return out


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
    ap.add_argument(
        "--tau-from",
        choices=("threshold", "extremal"),
        default="threshold",
        help="threshold: first crossing of dist / f_margin / nll rules below. "
        "extremal: ignore cutoffs — τ = step of max(d/sqrt d), max(ou), min(f_margin), max(nll); always finite if logged.",
    )
    ap.add_argument(
        "--fill-missing-tau",
        choices=("none", "last", "extremal"),
        default="extremal",
        help="When τ is None after threshold crossing: none=keep None; last=last logged step; "
        "extremal=argmax/argmin same as --tau-from extremal (avoids NaN group means).",
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
        help="Relative rise: nll_probe_mean > (1+this) * first logged nll_probe_mean",
    )
    ap.add_argument(
        "--nll-rise-abs",
        type=float,
        default=0.0,
        help="If >0, also fire when nll_probe_mean > first + this (OR with relative rule). "
        "Helps I3 / flat baselines where 25%% relative rise never occurs.",
    )
    ap.add_argument(
        "--aligned-probe",
        type=str,
        default="f_nll",
        help="Key in samples_metrics.npz for escape-aligned R̂ (default f_nll)",
    )
    ap.add_argument("--out-csv", type=str, default=None, help="Write summary CSV path")
    ap.add_argument(
        "--threshold-grid",
        choices=("none", "preset"),
        default="none",
        help="preset: fixed geometry grid + per-chain nll_0/m_0 predictive grid (see module docstring). "
        "Ignores --tau-from and single-threshold flags for τ definitions.",
    )
    ap.add_argument(
        "--abs-nll-ge",
        type=str,
        default="",
        metavar="C1,C2,...",
        help="With --threshold-grid preset only: comma-separated c; τ = first step with nll_probe_mean >= c. "
        "Prefer --abs-nll-ge=1.45,1.50 so a leading '-' is not parsed as a new flag.",
    )
    ap.add_argument(
        "--abs-f-margin-le",
        type=str,
        default="",
        metavar="C1,C2,...",
        help="With --threshold-grid preset only: comma-separated c; τ = first step with f_margin <= c. "
        "Values are usually negative: use equals form --abs-f-margin-le=-0.2,-0.25 (required; else argparse "
        "treats '-0.2,...' as a missing/invalid option).",
    )
    ap.add_argument(
        "--min-aligned-length",
        type=int,
        default=4,
        help="Minimum post-τ aligned samples per chain required to report escape-aligned R̂.",
    )
    args = ap.parse_args()
    abs_nll_ge = parse_csv_floats(args.abs_nll_ge)
    abs_f_margin_le = parse_csv_floats(args.abs_f_margin_le)

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

        if args.threshold_grid == "preset":
            criterion_ids = list(
                extended_preset_criterion_ids(
                    abs_nll_ge=abs_nll_ge,
                    abs_f_margin_le=abs_f_margin_le,
                )
            )
            taus = {cid: [] for cid in criterion_ids}
        else:
            criterion_ids = None
            taus = {
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
                crit_list = criterion_ids if criterion_ids is not None else (
                    "d_sqrt",
                    "ou",
                    "f_margin",
                    "nll_rise",
                )
                for crit_name in crit_list:
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

            if args.threshold_grid == "preset":
                tau_dict = compute_preset_taus_for_chain(
                    recs,
                    fill_missing_tau=args.fill_missing_tau,
                    abs_nll_ge=abs_nll_ge,
                    abs_f_margin_le=abs_f_margin_le,
                )
                assert criterion_ids is not None
                for cid in criterion_ids:
                    taus[cid].append(tau_dict[cid])
                print(f"  {p.name}: preset threshold-grid τ ({len(criterion_ids)} criteria)")
                for crit_name in criterion_ids:
                    tau_val = tau_dict[crit_name]
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
                npz_path = p / "samples_metrics.npz"
                if npz_path.exists() and args.aligned_probe:
                    data = np.load(npz_path)
                    if args.aligned_probe in data:
                        trace = np.asarray(data[args.aligned_probe], dtype=np.float64)
                        steps = (
                            np.asarray(data["step"], dtype=np.int64)
                            if "step" in data.files
                            else None
                        )
                        for cid in criterion_ids:
                            t_esc = tau_dict[cid]
                            if t_esc is None or steps is None or len(steps) != len(trace):
                                continue
                            idxs = [i for i in range(len(steps)) if int(steps[i]) >= t_esc]
                            if not idxs:
                                continue
                            k0 = idxs[0]
                            aligned_mats[cid].append(trace[k0:])
                continue

            if args.tau_from == "extremal":
                t_d = first_extremal_step(recs, "dist_to_ref_over_sqrt_d", "max")
                t_ou = first_extremal_step(recs, "dist_to_ref_over_ou_radius", "max")
                t_fm = first_extremal_step(recs, "f_margin", "min")
                t_nll = first_extremal_step(recs, "nll_probe_mean", "max")
            else:
                t_d = first_crossing(
                    recs,
                    lambda r: float(r.get("dist_to_ref_over_sqrt_d", float("nan")))
                    > args.thresh_d_sqrt
                    if isinstance(r.get("dist_to_ref_over_sqrt_d"), (int, float))
                    else False,
                )
                t_ou = first_crossing(
                    recs,
                    lambda r: float(r.get("dist_to_ref_over_ou_radius", float("nan")))
                    > args.thresh_ou
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
                if first_nll is not None and math.isfinite(first_nll):
                    thr_rel = first_nll * (1.0 + args.nll_rise_frac)
                    thr_abs = (
                        first_nll + args.nll_rise_abs if args.nll_rise_abs > 0 else None
                    )

                    def _nll_rise(r):
                        v = r.get("nll_probe_mean")
                        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                            return False
                        fv = float(v)
                        if fv > thr_rel:
                            return True
                        if thr_abs is not None and fv > thr_abs:
                            return True
                        return False

                    t_nll = first_crossing(recs, _nll_rise)

            if args.tau_from == "threshold" and args.fill_missing_tau != "none":
                last_s = last_logged_step(recs_sorted)
                if t_d is None:
                    t_d = (
                        first_extremal_step(recs, "dist_to_ref_over_sqrt_d", "max")
                        if args.fill_missing_tau == "extremal"
                        else last_s
                    )
                if t_ou is None:
                    t_ou = (
                        first_extremal_step(recs, "dist_to_ref_over_ou_radius", "max")
                        if args.fill_missing_tau == "extremal"
                        else last_s
                    )
                if t_fm is None:
                    t_fm = (
                        first_extremal_step(recs, "f_margin", "min")
                        if args.fill_missing_tau == "extremal"
                        else last_s
                    )
                if t_nll is None:
                    t_nll = (
                        first_extremal_step(recs, "nll_probe_mean", "max")
                        if args.fill_missing_tau == "extremal"
                        else last_s
                    )

            taus["d_sqrt"].append(t_d)
            taus["ou"].append(t_ou)
            taus["f_margin"].append(t_fm)
            taus["nll_rise"].append(t_nll)

            tag = f" [{args.tau_from}]" if args.tau_from == "extremal" else ""
            print(
                f"  {p.name}: τ(d_sqrt)={t_d} τ(ou)={t_ou} τ(f_margin)={t_fm} τ(nll_rise)={t_nll}{tag}"
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
            n_fin = sum(1 for v in taus[crit] if v is not None)
            n_tot = len(taus[crit])
            if n_fin == 0:
                print(f"  τ {crit}: no finite τ ({n_fin}/{n_tot}); tune thresholds or check iter_metrics")
            else:
                ms = f"{m:.4g}" if math.isfinite(m) else "nan"
                ss = f"{s:.4g}" if math.isfinite(s) else "nan"
                print(f"  τ {crit}: mean={ms} std={ss} (finite {n_fin}/{n_tot})")
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
        min_al = max(1, args.min_aligned_length)
        for crit, mats in aligned_mats.items():
            if len(mats) < 2:
                print(f"  R̂ aligned ({crit}, {args.aligned_probe}): need ≥2 chains with valid escape, got {len(mats)}")
                continue
            n_min = min(len(x) for x in mats)
            if n_min < min_al:
                print(f"  R̂ aligned ({crit}): aligned length {n_min} < {min_al}")
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
