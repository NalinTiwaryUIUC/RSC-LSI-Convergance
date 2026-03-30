#!/usr/bin/env python3
"""
Post-geometry predictive escape times.

For each chain, define geometry escape (same metric as preset d_sqrt grid):

  τ_geom(c_d) = inf{ t : dist_to_ref_over_sqrt_d(t) >= c_d }

Then predictive events only *after* that time:

  τ_NLL|geom(c_d, c_n)   = inf{ t >= τ_geom : nll_probe_mean(t) >= c_n }
  τ_margin|geom(c_d, c_m) = inf{ t >= τ_geom : f_margin(t) <= c_m }

Report per group (init family):
  - fraction of chains with finite τ_pred|geom
  - mean and std of Δτ = τ_pred|geom − τ_geom
  - Gelman–Rubin R̂ on aligned f_nll traces starting at first saved sample with step >= τ_pred|geom

Example (common absolute predictive grid):

  python scripts/analyze_post_geom_predictive.py \\
    --runs-dir experiments/runs --parent-glob 'w4_*_ul_initI*_chain*' --auto-group \\
    --geom-d 0.05,0.10 \\
    --abs-nll-ge=1.45,1.55,1.70 \\
    --abs-f-margin-le=-0.20,-0.30,-0.38 \\
    --out-csv experiments/summaries/postgeom_w4.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_escape_diagnostic import (  # noqa: E402
    chain_prefix,
    first_crossing,
    load_iter,
    parse_csv_floats,
)
from report_chain_convergence import gelman_rubin_rhat  # noqa: E402


def _thr_label(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _thr_label_signed(x: float) -> str:
    neg = x < 0
    ax = abs(x)
    body = f"{ax:g}".replace(".", "p")
    return f"m{body}" if neg else body


def tau_geom_sqrt_d(recs: list[dict[str, Any]], c_d: float) -> int | None:
    """First step with dist_to_ref_over_sqrt_d >= c_d."""
    return first_crossing(
        recs,
        lambda r, cc=c_d: (
            isinstance(r.get("dist_to_ref_over_sqrt_d"), (int, float))
            and math.isfinite(float(r["dist_to_ref_over_sqrt_d"]))
            and float(r["dist_to_ref_over_sqrt_d"]) >= cc
        ),
    )


def tau_nll_cond_geom(
    recs: list[dict[str, Any]], tau_geom: int | None, c_n: float
) -> int | None:
    if tau_geom is None:
        return None
    return first_crossing(
        recs,
        lambda r, tg=tau_geom, cn=c_n: (
            int(r.get("step", 0)) >= tg
            and isinstance(r.get("nll_probe_mean"), (int, float))
            and math.isfinite(float(r["nll_probe_mean"]))
            and float(r["nll_probe_mean"]) >= cn
        ),
    )


def tau_margin_cond_geom(
    recs: list[dict[str, Any]], tau_geom: int | None, c_m: float
) -> int | None:
    if tau_geom is None:
        return None
    return first_crossing(
        recs,
        lambda r, tg=tau_geom, cm=c_m: (
            int(r.get("step", 0)) >= tg
            and isinstance(r.get("f_margin"), (int, float))
            and math.isfinite(float(r["f_margin"]))
            and float(r["f_margin"]) <= cm
        ),
    )


def criterion_id_geom_nll(c_d: float, c_n: float) -> str:
    return f"geom_d{_thr_label(c_d)}_nll_ge_{_thr_label_signed(c_n)}"


def criterion_id_geom_margin(c_d: float, c_m: float) -> str:
    return f"geom_d{_thr_label(c_d)}_f_margin_le_{_thr_label_signed(c_m)}"


def all_criterion_ids(
    geom_d: tuple[float, ...],
    abs_nll_ge: tuple[float, ...],
    abs_f_margin_le: tuple[float, ...],
) -> tuple[str, ...]:
    ids: list[str] = []
    for c_d in geom_d:
        for c_n in abs_nll_ge:
            ids.append(criterion_id_geom_nll(c_d, c_n))
        for c_m in abs_f_margin_le:
            ids.append(criterion_id_geom_margin(c_d, c_m))
    return tuple(ids)


def compute_chain_taus(
    recs: list[dict[str, Any]],
    geom_d: tuple[float, ...],
    abs_nll_ge: tuple[float, ...],
    abs_f_margin_le: tuple[float, ...],
) -> dict[str, tuple[int | None, int | None, int | None]]:
    """
    Returns per criterion: (tau_geom, tau_pred, delta_tau).
    delta_tau is None if either tau is None.
    """
    tau_geom_by_cd: dict[float, int | None] = {
        c_d: tau_geom_sqrt_d(recs, c_d) for c_d in geom_d
    }
    out: dict[str, tuple[int | None, int | None, int | None]] = {}
    for c_d in geom_d:
        tg = tau_geom_by_cd[c_d]
        for c_n in abs_nll_ge:
            cid = criterion_id_geom_nll(c_d, c_n)
            tp = tau_nll_cond_geom(recs, tg, c_n)
            dt = (tp - tg) if (tg is not None and tp is not None) else None
            out[cid] = (tg, tp, dt)
        for c_m in abs_f_margin_le:
            cid = criterion_id_geom_margin(c_d, c_m)
            tp = tau_margin_cond_geom(recs, tg, c_m)
            dt = (tp - tg) if (tg is not None and tp is not None) else None
            out[cid] = (tg, tp, dt)
    return out


def _stat_mean_std(vals: list[int | None]) -> tuple[float, float]:
    finite = [v for v in vals if v is not None]
    if not finite:
        return float("nan"), float("nan")
    a = np.array(finite, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="τ_pred|geom and Δτ after dist_to_ref_over_sqrt_d geometry escape"
    )
    ap.add_argument("run_dirs", nargs="*", type=str, help="Run directories (with iter_metrics.jsonl)")
    ap.add_argument("--runs-dir", type=str, default="experiments/runs")
    ap.add_argument("--parent-glob", type=str, default=None)
    ap.add_argument("--auto-group", action="store_true")
    ap.add_argument(
        "--geom-d",
        type=str,
        default="0.05,0.10",
        help="Comma-separated c_d for τ_geom (dist_to_ref_over_sqrt_d >= c_d)",
    )
    ap.add_argument(
        "--abs-nll-ge",
        type=str,
        default="1.45,1.55,1.70",
        metavar="C1,C2,...",
        help="Absolute NLL thresholds (nll_probe_mean >= c). Use --abs-nll-ge=1.45,...",
    )
    ap.add_argument(
        "--abs-f-margin-le",
        type=str,
        default="-0.20,-0.30,-0.38",
        metavar="C1,C2,...",
        help="Margin thresholds (f_margin <= c). Use --abs-f-margin-le=-0.2,...",
    )
    ap.add_argument("--aligned-probe", type=str, default="f_nll")
    ap.add_argument("--min-aligned-length", type=int, default=4)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    geom_d = parse_csv_floats(args.geom_d)
    abs_nll_ge = parse_csv_floats(args.abs_nll_ge)
    abs_f_margin_le = parse_csv_floats(args.abs_f_margin_le)
    if not geom_d:
        print("No --geom-d values.", file=sys.stderr)
        sys.exit(1)

    crit_ids = all_criterion_ids(geom_d, abs_nll_ge, abs_f_margin_le)

    runs_base = Path(args.runs_dir)
    if not runs_base.is_absolute():
        runs_base = ROOT / runs_base

    paths: list[Path] = []
    if args.run_dirs:
        paths = [Path(p) for p in args.run_dirs]
    elif args.parent_glob:
        paths = sorted(p for p in runs_base.glob(args.parent_glob) if p.is_dir())
    if not paths:
        print("No run directories.", file=sys.stderr)
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
    min_al = max(1, args.min_aligned_length)

    for gname, gpaths in groups.items():
        print(f"\n=== Group: {gname} ({len(gpaths)} runs) ===")

        taus_geom: dict[str, list[int | None]] = {c: [] for c in crit_ids}
        taus_pred: dict[str, list[int | None]] = {c: [] for c in crit_ids}
        deltas: dict[str, list[int | None]] = {c: [] for c in crit_ids}
        aligned_mats: dict[str, list[np.ndarray]] = defaultdict(list)

        for p in gpaths:
            recs = load_iter(p / "iter_metrics.jsonl")
            if not recs:
                print(f"  skip (no iter_metrics): {p.name}")
                for cid in crit_ids:
                    taus_geom[cid].append(None)
                    taus_pred[cid].append(None)
                    deltas[cid].append(None)
                continue

            tmap = compute_chain_taus(recs, geom_d, abs_nll_ge, abs_f_margin_le)
            for cid in crit_ids:
                tg, tp, dt = tmap[cid]
                taus_geom[cid].append(tg)
                taus_pred[cid].append(tp)
                deltas[cid].append(dt)
            preview = ", ".join(
                f"{cid} geom={tmap[cid][0]} pred={tmap[cid][1]}" for cid in crit_ids[:3]
            )
            print(f"  {p.name}: {preview}" + (" …" if len(crit_ids) > 3 else ""))

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
                    if steps is not None and len(steps) == len(trace):
                        for cid in crit_ids:
                            _, tp, _ = tmap[cid]
                            if tp is None:
                                continue
                            idxs = [i for i in range(len(steps)) if int(steps[i]) >= tp]
                            if not idxs:
                                continue
                            k0 = idxs[0]
                            aligned_mats[cid].append(trace[k0:])

        for cid in crit_ids:
            tg_list = taus_geom[cid]
            tp_list = taus_pred[cid]
            dt_list = deltas[cid]
            n_tot = len(tp_list)
            n_fin_pred = sum(1 for v in tp_list if v is not None)
            n_fin_geom = sum(1 for v in tg_list if v is not None)
            m_g, s_g = _stat_mean_std(tg_list)
            m_p, s_p = _stat_mean_std(tp_list)
            m_d, s_d = _stat_mean_std(dt_list)
            frac = n_fin_pred / n_tot if n_tot else float("nan")
            print(
                f"  {cid}: finite pred {n_fin_pred}/{n_tot} (geom {n_fin_geom}/{n_tot}); "
                f"Δτ mean={m_d:.4g} std={s_d:.4g}; τ_geom mean={m_g:.4g} τ_pred mean={m_p:.4g}"
            )
            rows_out.append(
                {
                    "row_kind": "summary",
                    "group": gname,
                    "chain_run": "",
                    "criterion": cid,
                    "tau_geom": "",
                    "tau_pred": "",
                    "delta_tau": "",
                    "frac_finite_pred": frac,
                    "frac_finite_geom": n_fin_geom / n_tot if n_tot else float("nan"),
                    "tau_geom_mean": m_g,
                    "tau_geom_std": s_g,
                    "tau_pred_mean": m_p,
                    "tau_pred_std": s_p,
                    "delta_tau_mean": m_d,
                    "delta_tau_std": s_d,
                    "n_chains": n_tot,
                    "n_finite_pred": n_fin_pred,
                    "rhat_aligned": "",
                }
            )

            for i, rp in enumerate(gpaths):
                tg, tp, dt = (
                    taus_geom[cid][i],
                    taus_pred[cid][i],
                    deltas[cid][i],
                )
                rows_out.append(
                    {
                        "row_kind": "chain",
                        "group": gname,
                        "chain_run": rp.name,
                        "criterion": cid,
                        "tau_geom": tg if tg is not None else "",
                        "tau_pred": tp if tp is not None else "",
                        "delta_tau": dt if dt is not None else "",
                        "frac_finite_pred": "",
                        "frac_finite_geom": "",
                        "tau_geom_mean": "",
                        "tau_geom_std": "",
                        "tau_pred_mean": "",
                        "tau_pred_std": "",
                        "delta_tau_mean": "",
                        "delta_tau_std": "",
                        "n_chains": "",
                        "n_finite_pred": "",
                        "rhat_aligned": "",
                    }
                )

        for cid in crit_ids:
            mats = aligned_mats.get(cid, [])
            if len(mats) < 2:
                print(
                    f"  R̂ aligned ({cid}, {args.aligned_probe}): "
                    f"need ≥2 chains with valid τ_pred, got {len(mats)}"
                )
                rows_out.append(
                    {
                        "row_kind": "rhat",
                        "group": gname,
                        "chain_run": "",
                        "criterion": f"rhat_aligned_{cid}_{args.aligned_probe}",
                        "tau_geom": "",
                        "tau_pred": "",
                        "delta_tau": "",
                        "frac_finite_pred": "",
                        "frac_finite_geom": "",
                        "tau_geom_mean": "",
                        "tau_geom_std": "",
                        "tau_pred_mean": "",
                        "tau_pred_std": "",
                        "delta_tau_mean": "",
                        "delta_tau_std": "",
                        "n_chains": len(mats),
                        "n_finite_pred": "",
                        "rhat_aligned": "",
                    }
                )
                continue
            n_min = min(len(x) for x in mats)
            if n_min < min_al:
                print(f"  R̂ aligned ({cid}): aligned length {n_min} < {min_al}")
                rows_out.append(
                    {
                        "row_kind": "rhat",
                        "group": gname,
                        "chain_run": "",
                        "criterion": f"rhat_aligned_{cid}_{args.aligned_probe}",
                        "tau_geom": "",
                        "tau_pred": "",
                        "delta_tau": "",
                        "frac_finite_pred": "",
                        "frac_finite_geom": "",
                        "tau_geom_mean": "",
                        "tau_geom_std": "",
                        "tau_pred_mean": "",
                        "tau_pred_std": "",
                        "delta_tau_mean": "",
                        "delta_tau_std": "",
                        "n_chains": len(mats),
                        "n_finite_pred": "",
                        "rhat_aligned": "",
                    }
                )
                continue
            mat = np.stack([x[:n_min] for x in mats], axis=0)
            rh = gelman_rubin_rhat(mat)
            print(f"  R̂ aligned ({cid}, {args.aligned_probe}, n={n_min}): {rh:.4f}")
            rows_out.append(
                {
                    "row_kind": "rhat",
                    "group": gname,
                    "chain_run": "",
                    "criterion": f"rhat_aligned_{cid}_{args.aligned_probe}",
                    "tau_geom": "",
                    "tau_pred": "",
                    "delta_tau": "",
                    "frac_finite_pred": "",
                    "frac_finite_geom": "",
                    "tau_geom_mean": "",
                    "tau_geom_std": "",
                    "tau_pred_mean": "",
                    "tau_pred_std": "",
                    "delta_tau_mean": "",
                    "delta_tau_std": "",
                    "n_chains": len(mats),
                    "n_finite_pred": "",
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
            "tau_geom",
            "tau_pred",
            "delta_tau",
            "frac_finite_pred",
            "frac_finite_geom",
            "tau_geom_mean",
            "tau_geom_std",
            "tau_pred_mean",
            "tau_pred_std",
            "delta_tau_mean",
            "delta_tau_std",
            "n_chains",
            "n_finite_pred",
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
