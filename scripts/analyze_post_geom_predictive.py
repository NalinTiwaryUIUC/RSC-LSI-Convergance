#!/usr/bin/env python3
"""
Post-geometry predictive escape times.

For each chain, define geometry escape (same metric as preset d_sqrt grid):

  τ_geom(c_d) = inf{ t : dist_to_ref_over_sqrt_d(t) >= c_d }

Then predictive events only *after* that time:

  τ_NLL|geom(c_d, c_n)   = inf{ t >= τ_geom : nll_probe_mean(t) >= c_n }
  τ_margin|geom(c_d, c_m) = inf{ t >= τ_geom : f_margin(t) <= c_m }

Optional: inside each event-aligned window, scan **tail fractions** (e.g. last 80%, …, 25%)
of the aligned saves; report R̂, drift_z, ESS, ESS/T_phys per suffix and the first suffix
(in that order) with R̂ ≤ --window-stab-rhat and max drift_z ≤ --window-stab-drift-z.

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
from report_chain_convergence import (  # noqa: E402
    drift_z_analysis_window,
    gelman_rubin_rhat,
    grad_evals_for_step_span,
    multichain_ess_bulk_tail,
    physical_time_span_h,
)


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


def _first_saved_index_at_or_after(steps: np.ndarray, start_step: int) -> int | None:
    idxs = np.nonzero(steps >= int(start_step))[0]
    if len(idxs) == 0:
        return None
    return int(idxs[0])


def _extract_window_from_start(
    trace: np.ndarray,
    steps: np.ndarray,
    start_step: int,
    window_len_saves: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract analysis window beginning at first saved sample with step >= start_step.
    If window_len_saves <= 0, use full suffix.
    """
    k0 = _first_saved_index_at_or_after(steps, start_step)
    if k0 is None:
        return None
    if window_len_saves > 0:
        k1 = min(len(trace), k0 + int(window_len_saves))
    else:
        k1 = len(trace)
    if (k1 - k0) < 2:
        return None
    return np.asarray(trace[k0:k1], dtype=np.float64), np.asarray(steps[k0:k1], dtype=np.int64)


def _safe_float(x: float) -> float:
    return float(x) if (x is not None and math.isfinite(float(x))) else float("nan")


def _aligned_suffix_start_idx(n_min: int, frac: float) -> int:
    """Start column index so ``mat[:, start:]`` is the last ``frac`` of the aligned window."""
    if not (0.0 < frac <= 1.0):
        return 0
    start = int(math.floor(n_min * (1.0 - frac)))
    return max(0, min(start, n_min - 2))


def compute_aligned_suffix_metrics(
    mat: np.ndarray,
    step_mat: np.ndarray,
    start: int,
    h_val: float,
    is_underdamped: bool,
    min_draws: int,
) -> dict[str, Any] | None:
    """
    Multi-chain diagnostics on ``mat[:, start:]`` (aligned chains × draws).
    Returns None if the suffix is too short.
    """
    sub = mat[:, start:]
    sub_steps = step_mat[:, start:]
    ncol = sub.shape[1]
    need = max(4, min_draws)
    if ncol < need:
        return None
    if not np.all(np.isfinite(sub)):
        return None
    rh = gelman_rubin_rhat(sub)
    dz_list = [drift_z_analysis_window(sub[i, :]) for i in range(sub.shape[0])]
    dz_f = [z for z in dz_list if math.isfinite(z)]
    dz_mean = float(np.mean(dz_f)) if dz_f else float("nan")
    dz_max = float(np.max(dz_f)) if dz_f else float("nan")
    eb, et = multichain_ess_bulk_tail(sub)
    step_first = float(np.mean(sub_steps[:, 0]))
    step_last = float(np.mean(sub_steps[:, -1]))
    t_analysis = physical_time_span_h(h_val, int(round(step_first)), int(round(step_last)))
    grad_span = grad_evals_for_step_span(int(round(step_last - step_first)), is_underdamped)
    ess_per_t = float(eb / t_analysis) if (math.isfinite(eb) and t_analysis > 0) else float("nan")
    ess_per_1e6 = float((eb / grad_span) * 1e6) if (math.isfinite(eb) and grad_span > 0) else float("nan")
    return {
        "window_rhat": rh,
        "drift_z_mean": dz_mean,
        "drift_z_max": dz_max,
        "ess_bulk": eb,
        "ess_tail": et,
        "t_analysis_phys": t_analysis,
        "grad_evals_span": grad_span,
        "ess_per_t_analysis": ess_per_t,
        "ess_per_1e6_grad": ess_per_1e6,
        "suffix_start_idx": start,
        "suffix_n_draws": ncol,
        "step_first_mean": step_first,
        "step_last_mean": step_last,
    }


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
    ap.add_argument(
        "--window-geom-d",
        type=float,
        default=0.05,
        help="Event-window A start: τ_geom(c_d) with this c_d (default 0.05).",
    )
    ap.add_argument(
        "--window-nll-thr",
        type=float,
        default=1.45,
        help="Event-window B start: τ_NLL|geom(window-geom-d, c_n) with this c_n (default 1.45).",
    )
    ap.add_argument(
        "--window-len-saves",
        type=int,
        default=0,
        help="Window length in saved samples after start; <=0 means full suffix.",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Optional markdown report for event-window metrics.",
    )
    ap.add_argument(
        "--window-suffix-fracs",
        type=str,
        default="0.8,0.6,0.5,0.4,0.3,0.25",
        metavar="F1,F2,...",
        help="Inside each aligned window, also report last-F tail metrics (comma-separated). "
        "Empty disables. Order: try largest F first for stabilization pick.",
    )
    ap.add_argument(
        "--disable-window-suffix-scan",
        action="store_true",
        help="Skip tail-fraction scan inside event windows.",
    )
    ap.add_argument(
        "--window-stab-rhat",
        type=float,
        default=1.10,
        metavar="RMAX",
        help="Stabilization gate: R̂ ≤ this for --window-suffix-best row.",
    )
    ap.add_argument(
        "--window-stab-drift-z",
        type=float,
        default=0.5,
        metavar="ZMAX",
        help="Stabilization gate: max per-chain drift_z ≤ this for --window-suffix-best row.",
    )
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    suffix_fracs: tuple[float, ...] = ()
    if not args.disable_window_suffix_scan:
        sfx = [f for f in parse_csv_floats(args.window_suffix_fracs) if 0.0 < f <= 1.0]
        sfx.sort(reverse=True)
        suffix_fracs = tuple(sfx)

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
    window_a_name = f"post_geom_d{_thr_label(args.window_geom_d)}"
    window_b_name = (
        f"post_pred_nll{_thr_label_signed(args.window_nll_thr)}_given_geom_d{_thr_label(args.window_geom_d)}"
    )
    md_lines: list[str] = []
    if args.out_md:
        md_lines.append("# Event-aligned window diagnostics\n")
        md_lines.append(
            f"- Window A: `{window_a_name}` start at `τ_geom({args.window_geom_d:g})`\n"
        )
        md_lines.append(
            f"- Window B: `{window_b_name}` start at "
            f"`τ_NLL|geom({args.window_geom_d:g}, {args.window_nll_thr:g})`\n"
        )
        md_lines.append(
            f"- Window length (saves): `{args.window_len_saves}` "
            "(<=0 means full suffix)\n"
        )
        if suffix_fracs:
            md_lines.append(
                f"- **Suffix scan** (last fraction of each aligned window): "
                f"{', '.join(f'{100 * f:g}%' for f in suffix_fracs)}; "
                f"stabilization pick: first suffix with R̂≤{args.window_stab_rhat:g} "
                f"and max drift_z≤{args.window_stab_drift_z:g} (largest listed fraction tried first).\n"
            )

    for gname, gpaths in groups.items():
        print(f"\n=== Group: {gname} ({len(gpaths)} runs) ===")

        taus_geom: dict[str, list[int | None]] = {c: [] for c in crit_ids}
        taus_pred: dict[str, list[int | None]] = {c: [] for c in crit_ids}
        deltas: dict[str, list[int | None]] = {c: [] for c in crit_ids}
        aligned_mats: dict[str, list[np.ndarray]] = defaultdict(list)
        event_windows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            window_a_name: [],
            window_b_name: [],
        }

        for p in gpaths:
            recs = load_iter(p / "iter_metrics.jsonl")
            if not recs:
                print(f"  skip (no iter_metrics): {p.name}")
                for cid in crit_ids:
                    taus_geom[cid].append(None)
                    taus_pred[cid].append(None)
                    deltas[cid].append(None)
                for wname in event_windows:
                    event_windows[wname].append((np.array([], dtype=np.float64), np.array([], dtype=np.int64)))
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

            # Fixed event-window starts requested for ESS diagnostics
            tau_geom_win = tau_geom_sqrt_d(recs, args.window_geom_d)
            tau_pred_win = tau_nll_cond_geom(recs, tau_geom_win, args.window_nll_thr)

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
                    if steps is not None and len(steps) == len(trace):
                        win_a = (
                            _extract_window_from_start(trace, steps, tau_geom_win, args.window_len_saves)
                            if tau_geom_win is not None
                            else None
                        )
                        win_b = (
                            _extract_window_from_start(trace, steps, tau_pred_win, args.window_len_saves)
                            if tau_pred_win is not None
                            else None
                        )
                        event_windows[window_a_name].append(
                            win_a if win_a is not None else (np.array([], dtype=np.float64), np.array([], dtype=np.int64))
                        )
                        event_windows[window_b_name].append(
                            win_b if win_b is not None else (np.array([], dtype=np.float64), np.array([], dtype=np.int64))
                        )
                    else:
                        event_windows[window_a_name].append((np.array([], dtype=np.float64), np.array([], dtype=np.int64)))
                        event_windows[window_b_name].append((np.array([], dtype=np.float64), np.array([], dtype=np.int64)))
                else:
                    event_windows[window_a_name].append((np.array([], dtype=np.float64), np.array([], dtype=np.int64)))
                    event_windows[window_b_name].append((np.array([], dtype=np.float64), np.array([], dtype=np.int64)))
            else:
                event_windows[window_a_name].append((np.array([], dtype=np.float64), np.array([], dtype=np.int64)))
                event_windows[window_b_name].append((np.array([], dtype=np.float64), np.array([], dtype=np.int64)))

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
                    "window_kind": "",
                    "window_len_saves": "",
                    "n_chains_used": "",
                    "step_first_mean": "",
                    "step_last_mean": "",
                    "window_rhat": "",
                    "drift_z_mean": "",
                    "drift_z_max": "",
                    "ess_bulk": "",
                    "ess_tail": "",
                    "t_analysis_phys": "",
                    "grad_evals_span": "",
                    "ess_per_t_analysis": "",
                    "ess_per_1e6_grad": "",
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
                        "window_kind": "",
                        "window_len_saves": "",
                        "n_chains_used": "",
                        "step_first_mean": "",
                        "step_last_mean": "",
                        "window_rhat": "",
                        "drift_z_mean": "",
                        "drift_z_max": "",
                        "ess_bulk": "",
                        "ess_tail": "",
                        "t_analysis_phys": "",
                        "grad_evals_span": "",
                        "ess_per_t_analysis": "",
                        "ess_per_1e6_grad": "",
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
                        "window_kind": "",
                        "window_len_saves": "",
                        "n_chains_used": "",
                        "step_first_mean": "",
                        "step_last_mean": "",
                        "window_rhat": "",
                        "drift_z_mean": "",
                        "drift_z_max": "",
                        "ess_bulk": "",
                        "ess_tail": "",
                        "t_analysis_phys": "",
                        "grad_evals_span": "",
                        "ess_per_t_analysis": "",
                        "ess_per_1e6_grad": "",
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
                        "window_kind": "",
                        "window_len_saves": "",
                        "n_chains_used": "",
                        "step_first_mean": "",
                        "step_last_mean": "",
                        "window_rhat": "",
                        "drift_z_mean": "",
                        "drift_z_max": "",
                        "ess_bulk": "",
                        "ess_tail": "",
                        "t_analysis_phys": "",
                        "grad_evals_span": "",
                        "ess_per_t_analysis": "",
                        "ess_per_1e6_grad": "",
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
                    "window_kind": "",
                    "window_len_saves": "",
                    "n_chains_used": "",
                    "step_first_mean": "",
                    "step_last_mean": "",
                    "window_rhat": "",
                    "drift_z_mean": "",
                    "drift_z_max": "",
                    "ess_bulk": "",
                    "ess_tail": "",
                    "t_analysis_phys": "",
                    "grad_evals_span": "",
                    "ess_per_t_analysis": "",
                    "ess_per_1e6_grad": "",
                }
            )

        # Event-aligned window diagnostics (requested ESS analysis windows)
        cfg0 = None
        for p0 in gpaths:
            cfgp = p0 / "run_config.yaml"
            if cfgp.exists():
                import yaml  # local import to avoid mandatory dependency outside execution paths

                with open(cfgp) as f:
                    cfg0 = yaml.safe_load(f) or {}
                break
        h_val = float(cfg0.get("h", 0.0)) if cfg0 else 0.0
        is_underdamped = str(cfg0.get("sampler", "")).lower() == "underdamped" if cfg0 else False

        if args.out_md:
            md_lines.append(f"\n## Group `{gname}`\n")
            md_lines.append(
                "| window_kind | n_chains_used | step_first_mean | step_last_mean | "
                "R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | "
                "T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |\n"
            )
            md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for wname, chain_windows in event_windows.items():
            valid = [
                (tr, st)
                for (tr, st) in chain_windows
                if len(tr) >= max(2, min_al) and len(st) == len(tr)
            ]
            n_used = len(valid)
            if n_used < 2:
                print(f"  Window {wname}: need ≥2 chains with valid aligned window, got {n_used}")
                rows_out.append(
                    {
                        "row_kind": "window_summary",
                        "group": gname,
                        "chain_run": "",
                        "criterion": "",
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
                        "n_chains": "",
                        "n_finite_pred": "",
                        "rhat_aligned": "",
                        "window_kind": wname,
                        "window_len_saves": args.window_len_saves,
                        "n_chains_used": n_used,
                        "step_first_mean": float("nan"),
                        "step_last_mean": float("nan"),
                        "window_rhat": float("nan"),
                        "drift_z_mean": float("nan"),
                        "drift_z_max": float("nan"),
                        "ess_bulk": float("nan"),
                        "ess_tail": float("nan"),
                        "t_analysis_phys": float("nan"),
                        "grad_evals_span": float("nan"),
                        "ess_per_t_analysis": float("nan"),
                        "ess_per_1e6_grad": float("nan"),
                    }
                )
                continue

            n_min = min(len(tr) for tr, _ in valid)
            if n_min < min_al:
                print(f"  Window {wname}: aligned length {n_min} < {min_al}")
                continue
            mat = np.stack([tr[:n_min] for tr, _ in valid], axis=0)
            step_mat = np.stack([st[:n_min] for _, st in valid], axis=0)
            rh = gelman_rubin_rhat(mat)
            dz_list = [drift_z_analysis_window(mat[i, :]) for i in range(mat.shape[0])]
            dz_f = [z for z in dz_list if math.isfinite(z)]
            dz_mean = float(np.mean(dz_f)) if dz_f else float("nan")
            dz_max = float(np.max(dz_f)) if dz_f else float("nan")
            eb, et = multichain_ess_bulk_tail(mat)
            step_first = float(np.mean(step_mat[:, 0]))
            step_last = float(np.mean(step_mat[:, -1]))
            t_analysis = physical_time_span_h(h_val, int(round(step_first)), int(round(step_last)))
            grad_span = grad_evals_for_step_span(int(round(step_last - step_first)), is_underdamped)
            ess_per_t = float(eb / t_analysis) if (math.isfinite(eb) and t_analysis > 0) else float("nan")
            ess_per_1e6 = float((eb / grad_span) * 1e6) if (math.isfinite(eb) and grad_span > 0) else float("nan")
            print(
                f"  Window {wname}: R̂={rh:.4f} drift_z(mean/max)=({dz_mean:.4g}/{dz_max:.4g}) "
                f"ESS_bulk={eb:.4g} ESS_tail={et:.4g} n={n_min} chains={n_used}"
            )
            rows_out.append(
                {
                    "row_kind": "window_summary",
                    "group": gname,
                    "chain_run": "",
                    "criterion": "",
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
                    "n_chains": "",
                    "n_finite_pred": "",
                    "rhat_aligned": "",
                    "window_kind": wname,
                    "window_len_saves": args.window_len_saves,
                    "n_chains_used": n_used,
                    "step_first_mean": step_first,
                    "step_last_mean": step_last,
                    "window_rhat": rh,
                    "drift_z_mean": dz_mean,
                    "drift_z_max": dz_max,
                    "ess_bulk": eb,
                    "ess_tail": et,
                    "t_analysis_phys": t_analysis,
                    "grad_evals_span": grad_span,
                    "ess_per_t_analysis": ess_per_t,
                    "ess_per_1e6_grad": ess_per_1e6,
                }
            )
            if args.out_md:
                md_lines.append(
                    f"| `{wname}` | {n_used} | {step_first:.1f} | {step_last:.1f} | "
                    f"{rh:.4f} | {dz_mean:.4g} | {dz_max:.4g} | {eb:.4g} | {et:.4g} | "
                    f"{t_analysis:.4g} | {grad_span:.4g} | {ess_per_t:.4g} | {ess_per_1e6:.4g} |\n"
                )

            if suffix_fracs:
                if args.out_md:
                    md_lines.append(
                        f"\n#### `{wname}` — tail fractions (last *F* of **aligned** window)\n\n"
                    )
                    md_lines.append(
                        "| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | "
                        "ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |\n"
                    )
                    md_lines.append(
                        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
                    )
                best_frac: float | None = None
                best_met: dict[str, Any] | None = None
                for frac in suffix_fracs:
                    start_col = _aligned_suffix_start_idx(n_min, frac)
                    met = compute_aligned_suffix_metrics(
                        mat, step_mat, start_col, h_val, is_underdamped, min_al
                    )
                    if met is None:
                        print(
                            f"  Window {wname} last-{frac:.0%} of aligned window: "
                            "skip (too few draws in suffix)"
                        )
                        continue
                    rh_s = met["window_rhat"]
                    dzmx = met["drift_z_max"]
                    ok = (
                        math.isfinite(rh_s)
                        and rh_s <= args.window_stab_rhat
                        and math.isfinite(dzmx)
                        and dzmx <= args.window_stab_drift_z
                    )
                    if ok and best_met is None:
                        best_frac = frac
                        best_met = met
                    print(
                        f"  Window {wname} last-{frac:.0%}: R̂={rh_s:.4f} "
                        f"drift_z(mean/max)=({met['drift_z_mean']:.4g}/{dzmx:.4g}) "
                        f"ESS_bulk={met['ess_bulk']:.4g} n={met['suffix_n_draws']} "
                        f"passes_gate={ok}"
                    )
                    rows_out.append(
                        {
                            "row_kind": "window_suffix",
                            "group": gname,
                            "chain_run": "",
                            "criterion": "",
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
                            "n_chains": "",
                            "n_finite_pred": "",
                            "rhat_aligned": "",
                            "window_kind": wname,
                            "window_len_saves": args.window_len_saves,
                            "n_chains_used": n_used,
                            "step_first_mean": met["step_first_mean"],
                            "step_last_mean": met["step_last_mean"],
                            "window_rhat": rh_s,
                            "drift_z_mean": met["drift_z_mean"],
                            "drift_z_max": dzmx,
                            "ess_bulk": met["ess_bulk"],
                            "ess_tail": met["ess_tail"],
                            "t_analysis_phys": met["t_analysis_phys"],
                            "grad_evals_span": met["grad_evals_span"],
                            "ess_per_t_analysis": met["ess_per_t_analysis"],
                            "ess_per_1e6_grad": met["ess_per_1e6_grad"],
                            "window_suffix_frac": frac,
                            "suffix_start_idx": met["suffix_start_idx"],
                            "suffix_n_draws": met["suffix_n_draws"],
                            "stab_ok": int(ok),
                        }
                    )
                    if args.out_md:
                        md_lines.append(
                            f"| {frac:.2f} | {met['suffix_start_idx']} | {met['suffix_n_draws']} | "
                            f"{rh_s:.4f} | {met['drift_z_mean']:.4g} | {dzmx:.4g} | "
                            f"{met['ess_bulk']:.4g} | {met['ess_tail']:.4g} | "
                            f"{met['t_analysis_phys']:.4g} | {met['grad_evals_span']:.4g} | "
                            f"{met['ess_per_t_analysis']:.4g} | "
                            f"{'yes' if ok else 'no'} |\n"
                        )
                if best_met is not None and best_frac is not None:
                    print(
                        f"  Window {wname} stabilized suffix (first passing, largest F first): "
                        f"last-{best_frac:.0%} start_idx={best_met['suffix_start_idx']} "
                        f"R̂={best_met['window_rhat']:.4f} drift_z_max={best_met['drift_z_max']:.4g}"
                    )
                    rows_out.append(
                        {
                            "row_kind": "window_suffix_best",
                            "group": gname,
                            "chain_run": "",
                            "criterion": "",
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
                            "n_chains": "",
                            "n_finite_pred": "",
                            "rhat_aligned": "",
                            "window_kind": wname,
                            "window_len_saves": args.window_len_saves,
                            "n_chains_used": n_used,
                            "step_first_mean": best_met["step_first_mean"],
                            "step_last_mean": best_met["step_last_mean"],
                            "window_rhat": best_met["window_rhat"],
                            "drift_z_mean": best_met["drift_z_mean"],
                            "drift_z_max": best_met["drift_z_max"],
                            "ess_bulk": best_met["ess_bulk"],
                            "ess_tail": best_met["ess_tail"],
                            "t_analysis_phys": best_met["t_analysis_phys"],
                            "grad_evals_span": best_met["grad_evals_span"],
                            "ess_per_t_analysis": best_met["ess_per_t_analysis"],
                            "ess_per_1e6_grad": best_met["ess_per_1e6_grad"],
                            "window_suffix_frac": best_frac,
                            "suffix_start_idx": best_met["suffix_start_idx"],
                            "suffix_n_draws": best_met["suffix_n_draws"],
                            "stab_ok": 1,
                        }
                    )
                    if args.out_md:
                        md_lines.append(
                            f"\n*Stabilized pick:* last **{best_frac:.0%}**, "
                            f"R̂≤{args.window_stab_rhat:g}, max drift_z≤{args.window_stab_drift_z:g}.\n"
                        )
                else:
                    print(
                        f"  Window {wname}: no suffix on grid passed "
                        f"R̂≤{args.window_stab_rhat:g} and max drift_z≤{args.window_stab_drift_z:g}"
                    )
                    rows_out.append(
                        {
                            "row_kind": "window_suffix_best",
                            "group": gname,
                            "chain_run": "",
                            "criterion": "",
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
                            "n_chains": "",
                            "n_finite_pred": "",
                            "rhat_aligned": "",
                            "window_kind": wname,
                            "window_len_saves": args.window_len_saves,
                            "n_chains_used": n_used,
                            "step_first_mean": float("nan"),
                            "step_last_mean": float("nan"),
                            "window_rhat": float("nan"),
                            "drift_z_mean": float("nan"),
                            "drift_z_max": float("nan"),
                            "ess_bulk": float("nan"),
                            "ess_tail": float("nan"),
                            "t_analysis_phys": float("nan"),
                            "grad_evals_span": float("nan"),
                            "ess_per_t_analysis": float("nan"),
                            "ess_per_1e6_grad": float("nan"),
                            "window_suffix_frac": "",
                            "suffix_start_idx": "",
                            "suffix_n_draws": "",
                            "stab_ok": 0,
                        }
                    )
                    if args.out_md:
                        md_lines.append(
                            f"\n*Stabilized pick:* **none** on this grid / gates "
                            f"(R̂≤{args.window_stab_rhat:g}, max drift_z≤{args.window_stab_drift_z:g}).\n"
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
            "window_kind",
            "window_len_saves",
            "n_chains_used",
            "step_first_mean",
            "step_last_mean",
            "window_rhat",
            "drift_z_mean",
            "drift_z_max",
            "ess_bulk",
            "ess_tail",
            "t_analysis_phys",
            "grad_evals_span",
            "ess_per_t_analysis",
            "ess_per_1e6_grad",
            "window_suffix_frac",
            "suffix_start_idx",
            "suffix_n_draws",
            "stab_ok",
        ]
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows_out:
                w.writerow({fn: row.get(fn, "") for fn in fields})
        print("\nWrote", outp)
    if args.out_md:
        mdp = Path(args.out_md)
        mdp.parent.mkdir(parents=True, exist_ok=True)
        mdp.write_text("".join(md_lines))
        print("Wrote", mdp)


if __name__ == "__main__":
    main()
