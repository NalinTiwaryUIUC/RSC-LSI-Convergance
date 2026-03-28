#!/usr/bin/env python3
"""
Aggregate convergence and diagnostics across multiple run directories (e.g. 12 runs =
3 configs × 4 chains).

- Auto-groups runs by run_config.yaml (same experiment, different chain_id).
- From samples_metrics.npz: Gelman–Rubin R̂ (multi-chain), bulk ESS per chain, ESS rate.
- **Late-window analytics**: last 50% / 25% of saves — late-only R̂, drift_z, ArviZ **multi-chain**
  ESS (bulk + tail), ESS per physical time and per grad eval (optional `arviz`).
- From iter_metrics.jsonl: pooled stats + early/mid/late trends for primary/secondary keys.

Usage:
  python scripts/report_chain_convergence.py --runs_dir experiments/runs --glob 'w1_n512*T20000*'
  python scripts/report_chain_convergence.py --runs_dir experiments/runs \\
      --run_dirs experiments/runs/runA experiments/runs/runB ...

Outputs:
  experiments/summaries/chain_convergence_report.md
  experiments/summaries/chain_convergence_summary.csv
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import arviz as az  # type: ignore

    _HAS_ARVIZ = True
except ImportError:
    az = None  # type: ignore
    _HAS_ARVIZ = False

# Keys excluded when grouping (vary per chain or are runtime-only)
_GROUP_EXCLUDE = frozenset({
    "chain_id",
    "run_dir",
    "chain_seed",
    "param_count",
    "ou_radius_pred",
    "effective_batch_size",
    "num_microbatches",
    "microbatch_size",
})

# Probes in samples_metrics.npz for convergence
SAMPLE_PROBES_DEFAULT = ("f_nll", "f_margin", "f_dist")

# iter_metrics keys to summarize (extend as needed)
PRIMARY_ITER_KEYS = (
    "f_nll",
    "f_margin",
    "ce_mean_train",
    "margin_probe",
    "pmax_mean",
    "U_train",
    "grad_norm",
    "nll_probe_mean",
)
SECONDARY_ITER_KEYS = (
    "dist_to_ref",
    "dist_to_ref_sq_over_d",
    "dist_to_ref_over_sqrt_d",
    "dist_to_ref_over_ou_radius",
    "theta_norm",
    "v_norm",
    "kinetic_energy",
    "theta_v_cosine",
    "snr",
    "delta_U",
    "noise_step_norm",
    "drift_step_norm",
    # U decomposition (escape / MAP diagnostics; iter_metrics from run/chain.py)
    "U_prior",
    "U_data",
    "ce_mean_train",
    "U_data_minus_ce",
)

# Extra saved probe directions (same length as f_nll in samples_metrics.npz when present)
EXTRA_SAMPLE_PROBE_KEYS = ("f_proj1", "f_proj2", "f_pc1", "f_pc2")


def _ess_bulk(trace: np.ndarray, max_lag: int | None = None) -> float:
    """Bulk ESS from autocorrelation (single chain)."""
    n = len(trace)
    if n < 2:
        return 0.0
    t = np.asarray(trace, dtype=np.float64)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return float("nan")
    t = t - t.mean()
    if t.var() < 1e-20:
        return float("nan")
    if max_lag is None:
        max_lag = min(len(t) // 2, 1000)
    ac = np.correlate(t, t, mode="full")[len(t) - 1 :]
    ac = ac[: max_lag + 1]
    ac = ac / (ac[0] + 1e-12)
    total = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        total += ac[k]
    tau = 1.0 + 2.0 * total
    return len(t) / tau if tau > 0 else float("nan")


def _split_rhat(traces: np.ndarray) -> float:
    """Split each chain in half → 2×chains pseudo-chains (legacy / single-chain)."""
    n_chains, n = traces.shape
    half = n // 2
    if half < 2:
        return float("nan")
    first = traces[:, :half]
    second = traces[:, half : 2 * half]
    split = np.concatenate([first, second], axis=0)
    m, n_per = split.shape[0], split.shape[1]
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    overall_mean = chain_means.mean()
    B = n_per * ((chain_means - overall_mean) ** 2).sum() / (m - 1)
    W = chain_vars.mean()
    var_plus = (n_per - 1) / n_per * W + B / n_per
    if W <= 0:
        return float("nan")
    return float(np.sqrt(var_plus / W))


def gelman_rubin_rhat(traces: np.ndarray) -> float:
    """
    Standard R̂ for m ≥ 2 chains, shape (m, n_draws), same length n_draws.
    Gelman et al. / BDA between-within variance estimator.
    """
    traces = np.asarray(traces, dtype=np.float64)
    m, n = traces.shape
    if m < 2 or n < 4:
        return float("nan")
    if not np.all(np.isfinite(traces)):
        return float("nan")
    chain_means = traces.mean(axis=1)
    chain_vars = traces.var(axis=1, ddof=1)
    W = float(chain_vars.mean())
    grand = chain_means.mean()
    B = float(n * chain_means.var(ddof=1))
    var_hat = (n - 1) / n * W + B / n
    if W <= 1e-20:
        return float("nan")
    return float(math.sqrt(var_hat / W))


def sample_probe_half_means(chain_traces: list[np.ndarray]) -> tuple[float, float, float]:
    """
    Per chain: split saved samples at midpoint index; mean of first half vs second half.
    Return (mean across chains of first-half means, mean across chains of second-half means, Δ).
    """
    if not chain_traces:
        return float("nan"), float("nan"), float("nan")
    n_min = min(len(t) for t in chain_traces)
    if n_min < 2:
        return float("nan"), float("nan"), float("nan")
    mid = n_min // 2
    first_means: list[float] = []
    second_means: list[float] = []
    for t in chain_traces:
        tt = np.asarray(t[:n_min], dtype=np.float64)
        tt = tt[np.isfinite(tt)]
        if len(tt) < 2:
            continue
        mloc = len(tt) // 2
        first_means.append(float(np.mean(tt[:mloc])))
        second_means.append(float(np.mean(tt[mloc:])))
    if not first_means:
        return float("nan"), float("nan"), float("nan")
    m1 = float(np.mean(first_means))
    m2 = float(np.mean(second_means))
    return m1, m2, m2 - m1


def rhat_for_traces(traces_list: list[np.ndarray]) -> float:
    """Use Gelman–Rubin if ≥2 chains; else split-Rhat on one chain."""
    arrays = [np.asarray(t, dtype=np.float64) for t in traces_list]
    arrays = [a[np.isfinite(a)] for a in arrays]
    n_min = min(len(a) for a in arrays)
    if n_min < 4:
        return float("nan")
    mat = np.stack([a[:n_min] for a in arrays], axis=0)
    if mat.shape[0] >= 2:
        return gelman_rubin_rhat(mat)
    return _split_rhat(mat)


def _slice_late_traces(chain_traces: list[np.ndarray], frac: float) -> list[np.ndarray]:
    """Last `frac` fraction of each chain (aligned by min length before slicing)."""
    if not chain_traces or frac <= 0 or frac > 1:
        return []
    n_min = min(len(t) for t in chain_traces)
    if n_min < 4:
        return []
    start = int(math.floor(n_min * (1.0 - frac)))
    start = max(0, min(start, n_min - 2))
    return [np.asarray(t[start:n_min], dtype=np.float64) for t in chain_traces]


def drift_z_analysis_window(trace: np.ndarray) -> float:
    """|mean(2nd half) − mean(1st half)| / std(2nd half) on a single chain trace."""
    tt = np.asarray(trace, dtype=np.float64)
    tt = tt[np.isfinite(tt)]
    n = len(tt)
    if n < 4:
        return float("nan")
    mid = n // 2
    half1, half2 = tt[:mid], tt[mid:]
    if len(half2) < 2:
        return float("nan")
    sd2 = float(np.std(half2, ddof=1))
    if sd2 < 1e-20:
        return float("nan")
    return abs(float(np.mean(half2)) - float(np.mean(half1))) / sd2


def multichain_ess_bulk_tail(traces: np.ndarray) -> tuple[float, float]:
    """ArviZ multi-chain ESS (bulk + tail). `traces` shape (n_chain, n_draw)."""
    if not _HAS_ARVIZ or az is None:
        return float("nan"), float("nan")
    traces = np.asarray(traces, dtype=np.float64)
    m, n = traces.shape
    if m < 2 or n < 4:
        return float("nan"), float("nan")
    if not np.all(np.isfinite(traces)):
        return float("nan"), float("nan")
    idata = az.from_dict(
        {"posterior": {"x": traces}},
        coords={"chain": np.arange(m), "draw": np.arange(n)},
        dims={"x": ["chain", "draw"]},
    )
    eb = float(az.ess(idata, var_names=["x"], method="bulk")["x"].values.item())
    et = float(az.ess(idata, var_names=["x"], method="tail")["x"].values.item())
    return eb, et


def physical_time_span_h(h: float, step_first: int, step_last: int) -> float:
    """Physical time span between two chain step indices (× h)."""
    return max(0.0, float(step_last - step_first)) * float(h)


def grad_evals_for_step_span(span_steps: int, underdamped: bool) -> float:
    mult = 2.0 if underdamped else 1.0
    return max(0.0, float(span_steps)) * mult


def earliest_stabilization_suffix(
    traces: np.ndarray,
    steps: np.ndarray | None,
    rhat_max: float,
    drift_z_max: float,
) -> tuple[int | None, int | None]:
    """
    Smallest start index k such that suffix traces[:, k:] has Gelman–Rubin R̂ ≤ rhat_max
    and max per-chain drift_z ≤ drift_z_max. Returns (k, step_at_k if steps given).
    """
    traces = np.asarray(traces, dtype=np.float64)
    m, n = traces.shape
    if m < 2 or n < 8:
        return None, None
    for k in range(0, n - 3):
        sub = traces[:, k:]
        if sub.shape[1] < 4:
            break
        if not np.all(np.isfinite(sub)):
            continue
        rh = gelman_rubin_rhat(sub)
        dzs = [drift_z_analysis_window(sub[i, :]) for i in range(m)]
        finite_dz = [x for x in dzs if math.isfinite(x)]
        if not finite_dz:
            continue
        dz_mx = max(finite_dz)
        if math.isfinite(rh) and rh <= rhat_max and dz_mx <= drift_z_max:
            sk = int(steps[k]) if steps is not None and len(steps) > k else None
            return k, sk
    return None, None


def _chain_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"chain(\d+)", path.name)
    cid = int(m.group(1)) if m else 999
    return (cid, path.name)


def group_key(cfg: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    items = []
    for k in sorted(cfg.keys()):
        if k in _GROUP_EXCLUDE:
            continue
        v = cfg[k]
        if v is None or isinstance(v, (dict, list)):
            continue
        items.append((k, v))
    return tuple(items)


def load_iter_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
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


def grad_evals_per_chain(cfg: dict[str, Any]) -> int:
    """Total gradient evaluations after burn-in (approx)."""
    T = int(cfg.get("T", 0))
    B = int(cfg.get("B", 0))
    steps = max(0, T - B)
    mult = 2 if cfg.get("sampler") == "underdamped" else 1
    return steps * mult


def summarize_series(
    records: list[dict[str, Any]], key: str
) -> dict[str, float]:
    """Pooled stats + early/mid/late by step."""
    pts: list[tuple[int, float]] = []
    for r in records:
        if key not in r:
            continue
        v = r[key]
        if not isinstance(v, (int, float)):
            continue
        if not math.isfinite(float(v)):
            continue
        pts.append((int(r.get("step", 0)), float(v)))
    if not pts:
        return {
            "n": 0.0,
            "nan_frac": 1.0,
            "mean": float("nan"),
            "std": float("nan"),
            "early_mean": float("nan"),
            "mid_mean": float("nan"),
            "late_mean": float("nan"),
            "half1_mean": float("nan"),
            "half2_mean": float("nan"),
            "delta_2nd_minus_1st": float("nan"),
        }
    pts.sort(key=lambda x: x[0])
    steps = [p[0] for p in pts]
    vals = np.array([p[1] for p in pts], dtype=np.float64)
    s_min, s_max = min(steps), max(steps)
    span = max(s_max - s_min, 1)
    t1 = s_min + 0.33 * span
    t2 = s_min + 0.66 * span
    early = vals[[i for i, s in enumerate(steps) if s <= t1]]
    mid = vals[[i for i, s in enumerate(steps) if t1 < s <= t2]]
    late = vals[[i for i, s in enumerate(steps) if s > t2]]
    # First / second half by pooled record order (sorted by step): equal count split
    n_pts = len(vals)
    h = n_pts // 2
    first_by_count = vals[:h] if h > 0 else np.array([])
    second_by_count = vals[h:] if h < n_pts else np.array([])
    half1_mean = float(first_by_count.mean()) if len(first_by_count) else float("nan")
    half2_mean = float(second_by_count.mean()) if len(second_by_count) else float("nan")
    delta_2nd_minus_1st = (
        half2_mean - half1_mean
        if math.isfinite(half1_mean) and math.isfinite(half2_mean)
        else float("nan")
    )
    return {
        "n": float(len(vals)),
        "nan_frac": 0.0,
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "early_mean": float(early.mean()) if len(early) else float("nan"),
        "mid_mean": float(mid.mean()) if len(mid) else float("nan"),
        "late_mean": float(late.mean()) if len(late) else float("nan"),
        "half1_mean": half1_mean,
        "half2_mean": half2_mean,
        "delta_2nd_minus_1st": delta_2nd_minus_1st,
    }


def discover_runs(runs_dir: Path, glob_pat: str) -> list[Path]:
    runs_dir = runs_dir.resolve()
    candidates = sorted(runs_dir.glob(glob_pat), key=lambda p: p.name)
    out = []
    for p in candidates:
        if not p.is_dir():
            continue
        if (p / "run_config.yaml").exists() and (p / "samples_metrics.npz").exists():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-chain convergence + iter_metrics report")
    ap.add_argument("--runs_dir", type=str, default="experiments/runs", help="Parent directory")
    ap.add_argument("--glob", type=str, default="*", help="Glob under runs_dir (quoted)")
    ap.add_argument("--run_dirs", nargs="*", default=None, help="Explicit run dirs (skip glob)")
    ap.add_argument(
        "--out_md",
        type=str,
        default="experiments/summaries/chain_convergence_report.md",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="experiments/summaries/chain_convergence_summary.csv",
    )
    ap.add_argument(
        "--probes",
        type=str,
        default="f_nll,f_margin,f_dist",
        help="Comma-separated keys in samples_metrics.npz",
    )
    ap.add_argument(
        "--late-fracs",
        type=str,
        default="0.5,0.25",
        help="Comma-separated fractions (last 50%%, last 25%% of saved samples) for late-window analytics",
    )
    ap.add_argument(
        "--late-probes",
        type=str,
        default=(
            "f_nll,f_margin,f_dist,"
            "dist_to_ref_sq_over_d,dist_to_ref_over_sqrt_d,dist_to_ref_over_ou_radius,"
            "f_proj1,f_proj2,f_pc1,f_pc2"
        ),
        help="Probes for late-window R̂, drift_z, ESS (must exist in samples_metrics.npz)",
    )
    ap.add_argument(
        "--late-out-csv",
        type=str,
        default=None,
        help="Optional second CSV with late-window rows only (default: <out_csv> with _late suffix)",
    )
    ap.add_argument(
        "--stab-rhat",
        type=float,
        default=1.05,
        help="Heuristic: suffix R̂ threshold for stabilization time",
    )
    ap.add_argument(
        "--stab-drift-z",
        type=float,
        default=0.5,
        help="Heuristic: max per-chain drift_z threshold for stabilization time",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.run_dirs:
        run_paths = [Path(p).resolve() for p in args.run_dirs]
    else:
        run_paths = discover_runs(runs_dir, args.glob)

    if not run_paths:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    probes = tuple(p.strip() for p in args.probes.split(",") if p.strip())

    groups: dict[tuple, list[Path]] = defaultdict(list)
    cfg_by_path: dict[Path, dict[str, Any]] = {}
    for p in run_paths:
        cfg_path = p / "run_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        cfg_by_path[p] = cfg
        groups[group_key(cfg)].append(p)

    for k in groups:
        groups[k].sort(key=_chain_sort_key)

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    late_csv_rows: list[dict[str, Any]] = []

    late_fracs = []
    for x in args.late_fracs.split(","):
        x = x.strip()
        if x:
            late_fracs.append(float(x))
    late_probe_candidates = tuple(p.strip() for p in args.late_probes.split(",") if p.strip())

    lines: list[str] = []
    lines.append("# Chain convergence and diagnostics report\n")
    lines.append(f"Runs discovered: **{len(run_paths)}** in `{runs_dir}` (glob `{args.glob}`)\n")
    lines.append("## Method\n")
    lines.append(
        "- **R̂**: Gelman–Rubin on **parallel chains** (same config, different `chain_id`) "
        "using aligned `samples_metrics` traces; if only one chain, split-chain R̂ is used.\n"
    )
    lines.append(
        "- **ESS (bulk)**: Autocorrelation ESS per chain; table shows **mean** and **min** "
        "across chains.\n"
    )
    lines.append(
        "- **ESS rate**: mean ESS divided by approximate **post-burn-in gradient evaluations** "
        "(×2 for underdamped BAOAB).\n"
    )
    lines.append(
        "- **iter_metrics**: pooled early/mid/late means by step tertiles across logged rows.\n"
    )
    lines.append(
        "- **Half split**: for `iter_metrics`, records are sorted by `step` and split at the "
        "midpoint **by count** (first half vs second half of pooled rows). "
        "**Δ** = mean(2nd half) − mean(1st half). "
        "For `samples_metrics`, each chain’s trace is split at its midpoint index; "
        "reported means are averaged across chains, then **Δ** = mean(2nd) − mean(1st).\n"
    )
    lines.append(
        "- **Late-window analytics** (below): use only the **last *f*** fraction of saved "
        "samples per chain; **R̂** and **multi-chain ESS** (ArviZ bulk/tail) are computed on "
        "that window. **drift_z** = |mean(2nd half)−mean(1st half)|/std(2nd half) **within** "
        "the late window, per chain (mean/max across chains). "
        "**ESS/T_phys** = ESS_bulk / *T*_analysis; **ESS/(1e6 grad)** uses the same step span.\n\n"
    )

    sorted_group_items = sorted(groups.items(), key=lambda x: (len(x[1]), str(x[0])))

    for gkey, paths in sorted_group_items:
        if not paths:
            continue
        cfg0 = cfg_by_path[paths[0]]
        label = (
            f"sampler={cfg0.get('sampler', '?')}"
            f" γ={cfg0.get('gamma', '—')}"
            f" h={cfg0.get('h')} α={cfg0.get('alpha')} β={cfg0.get('beta')}"
            f" T={cfg0.get('T')} B={cfg0.get('B')} S={cfg0.get('S')}"
            f" n_train={cfg0.get('n_train')} arch={cfg0.get('arch')} nb={cfg0.get('num_blocks')}"
        )
        lines.append(f"## Group ({len(paths)} chains)\n")
        lines.append(f"**{label}**\n")
        lines.append("| run_dir | chain_id |")
        lines.append("|---------|----------|")
        for p in paths:
            cid = cfg_by_path[p].get("chain_id", "?")
            lines.append(f"| `{p.name}` | {cid} |")
        lines.append("")

        ge = grad_evals_per_chain(cfg0)

        # --- samples_metrics ---
        lines.append("### Convergence from `samples_metrics.npz`\n")
        probe_warnings: list[str] = []
        lines.append(
            "| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | "
            "mean 1st half | mean 2nd half | Δ(2nd−1st) |"
        )
        lines.append(
            "|-------|---|----------|---------|-------------------|-------------------------|"
            "---------------|---------------|------------|"
        )

        for probe in probes:
            chain_traces: list[np.ndarray] = []
            for p in paths:
                npz_path = p / "samples_metrics.npz"
                data = np.load(npz_path)
                if probe not in data:
                    continue
                chain_traces.append(np.asarray(data[probe], dtype=np.float64))
            if len(chain_traces) not in (0, len(paths)):
                probe_warnings.append(
                    f"`{probe}` in {len(chain_traces)}/{len(paths)} chains (R̂ uses available only)"
                )
            if not chain_traces:
                lines.append(f"| {probe} | — | — | — | — | — | — | — | — |")
                csv_rows.append(
                    {
                        "group_label": label[:200],
                        "n_chains": len(paths),
                        "probe": probe,
                        "rhat": float("nan"),
                        "ess_mean": float("nan"),
                        "ess_min": float("nan"),
                        "n_samples": float("nan"),
                        "ess_rate_1e6": float("nan"),
                        "sample_half1_mean": float("nan"),
                        "sample_half2_mean": float("nan"),
                        "sample_delta_2nd_minus_1st": float("nan"),
                    }
                )
                continue
            n_min = min(len(t) for t in chain_traces)
            rh = rhat_for_traces(chain_traces)
            ess_list = []
            for t in chain_traces:
                tt = t[:n_min]
                tt = tt[np.isfinite(tt)]
                ess_list.append(_ess_bulk(tt) if len(tt) >= 2 else float("nan"))
            ess_mean = float(np.nanmean(ess_list))
            ess_min = float(np.nanmin(ess_list))
            ess_rate = (ess_mean / ge * 1e6) if ge > 0 and math.isfinite(ess_mean) else float("nan")
            sh1, sh2, sdlt = sample_probe_half_means(chain_traces)
            lines.append(
                f"| {probe} | {rh:.4f} | {ess_mean:.1f} | {ess_min:.1f} | {n_min} | {ess_rate:.4f} | "
                f"{sh1:.6g} | {sh2:.6g} | {sdlt:.6g} |"
            )
            csv_rows.append(
                {
                    "group_label": label[:200],
                    "n_chains": len(paths),
                    "probe": probe,
                    "rhat": rh,
                    "ess_mean": ess_mean,
                    "ess_min": ess_min,
                    "n_samples": float(n_min),
                    "ess_rate_1e6": ess_rate,
                    "sample_half1_mean": sh1,
                    "sample_half2_mean": sh2,
                    "sample_delta_2nd_minus_1st": sdlt,
                }
            )

        lines.append("")
        if probe_warnings:
            lines.append("*Warnings:* " + "; ".join(probe_warnings) + "\n")
        lines.append(f"*Approx. post-burn grad evals per chain: **{ge}***\n")

        # --- iter_metrics trends ---
        lines.append("### `iter_metrics.jsonl` — primary diagnostics\n")
        lines.append(
            "| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |"
        )
        lines.append("|-----|-----|------|-----|-------|-----|------|----------|----------|------------|")
        all_recs: list[dict[str, Any]] = []
        for p in paths:
            all_recs.extend(load_iter_metrics(p / "iter_metrics.jsonl"))

        for key in PRIMARY_ITER_KEYS:
            s = summarize_series(all_recs, key)
            if s["n"] == 0:
                lines.append(f"| {key} | 0 | — | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {key} | {int(s['n'])} | {s['mean']:.6g} | {s['std']:.6g} | "
                f"{s['early_mean']:.6g} | {s['mid_mean']:.6g} | {s['late_mean']:.6g} | "
                f"{s['half1_mean']:.6g} | {s['half2_mean']:.6g} | {s['delta_2nd_minus_1st']:.6g} |"
            )

        lines.append("")
        lines.append("### `iter_metrics.jsonl` — secondary diagnostics\n")
        lines.append(
            "| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |"
        )
        lines.append("|-----|-----|------|-----|-------|-----|------|----------|----------|------------|")
        for key in SECONDARY_ITER_KEYS:
            s = summarize_series(all_recs, key)
            if s["n"] == 0:
                lines.append(f"| {key} | 0 | — | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {key} | {int(s['n'])} | {s['mean']:.6g} | {s['std']:.6g} | "
                f"{s['early_mean']:.6g} | {s['mid_mean']:.6g} | {s['late_mean']:.6g} | "
                f"{s['half1_mean']:.6g} | {s['half2_mean']:.6g} | {s['delta_2nd_minus_1st']:.6g} |"
            )

        # Behaviour hints
        lines.append("")
        lines.append("### Quick interpretation\n")
        abort_any = any(r.get("abort_suggested") for r in all_recs)
        bad_loc = sum(1 for r in all_recs if r.get("bad_locality"))
        lines.append(f"- **abort_suggested** ever: {abort_any}")
        lines.append(f"- **bad_locality** flags (count): {bad_loc}")
        finite_u = sum(1 for r in all_recs if isinstance(r.get("U_train"), (int, float)) and math.isfinite(float(r["U_train"])))
        lines.append(f"- **U_train** finite records: {finite_u} / {len(all_recs)}")
        lines.append("")

        # --- Stability gates (explicit) ---
        lines.append("### Stability gates\n")
        u_vals = [
            float(r["U_train"])
            for r in all_recs
            if isinstance(r.get("U_train"), (int, float)) and math.isfinite(float(r["U_train"]))
        ]
        u_max = max(u_vals) if u_vals else float("nan")
        theta_vals = [
            float(r["theta_norm"])
            for r in all_recs
            if isinstance(r.get("theta_norm"), (int, float)) and math.isfinite(float(r["theta_norm"]))
        ]
        theta_max = max(theta_vals) if theta_vals else float("nan")
        lines.append(
            f"- **max U_train** (iter_metrics): {u_max:.6g} — flag if blow-up vs typical scale.\n"
        )
        lines.append(
            f"- **max ||θ||** (iter_metrics): {theta_max:.6g}\n"
        )
        npz_ref = np.load(paths[0] / "samples_metrics.npz")
        steps_arr = np.asarray(npz_ref["step"], dtype=np.int64) if "step" in npz_ref.files else None
        nan_late50 = False
        if "f_nll" in npz_ref.files and steps_arr is not None:
            n_sa = len(steps_arr)
            st = int(math.floor(n_sa * 0.5))
            fn = np.asarray(npz_ref["f_nll"], dtype=np.float64)[st:]
            nan_late50 = bool(np.any(~np.isfinite(fn)))
        lines.append(
            f"- **NaNs in last 50% of saved f_nll** (chain0 ref): **{nan_late50}**\n"
        )
        lines.append("")

        # --- Late-window analytics ---
        lines.append("### Late-window analytics (stationarity + mixing)\n")
        if not _HAS_ARVIZ:
            lines.append(
                "*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; "
                "otherwise those cells are **—**.\n\n"
            )
        h_val = float(cfg0.get("h", 0.0))
        ud = cfg0.get("sampler") == "underdamped"

        def _stack_probe(paths_: list[Path], probe_name: str) -> np.ndarray | None:
            ars = []
            for pth in paths_:
                data = np.load(pth / "samples_metrics.npz")
                if probe_name not in data:
                    return None
                ars.append(np.asarray(data[probe_name], dtype=np.float64))
            n_m = min(len(a) for a in ars)
            if n_m < 4:
                return None
            return np.stack([a[:n_m] for a in ars], axis=0)

        stab_lines: list[str] = []
        for pname in ("f_nll", "f_margin"):
            matf = _stack_probe(paths, pname)
            if matf is None or steps_arr is None:
                stab_lines.append(f"- **{pname}**: stabilization scan **n/a** (missing data).\n")
                continue
            k0, sk = earliest_stabilization_suffix(
                matf, steps_arr, args.stab_rhat, args.stab_drift_z
            )
            if k0 is None:
                stab_lines.append(
                    f"- **{pname}**: no suffix found with R̂≤{args.stab_rhat} and "
                    f"max drift_z≤{args.stab_drift_z} (heuristic).\n"
                )
            else:
                stab_lines.append(
                    f"- **{pname}**: earliest save index **{k0}** "
                    f"(chain step **{sk}**), physical time **{float(sk or 0) * h_val:.6g}**.\n"
                )
        lines.extend(stab_lines)
        lines.append("")

        for frac in late_fracs:
            pct = int(round(100 * frac))
            lines.append(f"#### Last **{pct}%** of saved samples per chain\n")
            lines.append(
                "| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | "
                "T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |"
            )
            lines.append(
                "|-------|----------|-------------|-------------|----------|----------|"
                "---------------------|-------------------|----------|----------------|"
            )

            n_min_steps = None
            if steps_arr is not None:
                n_min_steps = len(steps_arr)
            start_idx = 0
            if n_min_steps is not None and n_min_steps >= 4:
                start_idx = max(0, min(int(math.floor(n_min_steps * (1.0 - frac))), n_min_steps - 2))
            step_first = int(steps_arr[start_idx]) if steps_arr is not None and len(steps_arr) > start_idx else 0
            step_last = int(steps_arr[-1]) if steps_arr is not None and len(steps_arr) else 0
            span_steps = max(0, step_last - step_first)
            t_phys = physical_time_span_h(h_val, step_first, step_last)
            grad_span = grad_evals_for_step_span(span_steps, ud)

            for probe_late in late_probe_candidates:
                chain_traces_l: list[np.ndarray] = []
                missing = False
                for pth in paths:
                    data = np.load(pth / "samples_metrics.npz")
                    if probe_late not in data:
                        missing = True
                        break
                    chain_traces_l.append(np.asarray(data[probe_late], dtype=np.float64))
                if missing or not chain_traces_l:
                    continue
                late_tr = _slice_late_traces(chain_traces_l, frac)
                if len(late_tr) < 2:
                    lines.append(f"| {probe_late} | — | — | — | — | — | — | — | — | — |")
                    continue
                mat = np.stack(late_tr, axis=0)
                if mat.shape[1] < 4:
                    lines.append(f"| {probe_late} | — | — | — | — | — | — | — | — | — |")
                    continue
                rh_l = gelman_rubin_rhat(mat) if mat.shape[0] >= 2 else float("nan")
                dz_list = [drift_z_analysis_window(mat[i]) for i in range(mat.shape[0])]
                dz_finite = [x for x in dz_list if math.isfinite(x)]
                dz_mean = float(np.mean(dz_finite)) if dz_finite else float("nan")
                dz_max = float(np.max(dz_finite)) if dz_finite else float("nan")
                eb, et = multichain_ess_bulk_tail(mat)
                ess_tp = (eb / t_phys) if t_phys > 1e-30 and math.isfinite(eb) else float("nan")
                ess_g1 = (eb / grad_span * 1e6) if grad_span > 1e-30 and math.isfinite(eb) else float("nan")
                eb_s = f"{eb:.4f}" if math.isfinite(eb) else "—"
                et_s = f"{et:.4f}" if math.isfinite(et) else "—"
                lines.append(
                    f"| {probe_late} | {rh_l:.4f} | {dz_mean:.4f} | {dz_max:.4f} | {eb_s} | {et_s} | "
                    f"{t_phys:.6g} | {grad_span:.0f} | {ess_tp:.6g} | {ess_g1:.6g} |"
                )
                late_csv_rows.append(
                    {
                        "group_label": label[:200],
                        "late_frac": frac,
                        "probe": probe_late,
                        "rhat_late": rh_l,
                        "drift_z_mean": dz_mean,
                        "drift_z_max": dz_max,
                        "ess_bulk": eb,
                        "ess_tail": et,
                        "T_analysis_phys": t_phys,
                        "grad_evals_span": grad_span,
                        "ess_per_phys": ess_tp,
                        "ess_per_1e6_grad": ess_g1,
                        "step_first_late": step_first,
                        "step_last": step_last,
                    }
                )
            lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out_md)

    # CSV
    if csv_rows:
        import csv

        fields = list(csv_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print("Wrote", out_csv)

    late_out = (
        Path(args.late_out_csv)
        if args.late_out_csv
        else out_csv.with_name(out_csv.stem + "_late" + out_csv.suffix)
    )
    if late_csv_rows:
        import csv

        fields = list(late_csv_rows[0].keys())
        late_out.parent.mkdir(parents=True, exist_ok=True)
        with open(late_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(late_csv_rows)
        print("Wrote", late_out)


if __name__ == "__main__":
    main()
