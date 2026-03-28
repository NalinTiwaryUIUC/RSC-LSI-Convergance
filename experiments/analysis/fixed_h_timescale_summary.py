#!/usr/bin/env python3
"""
Summarize convergence (R̂, ESS) and iter_metrics trends for fixed-h width sweeps
at two physical timescales: T_phys=0.2 (T_steps=40000) and T_phys=0.5 (T_steps=100000),
h=5e-6, n_train=512, underdamped.

Writes:
  experiments/summaries/fixed_h_timescale_convergence.md
  experiments/summaries/fixed_h_timescale_convergence.csv
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.report_chain_convergence import (  # noqa: E402
    _ess_bulk,
    drift_z_analysis_window,
    gelman_rubin_rhat,
    grad_evals_for_step_span,
    load_iter_metrics,
    multichain_ess_bulk_tail,
    summarize_series,
)

RUNS_DIR = ROOT / "experiments" / "runs"
OUT_MD = ROOT / "experiments" / "summaries" / "fixed_h_timescale_convergence.md"
OUT_CSV = ROOT / "experiments" / "summaries" / "fixed_h_timescale_convergence.csv"

# Preface (updated when re-running; tables below are source of truth)
EXEC_SUMMARY = """## Executive summary

This report aggregates **four chains** per (width, timescale) from `samples_metrics.npz` (stride **S=20**, burn-in **B=0**) and pooled **`iter_metrics.jsonl`** (log every 4 steps). Underdamped sampling implies **2 gradient evaluations per step**.

### Cross-width picture at T_phys = 0.2 (40k steps)

- **Predictive probes (`f_nll`, `f_margin` on saved samples):** Full-window Gelman–Rubin **R̂ ≈ 1.00** for all widths. **Late windows tell a different story:** for **`f_nll` last 25%**, R̂ is moderate at **w=1 (~1.07)** and **w=4 (~1.10)** but **much larger at w=2 (~1.76)** — chains disagree more on probe CE in the final quarter at the intermediate width. **`f_margin`** shows the same pattern (last-25% R̂ up to **~1.67** at w=2).
- **Geometry / locality (`f_dist`, normalized `dist_to_ref_*`):** R̂ stays **≈1.00** across full, last 50%, and last 25% at **all widths**. Per-chain bulk ESS is ~2.9–3.0 on these series; ArviZ multi-chain ESS_bulk is **~6** with high ESS_tail, consistent with smooth, chain-aligned drift.
- **Iter-metrics cross-chain R̂ (aligned time series):** Dense logs agree across chains for **`nll_probe_mean`**, **`U_train`**, **`f_nll`**, margins, and **distance norms** (R̂ ≈ 1.00). **`grad_norm`** is more volatile: **w=1** shows elevated R̂ (**~1.11**) vs ~1.00 at w=2,4 — worth watching but can reflect scale changes rather than multi-modality.
- **Trends (pooled early / mid / late by step):** **`U_train`**, raw **`dist_to_ref`**, and **`theta_norm`** increase with time; **width amplifies scale** (late `U_train` roughly **860 → 1880 → 4870** for w=1,2,4). **Normalized** `dist_to_ref_over_sqrt_d` and `dist_to_ref_over_ou_radius` track **nearly the same early→late path** across w=1–4 (~0.032→~0.153 and ~0.018→~0.084), supporting width-normalized locality as a stable summary. **`grad_norm`** does **not** monotonically grow**: it can **fall** in the late segment (especially w=2,4) while `U_train` still rises — consistent with landscape flattening or moving to regions with smaller training gradients.

### Cross-width picture at T_phys = 0.5 (100k steps; w=1 and w=4 only)

- **Sample-based R̂ for `f_nll` / `f_margin` blows up in late windows at long horizon**, especially **`f_nll` last 25%** (**R̂ ~2.8** at w=1, **~1.35** at w=4) and **`f_margin` last 25%** (**~2.1** vs **~2.2**). This indicates **substantial between-chain disagreement on predictive probes** late in the run, not a failure of the distance probes.
- **Distance-based saved probes** remain **well-matched across chains** (R̂ **~1.00** full and late). ESS per saved draw is similar to the 0.2 run; **ESS/(1e6 grad)** is lower on full windows than at 0.2 because the same ESS is spread over **more** gradient work.
- **Iter trends:** At w=4, **`U_train` late mean (~2.2e4)** is an order of magnitude above w=1 (~2e3). **Raw `dist_to_ref`** is much larger at w=4 than w=1; **normalized** `dist_to_ref_over_sqrt_d` late means are **similar** (~0.34 for both in pooled early/mid/late tables), so width-normalized locality stays comparable while raw distances differ. **`grad_norm`** again trends **down** in the second half for both widths while `U_train` can still increase.

### Practical takeaways

1. **Use two probe classes:** (i) predictive (`f_nll`, `f_margin`, `nll_probe_mean`) for “how wrong is the classifier on probes”, and (ii) **normalized distance** for “how far in parameter space”, because mixing behavior differs sharply.
2. **Late-window R̂ on predictive probes** is a **stark non-stationarity / cross-chain divergence signal** at the longer physical time — interpret alongside **drift_z** and **ESS_tail** in the tables.
3. **Do not read raw ESS alone:** compare **ESS/(1e6 grad)** or ESS per unit physical time when comparing 0.2 vs 0.5 runs.

4. **w=2 at T_phys=0.5** was not in this batch; only **w=1** and **w=4** long runs are compared at the 0.5 timescale.

---
"""

# Sample keys to report (must exist in npz for that run)
SAMPLE_KEYS = [
    "f_nll",
    "f_margin",
    "f_dist",
    "dist_to_ref_sq_over_d",
    "dist_to_ref_over_sqrt_d",
    "dist_to_ref_over_ou_radius",
]

PRIMARY_ITER = (
    "f_nll",
    "f_margin",
    "ce_mean_train",
    "margin_probe",
    "pmax_mean",
    "U_train",
    "grad_norm",
    "nll_probe_mean",
)

SECONDARY_ITER = (
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
)


def _stack_chains(paths: list[Path], key: str) -> tuple[np.ndarray | None, int]:
    ars = []
    for p in paths:
        npz = p / "samples_metrics.npz"
        if not npz.exists():
            return None, 0
        data = np.load(npz)
        if key not in data.files:
            return None, 0
        ars.append(np.asarray(data[key], dtype=np.float64))
    n = min(len(a) for a in ars)
    if n < 4:
        return None, n
    return np.stack([a[:n] for a in ars], axis=0), n


def _late_slice(mat: np.ndarray, frac: float) -> tuple[np.ndarray, int]:
    """Return (sliced mat, start index)."""
    _, n = mat.shape
    start = max(0, int(math.floor(n * (1.0 - frac))))
    start = min(start, n - 2)
    return mat[:, start:], start


def _steps_for_window(steps_arr: np.ndarray | None, start: int, n: int) -> np.ndarray | None:
    if steps_arr is None or len(steps_arr) < start + n:
        return None
    return steps_arr[start : start + n]


def analyze_sample_window(
    mat: np.ndarray,
    steps_sub: np.ndarray | None,
) -> dict[str, float]:
    """Rhat, per-chain ESS stats, ArviZ bulk/tail, drift_z max, ESS per 1e6 grad (approx)."""
    m, n = mat.shape
    if m < 2 or n < 4:
        return {"rhat": float("nan"), "ess_mean": float("nan"), "ess_min": float("nan"),
                "ess_bulk_az": float("nan"), "ess_tail_az": float("nan"),
                "drift_z_max": float("nan"), "ess_per_1e6_grad": float("nan")}
    rh = gelman_rubin_rhat(mat)
    ess_list = [_ess_bulk(mat[i]) for i in range(m)]
    ess_mean = float(np.nanmean(ess_list))
    ess_min = float(np.nanmin(ess_list))
    eb, et = multichain_ess_bulk_tail(mat)
    dzs = [drift_z_analysis_window(mat[i]) for i in range(m)]
    dz_max = float(np.nanmax(dzs)) if dzs else float("nan")
    if steps_sub is not None and len(steps_sub) >= n:
        s0, s1 = int(steps_sub[0]), int(steps_sub[n - 1])
        span = s1 - s0
    else:
        span = n
    ge = grad_evals_for_step_span(span, underdamped=True)
    ess_pg = (ess_mean / ge * 1e6) if ge > 0 and math.isfinite(ess_mean) else float("nan")
    return {
        "rhat": rh,
        "ess_mean": ess_mean,
        "ess_min": ess_min,
        "ess_bulk_az": eb,
        "ess_tail_az": et,
        "drift_z_max": dz_max,
        "ess_per_1e6_grad": ess_pg,
    }


def discover_groups() -> dict[str, dict[float, list[Path]]]:
    """Return {label: {width: [paths]}} for T40000 and T100000 fixed grids."""
    out: dict[str, dict[float, list[Path]]] = {
        "T_phys=0.2 (T=40000)": {},
        "T_phys=0.5 (T=100000)": {},
    }
    pat = re.compile(r"^w([0-9.]+)_n512_h5e-06_T(40000|100000)_a0\.3_b1p0_g3p0_ul_chain\d+$")
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        w = float(m.group(1))
        tstep = m.group(2)
        label = "T_phys=0.2 (T=40000)" if tstep == "40000" else "T_phys=0.5 (T=100000)"
        out[label].setdefault(w, []).append(d)
    for label in out:
        for w in out[label]:
            out[label][w] = sorted(out[label][w], key=lambda p: p.name)
    return out


def main() -> None:
    groups = discover_groups()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    lines: list[str] = [EXEC_SUMMARY]

    lines.append("\n# Detailed tables: fixed h = 5×10⁻⁶ width sweep\n")
    lines.append("\nRuns: `w{1,2,4}_n512_h5e-06_T{T_steps}_a0.3_b1p0_g3p0_ul_chain{0..3}`, underdamped.\n")
    lines.append(
        "\nPhysical time: T_phys = h·T_steps → 0.2 with 40000 steps, 0.5 with 100000 steps.\n"
    )
    lines.append("\n**Note:** At `T_phys=0.5` only **w=1** and **w=4** appear in this workspace (no w=2 long runs).\n")

    for label, by_w in groups.items():
        lines.append(f"\n## {label}\n")

        for w in sorted(by_w.keys()):
            paths = by_w[w]
            lines.append(f"\n### Width w = {w:g} ({len(paths)} chains)\n")

            # --- samples_metrics ---
            ref_npz = np.load(paths[0] / "samples_metrics.npz")
            steps_arr = np.asarray(ref_npz["step"], dtype=np.int64) if "step" in ref_npz.files else None

            lines.append("\n#### Saved samples: Gelman–Rubin R̂ and ESS\n")
            lines.append(
                "| probe | window | R̂ | ESS mean | ESS min | ArviZ ESS_bulk | ArviZ ESS_tail | drift_z max | ESS/(1e6 grad)* |"
            )
            lines.append("|-------|--------|-----|----------|---------|----------------|----------------|-------------|----------------|")

            for key in SAMPLE_KEYS:
                mat_full, n_full = _stack_chains(paths, key)
                if mat_full is None:
                    lines.append(f"| {key} | — | (missing npz key) | | | | | | | |")
                    continue
                for win_name, mat, st_idx in [
                    ("full", mat_full, 0),
                    ("last 50%", *_late_slice(mat_full, 0.5)),
                    ("last 25%", *_late_slice(mat_full, 0.25)),
                ]:
                    if mat.shape[1] < 4:
                        continue
                    steps_sub = _steps_for_window(steps_arr, st_idx, mat.shape[1])
                    st = analyze_sample_window(mat, steps_sub)
                    n_w = mat.shape[1]
                    csv_rows.append({
                        "timescale": label,
                        "width": w,
                        "source": "samples",
                        "probe": key,
                        "window": win_name,
                        "n_draws": n_w,
                        "rhat": st["rhat"],
                        "ess_mean": st["ess_mean"],
                        "ess_min": st["ess_min"],
                        "ess_bulk_az": st["ess_bulk_az"],
                        "ess_tail_az": st["ess_tail_az"],
                        "drift_z_max": st["drift_z_max"],
                        "ess_per_1e6_grad": st["ess_per_1e6_grad"],
                    })
                    lines.append(
                        f"| {key} | {win_name} | {st['rhat']:.4f} | {st['ess_mean']:.2f} | {st['ess_min']:.2f} | "
                        f"{st['ess_bulk_az']:.2f} | {st['ess_tail_az']:.2f} | {st['drift_z_max']:.4f} | {st['ess_per_1e6_grad']:.4f} |"
                    )
            lines.append("\n*ESS/(1e6 grad): approximate, using step span in the window ×2 (underdamped).\n")

            # --- iter_metrics pooled trends ---
            all_recs: list[dict[str, Any]] = []
            for p in paths:
                all_recs.extend(load_iter_metrics(p / "iter_metrics.jsonl"))

            lines.append("\n#### `iter_metrics.jsonl` — primary diagnostics (pooled across chains)\n")
            lines.append("| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |")
            lines.append("|-----|---|------|-----|-------|-----|------|-------------------------|")
            for k in PRIMARY_ITER:
                s = summarize_series(all_recs, k)
                if s["n"] == 0:
                    continue
                lines.append(
                    f"| {k} | {int(s['n'])} | {s['mean']:.6g} | {s['std']:.6g} | "
                    f"{s['early_mean']:.6g} | {s['mid_mean']:.6g} | {s['late_mean']:.6g} | {s['delta_2nd_minus_1st']:.6g} |"
                )

            lines.append("\n#### `iter_metrics.jsonl` — secondary diagnostics\n")
            lines.append("| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |")
            lines.append("|-----|---|------|-----|-------|-----|------|-------------------------|")
            for k in SECONDARY_ITER:
                s = summarize_series(all_recs, k)
                if s["n"] == 0:
                    continue
                lines.append(
                    f"| {k} | {int(s['n'])} | {s['mean']:.6g} | {s['std']:.6g} | "
                    f"{s['early_mean']:.6g} | {s['mid_mean']:.6g} | {s['late_mean']:.6g} | {s['delta_2nd_minus_1st']:.6g} |"
                )

            # Iter-based R̂ for dense series (align by min length across chains)
            lines.append("\n#### `iter_metrics` multi-chain R̂ (aligned record count)\n")
            lines.append("| key | R̂ | n aligned |")
            lines.append("|-----|-----|-----------|")
            for k in ("nll_probe_mean", "U_train", "grad_norm", "f_nll", "f_margin", "dist_to_ref", "dist_to_ref_over_sqrt_d"):
                series_list = []
                for p in paths:
                    recs = load_iter_metrics(p / "iter_metrics.jsonl")
                    vals = []
                    for r in sorted(recs, key=lambda x: x.get("step", 0)):
                        v = r.get(k)
                        if isinstance(v, (int, float)) and math.isfinite(float(v)):
                            vals.append(float(v))
                    if len(vals) >= 4:
                        series_list.append(np.array(vals, dtype=np.float64))
                if len(series_list) < 2:
                    continue
                n_al = min(len(s) for s in series_list)
                if n_al < 4:
                    continue
                mat = np.stack([s[:n_al] for s in series_list], axis=0)
                rh = gelman_rubin_rhat(mat)
                lines.append(f"| {k} | {rh:.4f} | {n_al} |")
                csv_rows.append({
                    "timescale": label,
                    "width": w,
                    "source": "iter_rhat",
                    "probe": k,
                    "window": "full_aligned",
                    "n_draws": n_al,
                    "rhat": rh,
                    "ess_mean": "",
                    "ess_min": "",
                    "ess_bulk_az": "",
                    "ess_tail_az": "",
                    "drift_z_max": "",
                    "ess_per_1e6_grad": "",
                })

    # Cross-cutting summary
    lines.append("\n## Interpretation (auto-generated bullets)\n")
    lines.append(
        "- **R̂ on saved probes** (`f_nll`, `f_margin`): often near 1.0 on the full window; "
        "check **last 50% / 25%** for growth — indicates chains still disagreeing late in physical time on predictive probes.\n"
    )
    lines.append(
        "- **Normalized distance probes** (`dist_to_ref_over_sqrt_d`, `dist_to_ref_over_ou_radius`): "
        "typically show R̂ ≈ 1 across windows when chains track similar drift; compare to `f_nll`.\n"
    )
    lines.append(
        "- **ESS** on saved samples is limited by stride `S=20` and strong autocorrelation; "
        "use **ESS/(1e6 grad)** for cross-timescale comparison (longer runs accumulate more grad evals).\n"
    )
    lines.append(
        "- **`iter_metrics` trends**: increasing `U_train`, `dist_to_ref*`, `theta_norm` over early/mid/late "
        "is consistent with outward drift under sampling; compare slopes across widths at fixed `T_phys`.\n"
    )

    OUT_MD.write_text("".join(l + "\n" for l in lines), encoding="utf-8")

    fieldnames = [
        "timescale", "width", "source", "probe", "window", "n_draws",
        "rhat", "ess_mean", "ess_min", "ess_bulk_az", "ess_tail_az",
        "drift_z_max", "ess_per_1e6_grad",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print("Wrote", OUT_MD)
    print("Wrote", OUT_CSV)


if __name__ == "__main__":
    main()
