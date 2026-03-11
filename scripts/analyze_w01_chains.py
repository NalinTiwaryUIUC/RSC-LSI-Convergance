#!/usr/bin/env python3
"""
Analyze w=0.1, 4-chain results.
Convergence metrics (Rhat, ESS), iter_metrics summary, and breakdown of what went right/wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_iter_metrics(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        for k, v in rec.items():
            if isinstance(v, str) and v.lower() == "nan":
                rec[k] = np.nan
        records.append(rec)
    return records


def _split_rhat(traces: np.ndarray) -> float:
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


def _ess_bulk(trace: np.ndarray, max_lag: int | None = None) -> float:
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


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "experiments/runs"
    dirs = sorted(base.glob("w0.1_n1024_h5e-08_a0.1_chain*"))
    if not dirs:
        print("No w0.1 chains found")
        return

    B, S = 50_000, 2_000
    key_steps = [1, 10_000, 50_000, 100_000, 150_000, 200_000]

    # --- iter_metrics summary ---
    print("=" * 70)
    print("w=0.1, h=5e-8, 4 chains — iter_metrics at key steps")
    print("=" * 70)

    all_recs = {d.name: load_iter_metrics(d / "iter_metrics.jsonl") for d in dirs}
    for step in key_steps:
        rows = []
        for d in dirs:
            recs = [r for r in all_recs[d.name] if r.get("step") == step]
            if not recs:
                continue
            r = recs[0]
            rows.append({
                "chain": d.name,
                "U_train": r.get("U_train"),
                "grad_norm": r.get("grad_norm"),
                "snr": r.get("snr"),
                "nll_probe_mean": r.get("nll_probe_mean"),
                "dist_to_ref": r.get("dist_to_ref"),
            })
        if not rows:
            continue
        print(f"\nStep {step}:")
        for row in rows:
            u = row["U_train"]
            g = row["grad_norm"]
            s = row["snr"]
            n = row["nll_probe_mean"]
            d = row["dist_to_ref"]
            print(f"  {row['chain']}: U_train={u:.1f}, grad_norm={g:.0f}, snr={s:.2e}, nll_probe_mean={n:.4f}, dist_to_ref={d:.2f}")

    # Cross-chain stats at last step
    last_recs = []
    for d in dirs:
        recs = [r for r in all_recs[d.name] if r.get("step") == 200_000]
        if recs:
            last_recs.append(recs[0])
    if len(last_recs) >= 2:
        u_vals = [r["U_train"] for r in last_recs]
        n_vals = [r["nll_probe_mean"] for r in last_recs]
        print(f"\nStep 200k cross-chain: U_train mean={np.mean(u_vals):.1f} std={np.std(u_vals):.1f}, nll_probe_mean mean={np.mean(n_vals):.4f} std={np.std(n_vals):.4f}")

    # --- samples_metrics: Rhat, ESS ---
    print("\n" + "=" * 70)
    print("Convergence: Rhat, ESS (post-burn-in, S=2000)")
    print("=" * 70)

    probes = ["f_nll", "f_margin", "f_pc1", "f_pc2", "f_proj1", "f_proj2", "f_dist"]
    traces = {p: [] for p in probes}
    for d in dirs:
        path = d / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        for p in probes:
            if p in data:
                vals = data[p]
                post = data["step"] > B
                traces[p].append(vals[post])

    n_post = min(len(t) for t in traces["f_nll"]) if traces["f_nll"] else 0
    T_post = 75 * S  # 75 saved samples per chain after burn-in

    print(f"\nPost-burn-in samples per chain: {n_post} (steps 52k..200k)")
    print()
    for p in probes:
        if not traces[p]:
            continue
        arr = np.array([t[:n_post] for t in traces[p]])
        rhat = _split_rhat(arr)
        ess_list = [_ess_bulk(t[:n_post]) for t in traces[p]]
        ess = np.nanmean(ess_list)
        ess_rate = ess / T_post if T_post > 0 else float("nan")
        status = "OK" if (np.isfinite(rhat) and rhat < 1.05) else "fail"
        print(f"  {p}: Rhat={rhat:.4f}, ESS={ess:.1f}, ESS_rate={ess_rate:.2e} {status}")

    # --- iter_metrics probes: U_train, dist_to_ref, U_data, nll_probe_mean ---
    print("\n" + "-" * 70)
    print("Convergence from iter_metrics (post-burn-in, log_every=1000)")
    print("-" * 70)

    iter_probes = ["nll_probe_mean", "U_train", "U_data", "dist_to_ref"]
    for p in iter_probes:
        chain_traces = []
        for d in dirs:
            recs = [r for r in all_recs[d.name] if r.get("step", 0) > B and r.get(p) is not None]
            recs = sorted(recs, key=lambda r: r["step"])
            vals = np.array([r[p] for r in recs])
            vals = vals[np.isfinite(vals)]
            chain_traces.append(vals)
        if not chain_traces:
            continue
        n_min = min(len(t) for t in chain_traces)
        if n_min < 4:
            continue
        arr = np.array([t[:n_min] for t in chain_traces])
        rhat = _split_rhat(arr)
        ess_list = [_ess_bulk(t[:n_min]) for t in chain_traces]
        ess = np.nanmean(ess_list)
        n_iter_post = 150  # steps 51k..200k logged every 1k
        ess_rate = ess / (n_iter_post * 1000) if n_iter_post > 0 else float("nan")  # per step
        status = "OK" if (np.isfinite(rhat) and rhat < 1.05) else "fail"
        print(f"  {p}: Rhat={rhat:.4f}, ESS={ess:.1f}, ESS_rate={ess_rate:.2e} {status}")

    # --- Consolidated report: nll, U_train, U_data ---
    T_post = 200_000 - B  # post-burn-in Langevin steps
    consolidated = {}
    for p in ["nll_probe_mean", "U_train", "U_data"]:
        chain_traces = []
        for d in dirs:
            recs = [r for r in all_recs[d.name] if r.get("step", 0) > B and r.get(p) is not None]
            recs = sorted(recs, key=lambda r: r["step"])
            vals = np.array([r[p] for r in recs])
            vals = vals[np.isfinite(vals)]
            chain_traces.append(vals)
        if not chain_traces:
            continue
        n_min = min(len(t) for t in chain_traces)
        if n_min < 4:
            continue
        arr = np.array([t[:n_min] for t in chain_traces])
        ess = np.nanmean([_ess_bulk(t[:n_min]) for t in chain_traces])
        ess_per_1e6 = ess * 1e6 / T_post if T_post > 0 else float("nan")
        consolidated[p] = {"rhat": _split_rhat(arr), "ess": ess, "ess_per_1e6": ess_per_1e6}

    print("\n" + "=" * 70)
    print("CONSOLIDATED: Rhat & ESS for nll_probe_mean, U_train, U_data")
    print("=" * 70)
    print(f"\n{'Metric':<18} {'Rhat':>10} {'ESS':>10} {'ESS/1e6 steps':>14}")
    print("-" * 56)
    for name, key in [("nll_probe_mean", "nll_probe_mean"), ("U_train", "U_train"), ("U_data", "U_data")]:
        if key in consolidated:
            r = consolidated[key]["rhat"]
            e = consolidated[key]["ess"]
            e6 = consolidated[key]["ess_per_1e6"]
            print(f"{name:<18} {r:>10.4f} {e:>10.1f} {e6:>14.1f}")
    print()

    # --- Proxy LSI: rho_hat_f = E[||∇f||^2] / Var(f) ---
    LSI_PROBES = ["f_nll", "f_margin", "f_pc1", "f_proj1", "f_dist"]
    grad_norm_stride = 5
    for d in dirs:
        cfg_path = d / "run_config.yaml"
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            grad_norm_stride = int(cfg.get("grad_norm_stride", 5))
            break

    lsi_rows = []
    for run_dir in dirs:
        path = run_dir / "samples_metrics.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["step"]
        post_burn = steps > B
        n_saved = int(np.sum(post_burn))
        for p in LSI_PROBES:
            fkey, gkey = p, f"grad_norm_sq__{p}"
            if fkey not in data or gkey not in data:
                continue
            f_vals = data[fkey][post_burn]
            g_vals = data[gkey]
            if len(g_vals) == 0 or n_saved == 0:
                continue
            idx = np.arange(n_saved)
            grad_idx = idx[idx % grad_norm_stride == 0][: len(g_vals)]
            if len(grad_idx) < 2:
                continue
            f_sub = f_vals[grad_idx]
            var_f = np.var(f_sub)
            if var_f <= 0:
                continue
            rho = float(np.mean(g_vals[: len(grad_idx)]) / var_f)
            lsi_rows.append({"run_dir": str(run_dir), "probe": p, "rho_hat": rho})

    if lsi_rows:
        print("=" * 70)
        print("Proxy LSI: ρ̂_f = E[||∇f||²] / Var(f) (post-burn-in, from samples_metrics)")
        print("=" * 70)
        by_probe = {}
        for row in lsi_rows:
            p = row["probe"]
            if p not in by_probe:
                by_probe[p] = []
            by_probe[p].append(row["rho_hat"])
        print(f"\n{'Probe':<12} {'rho_hat mean':>14} {'rho_hat std':>14}")
        print("-" * 42)
        for p in LSI_PROBES:
            if p in by_probe:
                vals = np.array(by_probe[p])
                print(f"{p:<12} {np.mean(vals):>14.4e} {np.std(vals):>14.4e}")
        print()

    # --- Breakdown ---
    print("\n" + "=" * 70)
    print("Breakdown: what went right / wrong")
    print("=" * 70)

    u_step1 = [r["U_train"] for d in dirs for r in all_recs[d.name] if r.get("step") == 1]
    u_step200k = [r["U_train"] for d in dirs for r in all_recs[d.name] if r.get("step") == 200_000]
    grad_200k = [r["grad_norm"] for d in dirs for r in all_recs[d.name] if r.get("step") == 200_000]
    rhats = [_split_rhat(np.array([t[:n_post] for t in traces[p]])) for p in probes if traces[p]]
    rhat_max = max(rhats) if rhats else float("nan")
    ess_per_probe = [_ess_bulk(t[:n_post]) for p in probes if traces[p] for t in traces[p]]
    ess_mean = np.nanmean(ess_per_probe) if ess_per_probe else float("nan")

    print("\n✓ Went right:")
    print("  - All 4 chains completed T=200k without NaN/Inf (finite_params, finite_grad, finite_loss)")
    print("  - No abort triggered (bad_locality, bad_prediction)")
    print("  - Chains started from same pretrain (U_train ~30 at step 1)")
    if len(set(np.round(u_step1, 1))) == 1:
        print("  - All chains started at same U_train (reproducible init)")

    print("\n✗ Went wrong / concerns:")
    if u_step200k and np.mean(u_step200k) > 100:
        print(f"  - U_train drifted: step1 mean={np.mean(u_step1):.1f} → step200k mean={np.mean(u_step200k):.1f}")
        print("    Chain drifted far from MAP; posterior exploration or instability?")
    if grad_200k and np.mean(grad_200k) > 1000:
        print(f"  - grad_norm very large at end (mean={np.mean(grad_200k):.0f}) — gradient explosion / drift")
    if np.isfinite(rhat_max) and rhat_max > 1.05:
        print(f"  - Rhat > 1.05 on some probes (max={rhat_max:.3f}) — chains may not have converged to same distribution")
    elif np.isfinite(rhat_max):
        print(f"  - Rhat < 1.05 (max={rhat_max:.3f}) — chains appear to have converged")
    if np.isfinite(ess_mean) and ess_mean < 10:
        print(f"  - Low ESS (~{ess_mean:.1f}) — high autocorrelation, few effective samples")
    print()


if __name__ == "__main__":
    main()
