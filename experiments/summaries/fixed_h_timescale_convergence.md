## Executive summary

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


# Detailed tables: fixed h = 5×10⁻⁶ width sweep


Runs: `w{1,2,4}_n512_h5e-06_T{T_steps}_a0.3_b1p0_g3p0_ul_chain{0..3}`, underdamped.


Physical time: T_phys = h·T_steps → 0.2 with 40000 steps, 0.5 with 100000 steps.


**Note:** At `T_phys=0.5` only **w=1** and **w=4** appear in this workspace (no w=2 long runs).


## T_phys=0.2 (T=40000)


### Width w = 1 (4 chains)


#### Saved samples: Gelman–Rubin R̂ and ESS

| probe | window | R̂ | ESS mean | ESS min | ArviZ ESS_bulk | ArviZ ESS_tail | drift_z max | ESS/(1e6 grad)* |
|-------|--------|-----|----------|---------|----------------|----------------|-------------|----------------|
| f_nll | full | 1.0004 | 2.89 | 2.87 | 6.06 | 28.30 | 7.1282 | 36.1640 |
| f_nll | last 50% | 1.0206 | 2.94 | 2.93 | 5.84 | 21.35 | 4.6000 | 73.5789 |
| f_nll | last 25% | 1.0723 | 3.00 | 2.85 | 5.60 | 13.38 | 4.7763 | 150.4719 |
| f_margin | full | 1.0002 | 3.39 | 3.36 | 6.05 | 27.70 | 11.8085 | 42.3637 |
| f_margin | last 50% | 1.0343 | 3.01 | 2.95 | 5.78 | 19.01 | 6.6552 | 75.3972 |
| f_margin | last 25% | 1.1454 | 3.02 | 2.88 | 5.42 | 9.72 | 8.3839 | 151.4671 |
| f_dist | full | 0.9998 | 2.93 | 2.93 | 6.09 | 29.87 | 2.4449 | 36.6835 |
| f_dist | last 50% | 0.9995 | 2.88 | 2.88 | 6.11 | 29.97 | 3.0721 | 72.0081 |
| f_dist | last 25% | 0.9991 | 2.87 | 2.87 | 6.15 | 30.13 | 3.2842 | 143.9708 |
| dist_to_ref_sq_over_d | full | 0.9998 | 2.93 | 2.93 | 6.09 | 29.87 | 2.4449 | 36.6835 |
| dist_to_ref_sq_over_d | last 50% | 0.9995 | 2.88 | 2.88 | 6.11 | 29.97 | 3.0721 | 72.0081 |
| dist_to_ref_sq_over_d | last 25% | 0.9991 | 2.87 | 2.87 | 6.15 | 30.13 | 3.2842 | 143.9708 |
| dist_to_ref_over_sqrt_d | full | 0.9998 | 2.88 | 2.88 | 6.09 | 29.87 | 3.6543 | 35.9640 |
| dist_to_ref_over_sqrt_d | last 50% | 0.9995 | 2.87 | 2.87 | 6.11 | 29.97 | 3.5497 | 71.9135 |
| dist_to_ref_over_sqrt_d | last 25% | 0.9991 | 2.87 | 2.87 | 6.15 | 30.13 | 3.4987 | 143.9451 |
| dist_to_ref_over_ou_radius | full | 0.9998 | 2.88 | 2.88 | 6.09 | 29.87 | 3.6543 | 35.9640 |
| dist_to_ref_over_ou_radius | last 50% | 0.9995 | 2.87 | 2.87 | 6.11 | 29.97 | 3.5497 | 71.9135 |
| dist_to_ref_over_ou_radius | last 25% | 0.9991 | 2.87 | 2.87 | 6.15 | 30.13 | 3.4987 | 143.9451 |

*ESS/(1e6 grad): approximate, using step span in the window ×2 (underdamped).


#### `iter_metrics.jsonl` — primary diagnostics (pooled across chains)

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| f_nll | 40004 | 332.276 | 166.858 | 121.878 | 367.966 | 501.907 | 285.418 |
| f_margin | 40004 | 1.45141 | 1.16219 | 2.88953 | 1.02926 | 0.464907 | -1.76231 |
| ce_mean_train | 40004 | 0.648932 | 0.325894 | 0.238001 | 0.718635 | 0.980245 | 0.557459 |
| margin_probe | 40004 | 1.45141 | 1.16219 | 2.88953 | 1.02926 | 0.464907 | -1.76231 |
| pmax_mean | 40004 | 0.617419 | 0.0850908 | 0.723233 | 0.601361 | 0.530272 | -0.147659 |
| U_train | 40004 | 553.114 | 274.647 | 226.361 | 561.993 | 861.736 | 478.604 |
| grad_norm | 40004 | 1263.54 | 361.794 | 1409.66 | 1344.18 | 1043.42 | -288.212 |
| nll_probe_mean | 40004 | 0.648976 | 0.325895 | 0.238043 | 0.718683 | 0.980287 | 0.557456 |

#### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| dist_to_ref | 40004 | 25.808 | 14.3897 | 8.92271 | 25.9467 | 42.0669 | 24.9058 |
| dist_to_ref_sq_over_d | 40004 | 0.0114147 | 0.00981894 | 0.00137983 | 0.00909564 | 0.0234082 | 0.0166657 |
| dist_to_ref_over_sqrt_d | 40004 | 0.093315 | 0.0520294 | 0.0322622 | 0.0938167 | 0.152103 | 0.0900528 |
| dist_to_ref_over_ou_radius | 40004 | 0.0511107 | 0.0284977 | 0.0176708 | 0.0513855 | 0.0833103 | 0.049324 |
| theta_norm | 40004 | 37.1026 | 9.78833 | 26.3363 | 35.8009 | 48.8187 | 17.0796 |
| v_norm | 40004 | 274.988 | 1.01811 | 276.178 | 274.818 | 273.997 | -1.64554 |
| kinetic_energy | 40004 | 37809.7 | 280.066 | 38137.4 | 37762.6 | 37537.3 | -452.599 |
| theta_v_cosine | 40004 | 0.572115 | 0.205475 | 0.318242 | 0.65677 | 0.736429 | 0.303313 |
| delta_U | 40000 | 0.0906027 | 0.0395618 | 0.0872344 | 0.099168 | 0.0855585 | -0.00668547 |
| noise_step_norm | 40004 | 1.51431 | 0.00384352 | 1.51426 | 1.51439 | 1.51429 | 1.98235e-05 |

#### `iter_metrics` multi-chain R̂ (aligned record count)

| key | R̂ | n aligned |
|-----|-----|-----------|
| nll_probe_mean | 1.0006 | 10001 |
| U_train | 1.0002 | 10001 |
| grad_norm | 1.1102 | 10001 |
| f_nll | 1.0006 | 10001 |
| f_margin | 1.0004 | 10001 |
| dist_to_ref | 1.0000 | 10001 |
| dist_to_ref_over_sqrt_d | 1.0000 | 10001 |

### Width w = 2 (4 chains)


#### Saved samples: Gelman–Rubin R̂ and ESS

| probe | window | R̂ | ESS mean | ESS min | ArviZ ESS_bulk | ArviZ ESS_tail | drift_z max | ESS/(1e6 grad)* |
|-------|--------|-----|----------|---------|----------------|----------------|-------------|----------------|
| f_nll | full | 1.0007 | 3.71 | 3.65 | 5.81 | 15.90 | 19.5276 | 46.3758 |
| f_nll | last 50% | 1.1110 | 3.28 | 3.04 | 5.31 | 8.63 | 7.8440 | 82.1381 |
| f_nll | last 25% | 1.7561 | 3.14 | 2.89 | 4.64 | 4.67 | 4.1045 | 157.3119 |
| f_margin | full | 1.0001 | 5.57 | 5.48 | 5.89 | 15.18 | 34.1014 | 69.6102 |
| f_margin | last 50% | 1.0884 | 3.24 | 2.89 | 5.29 | 6.71 | 6.7889 | 81.1353 |
| f_margin | last 25% | 1.6654 | 3.60 | 3.20 | 4.88 | 5.06 | 4.7059 | 180.1941 |
| f_dist | full | 0.9998 | 2.93 | 2.93 | 6.09 | 29.87 | 2.4358 | 36.7032 |
| f_dist | last 50% | 0.9995 | 2.88 | 2.88 | 6.11 | 29.98 | 3.0684 | 72.0051 |
| f_dist | last 25% | 0.9990 | 2.87 | 2.87 | 6.16 | 30.18 | 3.2847 | 143.9654 |
| dist_to_ref_sq_over_d | full | 0.9998 | 2.93 | 2.93 | 6.09 | 29.87 | 2.4358 | 36.7032 |
| dist_to_ref_sq_over_d | last 50% | 0.9995 | 2.88 | 2.88 | 6.11 | 29.98 | 3.0684 | 72.0051 |
| dist_to_ref_sq_over_d | last 25% | 0.9990 | 2.87 | 2.87 | 6.16 | 30.18 | 3.2847 | 143.9654 |
| dist_to_ref_over_sqrt_d | full | 0.9998 | 2.88 | 2.87 | 6.09 | 29.87 | 3.6425 | 35.9588 |
| dist_to_ref_over_sqrt_d | last 50% | 0.9995 | 2.87 | 2.87 | 6.11 | 29.98 | 3.5483 | 71.9075 |
| dist_to_ref_over_sqrt_d | last 25% | 0.9990 | 2.87 | 2.87 | 6.16 | 30.18 | 3.5004 | 143.9413 |
| dist_to_ref_over_ou_radius | full | 0.9998 | 2.88 | 2.87 | 6.09 | 29.87 | 3.6425 | 35.9588 |
| dist_to_ref_over_ou_radius | last 50% | 0.9995 | 2.87 | 2.87 | 6.11 | 29.98 | 3.5483 | 71.9075 |
| dist_to_ref_over_ou_radius | last 25% | 0.9990 | 2.87 | 2.87 | 6.16 | 30.18 | 3.5004 | 143.9413 |

*ESS/(1e6 grad): approximate, using step span in the window ×2 (underdamped).


#### `iter_metrics.jsonl` — primary diagnostics (pooled across chains)

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| f_nll | 40004 | 567.673 | 212.831 | 309.788 | 659.983 | 728.454 | 298.482 |
| f_margin | 40004 | 0.581266 | 1.17061 | 1.74018 | 0.0975597 | -0.0744214 | -1.25675 |
| ce_mean_train | 40004 | 1.10863 | 0.415684 | 0.604944 | 1.28893 | 1.42265 | 0.582966 |
| margin_probe | 40004 | 0.581266 | 1.17061 | 1.74018 | 0.0975597 | -0.0744214 | -1.25675 |
| pmax_mean | 40004 | 0.457092 | 0.0993728 | 0.572013 | 0.411092 | 0.390164 | -0.127339 |
| U_train | 40004 | 1173.99 | 621.679 | 460.162 | 1160.27 | 1880.35 | 1056.17 |
| grad_norm | 40004 | 1562.82 | 1038.95 | 2805.54 | 1164.36 | 743.008 | -1500.26 |
| nll_probe_mean | 40004 | 1.10874 | 0.415685 | 0.605056 | 1.28903 | 1.42276 | 0.582973 |

#### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| dist_to_ref | 40004 | 51.2332 | 28.612 | 17.6762 | 51.4711 | 83.5821 | 49.5269 |
| dist_to_ref_sq_over_d | 40004 | 0.0114619 | 0.00987718 | 0.00137901 | 0.00911458 | 0.0235296 | 0.016757 |
| dist_to_ref_over_sqrt_d | 40004 | 0.0934722 | 0.0522011 | 0.0322492 | 0.0939062 | 0.152491 | 0.0903592 |
| dist_to_ref_over_ou_radius | 40004 | 0.0511968 | 0.0285917 | 0.0176636 | 0.0514346 | 0.0835229 | 0.0494918 |
| theta_norm | 40004 | 58.7774 | 24.2433 | 31.1588 | 57.1241 | 87.1965 | 42.4533 |
| v_norm | 40004 | 546.019 | 1.0744 | 547.214 | 545.934 | 544.941 | -1.56911 |
| kinetic_energy | 40004 | 149069 | 586.867 | 149722 | 149022 | 148480 | -856.955 |
| theta_v_cosine | 40004 | 0.714688 | 0.193708 | 0.508438 | 0.815321 | 0.817259 | 0.211822 |
| delta_U | 40000 | 0.221491 | 0.0687805 | 0.228576 | 0.195033 | 0.240294 | 0.0133706 |
| noise_step_norm | 40004 | 3.0012 | 0.00387201 | 3.00123 | 3.00127 | 3.0011 | -0.000131842 |

#### `iter_metrics` multi-chain R̂ (aligned record count)

| key | R̂ | n aligned |
|-----|-----|-----------|
| nll_probe_mean | 1.0009 | 10001 |
| U_train | 1.0001 | 10001 |
| grad_norm | 1.0004 | 10001 |
| f_nll | 1.0009 | 10001 |
| f_margin | 1.0003 | 10001 |
| dist_to_ref | 1.0000 | 10001 |
| dist_to_ref_over_sqrt_d | 1.0000 | 10001 |

### Width w = 4 (4 chains)


#### Saved samples: Gelman–Rubin R̂ and ESS

| probe | window | R̂ | ESS mean | ESS min | ArviZ ESS_bulk | ArviZ ESS_tail | drift_z max | ESS/(1e6 grad)* |
|-------|--------|-----|----------|---------|----------------|----------------|-------------|----------------|
| f_nll | full | 1.0003 | 3.03 | 3.01 | 6.03 | 27.25 | 6.7208 | 37.8564 |
| f_nll | last 50% | 1.0126 | 2.97 | 2.91 | 5.95 | 19.35 | 6.5596 | 74.3374 |
| f_nll | last 25% | 1.1039 | 2.83 | 2.76 | 5.32 | 10.90 | 5.3840 | 141.5411 |
| f_margin | full | 1.0000 | 4.26 | 4.18 | 6.00 | 23.04 | 14.5053 | 53.2632 |
| f_margin | last 50% | 1.0315 | 3.03 | 3.00 | 5.84 | 17.11 | 5.9863 | 75.9228 |
| f_margin | last 25% | 1.2811 | 2.90 | 2.74 | 5.06 | 4.76 | 4.5624 | 145.3423 |
| f_dist | full | 0.9998 | 2.93 | 2.93 | 6.09 | 29.87 | 2.4333 | 36.7038 |
| f_dist | last 50% | 0.9995 | 2.88 | 2.88 | 6.11 | 29.98 | 3.0667 | 72.0066 |
| f_dist | last 25% | 0.9990 | 2.87 | 2.87 | 6.16 | 30.20 | 3.2841 | 143.9638 |
| dist_to_ref_sq_over_d | full | 0.9998 | 2.93 | 2.93 | 6.09 | 29.87 | 2.4333 | 36.7038 |
| dist_to_ref_sq_over_d | last 50% | 0.9995 | 2.88 | 2.88 | 6.11 | 29.98 | 3.0667 | 72.0066 |
| dist_to_ref_sq_over_d | last 25% | 0.9990 | 2.87 | 2.87 | 6.16 | 30.20 | 3.2841 | 143.9638 |
| dist_to_ref_over_sqrt_d | full | 0.9998 | 2.87 | 2.87 | 6.09 | 29.87 | 3.6390 | 35.9552 |
| dist_to_ref_over_sqrt_d | last 50% | 0.9995 | 2.87 | 2.87 | 6.11 | 29.98 | 3.5463 | 71.9088 |
| dist_to_ref_over_sqrt_d | last 25% | 0.9990 | 2.87 | 2.87 | 6.16 | 30.20 | 3.5000 | 143.9393 |
| dist_to_ref_over_ou_radius | full | 0.9998 | 2.87 | 2.87 | 6.09 | 29.87 | 3.6390 | 35.9552 |
| dist_to_ref_over_ou_radius | last 50% | 0.9995 | 2.87 | 2.87 | 6.11 | 29.98 | 3.5463 | 71.9088 |
| dist_to_ref_over_ou_radius | last 25% | 0.9990 | 2.87 | 2.87 | 6.16 | 30.20 | 3.5000 | 143.9393 |

*ESS/(1e6 grad): approximate, using step span in the window ×2 (underdamped).


#### `iter_metrics.jsonl` — primary diagnostics (pooled across chains)

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| f_nll | 40004 | 380.744 | 181.809 | 152.417 | 426.493 | 558.019 | 300.539 |
| f_margin | 40004 | 1.3086 | 1.42954 | 2.92342 | 0.727351 | 0.304952 | -1.84828 |
| ce_mean_train | 40004 | 0.743325 | 0.354963 | 0.297533 | 0.832672 | 1.08942 | 0.586761 |
| margin_probe | 40004 | 1.3086 | 1.42954 | 2.92342 | 0.727351 | 0.304952 | -1.84828 |
| pmax_mean | 40004 | 0.589288 | 0.10545 | 0.720849 | 0.557523 | 0.49239 | -0.167123 |
| U_train | 40004 | 2525.27 | 1934.7 | 488.28 | 2149.41 | 4867.77 | 3306.88 |
| grad_norm | 40004 | 1748.15 | 749.426 | 2492.23 | 1594.64 | 1174.74 | -1006.56 |
| nll_probe_mean | 40004 | 0.743641 | 0.355097 | 0.297689 | 0.832995 | 1.08988 | 0.586991 |

#### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| dist_to_ref | 40004 | 102.173 | 57.0877 | 35.2194 | 102.642 | 166.721 | 98.8241 |
| dist_to_ref_sq_over_d | 40004 | 0.0115046 | 0.00991773 | 0.00138161 | 0.00914596 | 0.0236222 | 0.0168255 |
| dist_to_ref_over_sqrt_d | 40004 | 0.0936354 | 0.0523176 | 0.0322766 | 0.0940651 | 0.15279 | 0.0905666 |
| dist_to_ref_over_ou_radius | 40004 | 0.0512862 | 0.0286555 | 0.0176786 | 0.0515216 | 0.0836868 | 0.0496053 |
| theta_norm | 40004 | 106.923 | 53.5309 | 44.7611 | 105.606 | 168.552 | 93.4114 |
| v_norm | 40004 | 1088.72 | 1.64266 | 1090.48 | 1088.92 | 1086.83 | -2.6549 |
| kinetic_energy | 40004 | 592660 | 1788.22 | 594570 | 592868 | 590604 | -2890.26 |
| theta_v_cosine | 40004 | 0.8038 | 0.159716 | 0.686881 | 0.88063 | 0.842745 | 0.0983746 |
| delta_U | 40000 | 0.646356 | 0.28578 | 0.317607 | 0.662846 | 0.949432 | 0.472219 |
| noise_step_norm | 40004 | 5.97474 | 0.00390692 | 5.97476 | 5.97474 | 5.97473 | -2.65159e-05 |

#### `iter_metrics` multi-chain R̂ (aligned record count)

| key | R̂ | n aligned |
|-----|-----|-----------|
| nll_probe_mean | 1.0005 | 10001 |
| U_train | 1.0000 | 10001 |
| grad_norm | 1.0055 | 10001 |
| f_nll | 1.0005 | 10001 |
| f_margin | 1.0002 | 10001 |
| dist_to_ref | 1.0000 | 10001 |
| dist_to_ref_over_sqrt_d | 1.0000 | 10001 |

## T_phys=0.5 (T=100000)


### Width w = 1 (4 chains)


#### Saved samples: Gelman–Rubin R̂ and ESS

| probe | window | R̂ | ESS mean | ESS min | ArviZ ESS_bulk | ArviZ ESS_tail | drift_z max | ESS/(1e6 grad)* |
|-------|--------|-----|----------|---------|----------------|----------------|-------------|----------------|
| f_nll | full | 1.0010 | 4.32 | 4.22 | 5.60 | 7.71 | 15.7984 | 21.6094 |
| f_nll | last 50% | 1.2652 | 3.27 | 3.00 | 5.16 | 4.61 | 9.2116 | 32.7010 |
| f_nll | last 25% | 2.8296 | 3.80 | 2.94 | 4.63 | 7.57 | 3.1221 | 76.1561 |
| f_margin | full | 1.0003 | 5.54 | 5.40 | 5.56 | 10.13 | 29.1170 | 27.7009 |
| f_margin | last 50% | 1.2973 | 3.52 | 3.03 | 5.12 | 5.67 | 4.5979 | 35.2326 |
| f_margin | last 25% | 2.0902 | 6.04 | 2.85 | 4.76 | 7.28 | 4.5547 | 120.8477 |
| f_dist | full | 0.9999 | 3.53 | 3.53 | 6.08 | 29.80 | 2.6325 | 17.6611 |
| f_dist | last 50% | 0.9998 | 2.87 | 2.87 | 6.09 | 29.83 | 3.2139 | 28.7202 |
| f_dist | last 25% | 0.9998 | 2.87 | 2.87 | 6.09 | 29.85 | 3.3639 | 57.4871 |
| dist_to_ref_sq_over_d | full | 0.9999 | 3.53 | 3.53 | 6.08 | 29.80 | 2.6325 | 17.6611 |
| dist_to_ref_sq_over_d | last 50% | 0.9998 | 2.87 | 2.87 | 6.09 | 29.83 | 3.2139 | 28.7202 |
| dist_to_ref_sq_over_d | last 25% | 0.9998 | 2.87 | 2.87 | 6.09 | 29.85 | 3.3639 | 57.4871 |
| dist_to_ref_over_sqrt_d | full | 0.9999 | 3.56 | 3.56 | 6.08 | 29.80 | 3.9014 | 17.8103 |
| dist_to_ref_over_sqrt_d | last 50% | 0.9998 | 2.88 | 2.88 | 6.09 | 29.83 | 3.6706 | 28.7714 |
| dist_to_ref_over_sqrt_d | last 25% | 0.9998 | 2.87 | 2.87 | 6.09 | 29.85 | 3.5598 | 57.5126 |
| dist_to_ref_over_ou_radius | full | 0.9999 | 3.56 | 3.56 | 6.08 | 29.80 | 3.9014 | 17.8103 |
| dist_to_ref_over_ou_radius | last 50% | 0.9998 | 2.88 | 2.88 | 6.09 | 29.83 | 3.6706 | 28.7714 |
| dist_to_ref_over_ou_radius | last 25% | 0.9998 | 2.87 | 2.87 | 6.09 | 29.85 | 3.5598 | 57.5126 |

*ESS/(1e6 grad): approximate, using step span in the window ×2 (underdamped).


#### `iter_metrics.jsonl` — primary diagnostics (pooled across chains)

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| f_nll | 100004 | 492.612 | 169.387 | 291.518 | 565.121 | 617.437 | 230.382 |
| f_margin | 100004 | 0.678039 | 0.970816 | 1.67671 | 0.259933 | 0.114434 | -1.08012 |
| ce_mean_train | 100004 | 0.962085 | 0.330831 | 0.569327 | 1.10371 | 1.20588 | 0.449957 |
| margin_probe | 100004 | 0.678039 | 0.970816 | 1.67671 | 0.259933 | 0.114434 | -1.08012 |
| pmax_mean | 100004 | 0.541684 | 0.0830509 | 0.63816 | 0.505318 | 0.48333 | -0.106945 |
| U_train | 100004 | 1253.07 | 686.687 | 472.531 | 1217.18 | 2045.57 | 1175.8 |
| grad_norm | 100004 | 923.48 | 396.059 | 1327.35 | 834.567 | 617.738 | -520.508 |
| nll_probe_mean | 100004 | 0.962132 | 0.330834 | 0.569372 | 1.10375 | 1.20593 | 0.449964 |

#### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| dist_to_ref | 100004 | 58.8161 | 31.3719 | 21.545 | 60.1481 | 93.7027 | 54.165 |
| dist_to_ref_sq_over_d | 100004 | 0.0580929 | 0.0475842 | 0.00797711 | 0.0486815 | 0.115875 | 0.0817969 |
| dist_to_ref_over_sqrt_d | 100004 | 0.212664 | 0.113433 | 0.0779012 | 0.21748 | 0.338805 | 0.195847 |
| dist_to_ref_over_ou_radius | 100004 | 0.116481 | 0.0621297 | 0.0426683 | 0.119119 | 0.185571 | 0.10727 |
| theta_norm | 100004 | 65.7576 | 27.3092 | 33.9103 | 65.2401 | 97.1744 | 47.84 |
| v_norm | 100004 | 274.084 | 1.12797 | 275.211 | 274.076 | 272.997 | -1.50247 |
| kinetic_energy | 100004 | 37561.6 | 309.543 | 37871 | 37558.9 | 37263.9 | -412.071 |
| theta_v_cosine | 100004 | 0.641651 | 0.144778 | 0.536126 | 0.72847 | 0.659818 | 0.0726648 |
| delta_U | 100000 | 0.0965534 | 0.0323581 | 0.0917435 | 0.0899506 | 0.107631 | 0.0140051 |
| noise_step_norm | 100004 | 1.51431 | 0.00385886 | 1.51433 | 1.51433 | 1.51428 | -2.83729e-05 |

#### `iter_metrics` multi-chain R̂ (aligned record count)

| key | R̂ | n aligned |
|-----|-----|-----------|
| nll_probe_mean | 1.0011 | 25001 |
| U_train | 1.0001 | 25001 |
| grad_norm | 1.1005 | 25001 |
| f_nll | 1.0011 | 25001 |
| f_margin | 1.0004 | 25001 |
| dist_to_ref | 1.0000 | 25001 |
| dist_to_ref_over_sqrt_d | 1.0000 | 25001 |

### Width w = 4 (4 chains)


#### Saved samples: Gelman–Rubin R̂ and ESS

| probe | window | R̂ | ESS mean | ESS min | ArviZ ESS_bulk | ArviZ ESS_tail | drift_z max | ESS/(1e6 grad)* |
|-------|--------|-----|----------|---------|----------------|----------------|-------------|----------------|
| f_nll | full | 1.0004 | 4.58 | 4.54 | 5.82 | 16.36 | 18.6662 | 22.9181 |
| f_nll | last 50% | 1.0941 | 3.00 | 2.78 | 5.49 | 8.02 | 5.9916 | 30.0277 |
| f_nll | last 25% | 1.3451 | 4.56 | 3.23 | 5.73 | 7.83 | 4.4249 | 91.2972 |
| f_margin | full | 1.0003 | 7.11 | 7.02 | 5.75 | 9.65 | 30.9218 | 35.5715 |
| f_margin | last 50% | 1.2266 | 3.79 | 2.85 | 5.62 | 4.57 | 5.6679 | 37.8951 |
| f_margin | last 25% | 2.1556 | 6.54 | 5.69 | 6.10 | 7.50 | 1.6281 | 130.8550 |
| f_dist | full | 0.9999 | 3.53 | 3.53 | 6.08 | 29.81 | 2.6288 | 17.6611 |
| f_dist | last 50% | 0.9998 | 2.87 | 2.87 | 6.09 | 29.85 | 3.2032 | 28.7196 |
| f_dist | last 25% | 0.9996 | 2.87 | 2.87 | 6.10 | 29.92 | 3.3601 | 57.4888 |
| dist_to_ref_sq_over_d | full | 0.9999 | 3.53 | 3.53 | 6.08 | 29.81 | 2.6288 | 17.6611 |
| dist_to_ref_sq_over_d | last 50% | 0.9998 | 2.87 | 2.87 | 6.09 | 29.85 | 3.2032 | 28.7196 |
| dist_to_ref_sq_over_d | last 25% | 0.9996 | 2.87 | 2.87 | 6.10 | 29.92 | 3.3601 | 57.4888 |
| dist_to_ref_over_sqrt_d | full | 0.9999 | 3.56 | 3.56 | 6.08 | 29.81 | 3.8972 | 17.8069 |
| dist_to_ref_over_sqrt_d | last 50% | 0.9998 | 2.88 | 2.88 | 6.09 | 29.85 | 3.6587 | 28.7697 |
| dist_to_ref_over_sqrt_d | last 25% | 0.9996 | 2.87 | 2.87 | 6.10 | 29.92 | 3.5567 | 57.5142 |
| dist_to_ref_over_ou_radius | full | 0.9999 | 3.56 | 3.56 | 6.08 | 29.81 | 3.8972 | 17.8069 |
| dist_to_ref_over_ou_radius | last 50% | 0.9998 | 2.88 | 2.88 | 6.09 | 29.85 | 3.6587 | 28.7697 |
| dist_to_ref_over_ou_radius | last 25% | 0.9996 | 2.87 | 2.87 | 6.10 | 29.92 | 3.5567 | 57.5142 |

*ESS/(1e6 grad): approximate, using step span in the window ×2 (underdamped).


#### `iter_metrics.jsonl` — primary diagnostics (pooled across chains)

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| f_nll | 100004 | 538.752 | 173.603 | 338.915 | 614.782 | 658.941 | 224.492 |
| f_margin | 100004 | 0.554434 | 1.09523 | 1.53289 | 0.1405 | 0.00640494 | -1.04724 |
| ce_mean_train | 100004 | 1.05169 | 0.338856 | 0.661665 | 1.20015 | 1.28619 | 0.43807 |
| margin_probe | 100004 | 0.554434 | 1.09523 | 1.53289 | 0.1405 | 0.00640494 | -1.04724 |
| pmax_mean | 100004 | 0.508607 | 0.0940212 | 0.612077 | 0.462555 | 0.452865 | -0.111758 |
| U_train | 100004 | 11121.5 | 8728.15 | 1862.44 | 9495.48 | 21687.4 | 15011.2 |
| grad_norm | 100004 | 1231.17 | 653.571 | 1878.96 | 975.967 | 850.043 | -737.866 |
| nll_probe_mean | 100004 | 1.05225 | 0.339068 | 0.661942 | 1.20075 | 1.28699 | 0.438461 |

#### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | Δ(2nd half − 1st half) |
|-----|---|------|-----|-------|-----|------|-------------------------|
| dist_to_ref | 100004 | 233.254 | 124.566 | 85.242 | 238.565 | 371.774 | 215.082 |
| dist_to_ref_sq_over_d | 100004 | 0.0587265 | 0.0481398 | 0.00802928 | 0.0492011 | 0.117184 | 0.0827499 |
| dist_to_ref_over_sqrt_d | 100004 | 0.213763 | 0.114157 | 0.0781193 | 0.218631 | 0.34071 | 0.197111 |
| dist_to_ref_over_ou_radius | 100004 | 0.117083 | 0.0625264 | 0.0427877 | 0.119749 | 0.186614 | 0.107962 |
| theta_norm | 100004 | 235.786 | 122.305 | 90.6447 | 239.906 | 372.675 | 211.983 |
| v_norm | 100004 | 1085.07 | 3.43704 | 1089.22 | 1084.74 | 1081.36 | -6.05587 |
| kinetic_energy | 100004 | 588691 | 3731.07 | 593197 | 588327 | 584670 | -6572.2 |
| theta_v_cosine | 100004 | 0.754439 | 0.116408 | 0.797956 | 0.786553 | 0.681027 | -0.0978137 |
| delta_U | 100000 | 1.15183 | 0.477991 | 0.568323 | 1.22967 | 1.64262 | 0.80247 |
| noise_step_norm | 100004 | 5.97472 | 0.00389466 | 5.97473 | 5.97478 | 5.97465 | -8.74517e-05 |

#### `iter_metrics` multi-chain R̂ (aligned record count)

| key | R̂ | n aligned |
|-----|-----|-----------|
| nll_probe_mean | 1.0005 | 25001 |
| U_train | 1.0000 | 25001 |
| grad_norm | 1.0053 | 25001 |
| f_nll | 1.0005 | 25001 |
| f_margin | 1.0004 | 25001 |
| dist_to_ref | 1.0000 | 25001 |
| dist_to_ref_over_sqrt_d | 1.0000 | 25001 |

## Interpretation (auto-generated bullets)

- **R̂ on saved probes** (`f_nll`, `f_margin`): often near 1.0 on the full window; check **last 50% / 25%** for growth — indicates chains still disagreeing late in physical time on predictive probes.

- **Normalized distance probes** (`dist_to_ref_over_sqrt_d`, `dist_to_ref_over_ou_radius`): typically show R̂ ≈ 1 across windows when chains track similar drift; compare to `f_nll`.

- **ESS** on saved samples is limited by stride `S=20` and strong autocorrelation; use **ESS/(1e6 grad)** for cross-timescale comparison (longer runs accumulate more grad evals).

- **`iter_metrics` trends**: increasing `U_train`, `dist_to_ref*`, `theta_norm` over early/mid/late is consistent with outward drift under sampling; compare slopes across widths at fixed `T_phys`.

