# Chain convergence and diagnostics report

Runs discovered: **16** in `experiments/runs` (glob `w4_*_ul_initI*_chain*`)

## Method

- **R̂**: Gelman–Rubin on **parallel chains** (same config, different `chain_id`) using aligned `samples_metrics` traces; if only one chain, split-chain R̂ is used.

- **ESS (bulk)**: Autocorrelation ESS per chain; table shows **mean** and **min** across chains.

- **ESS rate**: mean ESS divided by approximate **post-burn-in gradient evaluations** (×2 for underdamped BAOAB).

- **iter_metrics**: pooled early/mid/late means by step tertiles across logged rows.

- **Half split**: for `iter_metrics`, records are sorted by `step` and split at the midpoint **by count** (first half vs second half of pooled rows). **Δ** = mean(2nd half) − mean(1st half). For `samples_metrics`, each chain’s trace is split at its midpoint index; reported means are averaged across chains, then **Δ** = mean(2nd) − mean(1st).

- **Late-window analytics** (below): use only the **last *f*** fraction of saved samples per chain; **R̂** and **multi-chain ESS** (ArviZ bulk/tail) are computed on that window. **drift_z** = |mean(2nd half)−mean(1st half)|/std(2nd half) **within** the late window, per chain (mean/max across chains). **ESS/T_phys** = ESS_bulk / *T*_analysis; **ESS/(1e6 grad)** uses the same step span.


## Group (4 chains)

**sampler=underdamped γ=3.0 h=5e-06 α=0.3 β=1.0 T=100000 B=0 S=20 n_train=512 arch=small_resnet_ln nb=1**

| run_dir | chain_id |
|---------|----------|
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain0` | 0 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain1` | 1 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain2` | 2 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0209 | 12.0 | 7.5 | 5000 | 59.8441 | 940.956 | 913.359 | -27.5963 |
| f_margin | 1.0576 | 103.2 | 8.1 | 5000 | 515.7845 | -0.474694 | -0.423247 | 0.051447 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6573 | 21158.3 | 119765 | 98607.1 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 400000 | 927.196 | 34.2698 | 940.496 | 933.106 | 908.551 | 941.026 | 913.366 | -27.6597 |
| f_margin | 400000 | -0.449072 | 0.0802891 | -0.485559 | -0.448154 | -0.414548 | -0.474889 | -0.423254 | 0.0516352 |
| ce_mean_train | 400000 | 1.81039 | 0.0670394 | 1.83668 | 1.82188 | 1.77372 | 1.83761 | 1.78316 | -0.054449 |
| margin_probe | 400000 | -0.449072 | 0.0802891 | -0.485559 | -0.448154 | -0.414548 | -0.474889 | -0.423254 | 0.0516352 |
| pmax_mean | 400000 | 0.337626 | 0.0434874 | 0.350007 | 0.323496 | 0.339324 | 0.338484 | 0.336768 | -0.0017158 |
| U_train | 400000 | 11651.2 | 8587.84 | 2604.98 | 9955.23 | 22077.5 | 4271.76 | 19030.7 | 14758.9 |
| grad_norm | 400000 | 1351.46 | 1847.13 | 2575.52 | 788.161 | 710.128 | 1974.3 | 728.619 | -1245.68 |
| nll_probe_mean | 400000 | 1.81093 | 0.0669332 | 1.83691 | 1.82247 | 1.77451 | 1.83794 | 1.78392 | -0.0540228 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 400000 | 235.356 | 122.685 | 89.7157 | 239.659 | 372.538 | 129.09 | 341.623 | 212.533 |
| dist_to_ref_sq_over_d | 400000 | 0.0591636 | 0.0481711 | 0.00843119 | 0.0496293 | 0.117658 | 0.0177609 | 0.100566 | 0.0828055 |
| dist_to_ref_over_sqrt_d | 400000 | 0.21569 | 0.112433 | 0.0822192 | 0.219633 | 0.341409 | 0.118303 | 0.313078 | 0.194774 |
| dist_to_ref_over_ou_radius | 400000 | 0.118138 | 0.0615823 | 0.0450333 | 0.120298 | 0.186997 | 0.0647973 | 0.17148 | 0.106682 |
| theta_norm | 400000 | 239.028 | 119.838 | 97.131 | 241.921 | 373.943 | 134.864 | 343.191 | 208.327 |
| v_norm | 400000 | 1085.33 | 3.52001 | 1089.57 | 1085.03 | 1081.5 | 1088.44 | 1082.22 | -6.2251 |
| kinetic_energy | 400000 | 588975 | 3821.93 | 593580 | 588644 | 584826 | 592353 | 585596 | -6757.38 |
| theta_v_cosine | 400000 | 0.72904 | 0.124318 | 0.730845 | 0.77935 | 0.678456 | 0.755863 | 0.702216 | -0.0536464 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 399996 | 0.277286 | 0.162284 | 0.114634 | 0.303842 | 0.40938 | 0.167883 | 0.386689 | 0.218806 |
| noise_step_norm | 400000 | 5.97472 | 0.00386726 | 5.97471 | 5.97474 | 5.97471 | 5.97472 | 5.97472 | 7.20935e-06 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 400000 | 10724.3 | 8601.9 | 1664.6 | 9022.42 | 21169.4 | 3330.9 | 18117.7 | 14786.8 |
| U_data | 400000 | 926.919 | 34.3241 | 940.381 | 932.803 | 908.142 | 940.858 | 912.98 | -27.8779 |
| ce_mean_train | 400000 | 1.81039 | 0.0670394 | 1.83668 | 1.82188 | 1.77372 | 1.83761 | 1.78316 | -0.054449 |
| U_data_minus_ce | 400000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 400000 / 400000

### Stability gates

- **max U_train** (iter_metrics): 29295.3 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 435.11

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; otherwise those cells are **—**.


- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.0155 | 2.1927 | 3.2663 | — | — | 0.2499 | 99960 | nan | nan |
| f_margin | 1.0788 | 0.9504 | 1.6066 | — | — | 0.2499 | 99960 | nan | nan |
| f_dist | 0.9998 | 3.2028 | 3.2034 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9998 | 3.2028 | 3.2034 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6561 | 3.6569 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6561 | 3.6569 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj1 | 3.6173 | 1.7217 | 2.1500 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj2 | 8.7488 | 4.2561 | 7.3777 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc1 | 1.9266 | 1.2699 | 1.6824 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc2 | 2.4600 | 1.3538 | 2.6073 | — | — | 0.2499 | 99960 | nan | nan |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.0847 | 1.9610 | 4.0862 | — | — | 0.1249 | 49960 | nan | nan |
| f_margin | 1.0713 | 1.4006 | 1.9649 | — | — | 0.1249 | 49960 | nan | nan |
| f_dist | 0.9996 | 3.3589 | 3.3602 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9996 | 3.3589 | 3.3602 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9996 | 3.5548 | 3.5562 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9996 | 3.5548 | 3.5562 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj1 | 4.4270 | 2.7684 | 3.2570 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj2 | 18.6284 | 3.9816 | 4.7101 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc1 | 2.9207 | 0.6808 | 1.2857 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc2 | 3.4178 | 1.8526 | 2.5299 | — | — | 0.1249 | 49960 | nan | nan |

## Group (12 chains)

**sampler=underdamped γ=3.0 h=5e-06 α=0.3 β=1.0 T=100000 B=0 S=20 n_train=512 arch=small_resnet_ln nb=1**

| run_dir | chain_id |
|---------|----------|
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain0` | 0 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain0` | 0 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain0` | 0 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain1` | 1 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain1` | 1 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain1` | 1 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain2` | 2 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain2` | 2 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain2` | 2 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain3` | 3 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain3` | 3 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0244 | 15.2 | 12.0 | 5000 | 75.8031 | 818.883 | 860.003 | 41.1205 |
| f_margin | 1.0120 | 16.0 | 13.6 | 5000 | 79.9672 | -0.138931 | -0.328022 | -0.189092 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6578 | 20664.5 | 119208 | 98543.2 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 1200000 | 839.378 | 118.833 | 787.946 | 872.915 | 856.746 | 818.748 | 860.007 | 41.2587 |
| f_margin | 1200000 | -0.233227 | 0.370622 | -0.0389656 | -0.33239 | -0.325529 | -0.138431 | -0.328023 | -0.189592 |
| ce_mean_train | 1200000 | 1.63885 | 0.232024 | 1.53867 | 1.70432 | 1.67253 | 1.59875 | 1.67895 | 0.0801986 |
| margin_probe | 1200000 | -0.233227 | 0.370622 | -0.0389656 | -0.33239 | -0.325529 | -0.138431 | -0.328023 | -0.189592 |
| pmax_mean | 1200000 | 0.34082 | 0.0484259 | 0.338786 | 0.331104 | 0.352225 | 0.33326 | 0.348381 | 0.0151209 |
| U_train | 1200000 | 11485.3 | 8625.27 | 2379.1 | 9816.99 | 21942.8 | 4075.03 | 18895.5 | 14820.5 |
| grad_norm | 1200000 | 700.742 | 856.198 | 1173.17 | 467.066 | 469.016 | 936.399 | 465.085 | -471.314 |
| nll_probe_mean | 1200000 | 1.63941 | 0.232096 | 1.53896 | 1.70491 | 1.67333 | 1.59912 | 1.6797 | 0.0805834 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 1200000 | 233.24 | 124.569 | 85.2208 | 238.532 | 371.769 | 125.693 | 340.787 | 215.094 |
| dist_to_ref_sq_over_d | 1200000 | 0.058722 | 0.0481415 | 0.00802476 | 0.0491884 | 0.117181 | 0.0173461 | 0.100098 | 0.0827518 |
| dist_to_ref_over_sqrt_d | 1200000 | 0.213751 | 0.11416 | 0.0780999 | 0.218601 | 0.340705 | 0.11519 | 0.312312 | 0.197121 |
| dist_to_ref_over_ou_radius | 1200000 | 0.117076 | 0.0625281 | 0.0427771 | 0.119733 | 0.186612 | 0.0630924 | 0.17106 | 0.107968 |
| theta_norm | 1200000 | 237.352 | 120.99 | 93.9273 | 240.815 | 373.197 | 132.327 | 342.377 | 210.05 |
| v_norm | 1200000 | 1085.01 | 3.31314 | 1088.98 | 1084.77 | 1081.4 | 1087.94 | 1082.08 | -5.86238 |
| kinetic_energy | 1200000 | 588632 | 3596.13 | 592938 | 588361 | 584716 | 591813 | 585451 | -6361.67 |
| theta_v_cosine | 1200000 | 0.740465 | 0.120734 | 0.760587 | 0.782925 | 0.679724 | 0.777086 | 0.703844 | -0.0732413 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 1199988 | 0.288056 | 0.116076 | 0.147002 | 0.303969 | 0.409517 | 0.189247 | 0.386866 | 0.197619 |
| noise_step_norm | 1200000 | 5.97472 | 0.00386726 | 5.97471 | 5.97474 | 5.97471 | 5.97472 | 5.97472 | 7.20935e-06 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 1200000 | 10646.2 | 8597.82 | 1591.3 | 8944.38 | 21086.5 | 3256.47 | 18035.9 | 14779.4 |
| U_data | 1200000 | 839.09 | 118.796 | 787.799 | 872.611 | 856.337 | 818.559 | 859.621 | 41.0617 |
| ce_mean_train | 1200000 | 1.63885 | 0.232024 | 1.53867 | 1.70432 | 1.67253 | 1.59875 | 1.67895 | 0.0801986 |
| U_data_minus_ce | 1200000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 1200000 / 1200000

### Stability gates

- **max U_train** (iter_metrics): 29185.6 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 434.483

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; otherwise those cells are **—**.


- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 2.5289 | 3.1757 | 5.3700 | — | — | 0.2499 | 99960 | nan | nan |
| f_margin | 1.8724 | 1.2059 | 2.1563 | — | — | 0.2499 | 99960 | nan | nan |
| f_dist | 0.9998 | 3.2025 | 3.2031 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9998 | 3.2025 | 3.2031 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6579 | 3.6588 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6579 | 3.6588 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj1 | 3.0835 | 1.7307 | 2.1238 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj2 | 7.6811 | 4.1349 | 6.9254 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc1 | 2.1888 | 1.5517 | 3.3348 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc2 | 1.8697 | 1.2919 | 2.9998 | — | — | 0.2499 | 99960 | nan | nan |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 4.9038 | 1.3250 | 2.2988 | — | — | 0.1249 | 49960 | nan | nan |
| f_margin | 2.6073 | 1.0037 | 3.0838 | — | — | 0.1249 | 49960 | nan | nan |
| f_dist | 0.9996 | 3.3588 | 3.3601 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9996 | 3.3588 | 3.3601 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9996 | 3.5554 | 3.5567 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9996 | 3.5554 | 3.5567 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj1 | 3.7480 | 2.7668 | 3.2461 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj2 | 16.4622 | 4.0309 | 4.7938 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc1 | 3.0815 | 1.3438 | 2.7555 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc2 | 1.8219 | 2.4348 | 4.3229 | — | — | 0.1249 | 49960 | nan | nan |
