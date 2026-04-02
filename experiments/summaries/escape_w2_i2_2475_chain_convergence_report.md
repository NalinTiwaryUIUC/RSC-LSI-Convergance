# Chain convergence and diagnostics report

Runs discovered: **4** in `experiments/runs` (glob `w2_*_ul_initI2_step2475_chain*`)

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
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain0` | 0 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain1` | 1 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain2` | 2 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0010 | 8.8 | 8.6 | 5000 | 44.0652 | 709.923 | 806.566 | 96.6437 |
| f_margin | 1.0001 | 9.8 | 9.5 | 5000 | 49.1284 | 0.070974 | -0.259742 | -0.330716 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6614 | 5196.54 | 29984.8 | 24788.3 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 800000 | 758.177 | 139.519 | 655.949 | 812 | 805.159 | 709.787 | 806.567 | 96.7802 |
| f_margin | 800000 | -0.0941197 | 0.450723 | 0.231596 | -0.24567 | -0.263162 | 0.071501 | -0.25974 | -0.331241 |
| ce_mean_train | 800000 | 1.48066 | 0.272487 | 1.28105 | 1.58579 | 1.57238 | 1.38619 | 1.57514 | 0.188948 |
| margin_probe | 800000 | -0.0941197 | 0.450723 | 0.231596 | -0.24567 | -0.263162 | 0.071501 | -0.25974 | -0.331241 |
| pmax_mean | 800000 | 0.372615 | 0.0535436 | 0.393654 | 0.35473 | 0.369556 | 0.378954 | 0.366277 | -0.012677 |
| U_train | 800000 | 3491.95 | 2228 | 1108.64 | 3116.73 | 6169.34 | 1581.95 | 5401.95 | 3820 |
| grad_norm | 800000 | 613.617 | 432.42 | 1002.68 | 422.296 | 421.688 | 809.946 | 417.287 | -392.66 |
| nll_probe_mean | 800000 | 1.48082 | 0.272498 | 1.28115 | 1.58594 | 1.57258 | 1.3863 | 1.57533 | 0.189024 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 800000 | 116.974 | 62.4756 | 42.7499 | 119.606 | 186.462 | 63.0375 | 170.911 | 107.874 |
| dist_to_ref_sq_over_d | 800000 | 0.0585375 | 0.0480031 | 0.00800124 | 0.0490146 | 0.11683 | 0.0172879 | 0.0997871 | 0.0824992 |
| dist_to_ref_over_sqrt_d | 800000 | 0.213414 | 0.113983 | 0.0779949 | 0.218214 | 0.34019 | 0.115009 | 0.311819 | 0.19681 |
| dist_to_ref_over_ou_radius | 800000 | 0.116891 | 0.0624313 | 0.0427196 | 0.119521 | 0.18633 | 0.0629928 | 0.17079 | 0.107797 |
| theta_norm | 800000 | 121.336 | 59.189 | 51.4073 | 122.325 | 188.247 | 69.7985 | 172.873 | 103.074 |
| v_norm | 800000 | 544.577 | 1.43524 | 546.288 | 544.317 | 543.17 | 545.713 | 543.442 | -2.27057 |
| kinetic_energy | 800000 | 148283 | 782.247 | 149216 | 148141 | 147517 | 148902 | 147665 | -1236.94 |
| theta_v_cosine | 800000 | 0.713033 | 0.128802 | 0.691087 | 0.772761 | 0.676364 | 0.726786 | 0.699281 | -0.0275057 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 799992 | 0.0779643 | 0.0256101 | 0.0534446 | 0.0764411 | 0.103241 | 0.0584151 | 0.0975135 | 0.0390984 |
| noise_step_norm | 800000 | 3.00117 | 0.00386527 | 3.00118 | 3.00118 | 3.00115 | 3.00118 | 3.00117 | -9.78425e-06 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 800000 | 2733.85 | 2166.26 | 452.745 | 2304.81 | 5364.29 | 872.22 | 4595.48 | 3723.26 |
| U_data | 800000 | 758.099 | 139.513 | 655.895 | 811.924 | 805.056 | 709.729 | 806.47 | 96.7413 |
| ce_mean_train | 800000 | 1.48066 | 0.272487 | 1.28105 | 1.58579 | 1.57238 | 1.38619 | 1.57514 | 0.188948 |
| U_data_minus_ce | 800000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 800000 / 800000

### Stability gates

- **max U_train** (iter_metrics): 8000.32 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 218.898

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; otherwise those cells are **—**.


- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 2.1096 | 2.4105 | 3.4105 | — | — | 0.2499 | 99960 | nan | nan |
| f_margin | 1.1539 | 1.6993 | 2.6081 | — | — | 0.2499 | 99960 | nan | nan |
| f_dist | 0.9998 | 3.1992 | 3.2008 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9998 | 3.1992 | 3.2008 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6545 | 3.6566 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6545 | 3.6566 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj1 | 3.8942 | 3.5092 | 4.0323 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj2 | 3.2632 | 6.6859 | 8.6269 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc1 | 1.6315 | 1.7864 | 3.3402 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc2 | 1.4682 | 2.0139 | 2.3498 | — | — | 0.2499 | 99960 | nan | nan |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 4.0917 | 0.4002 | 0.5229 | — | — | 0.1249 | 49960 | nan | nan |
| f_margin | 1.9254 | 0.9045 | 1.6401 | — | — | 0.1249 | 49960 | nan | nan |
| f_dist | 0.9996 | 3.3580 | 3.3600 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9996 | 3.3580 | 3.3600 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9996 | 3.5548 | 3.5571 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9996 | 3.5548 | 3.5571 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj1 | 8.8297 | 3.4506 | 4.1238 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj2 | 12.9620 | 4.9846 | 8.3673 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc1 | 1.6851 | 4.1825 | 8.1491 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc2 | 1.2610 | 2.1507 | 4.3939 | — | — | 0.1249 | 49960 | nan | nan |
