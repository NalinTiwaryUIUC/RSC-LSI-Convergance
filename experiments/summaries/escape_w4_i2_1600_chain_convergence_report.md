# Chain convergence and diagnostics report

Runs discovered: **4** in `experiments/runs` (glob `w4_*_ul_initI2_step1600_chain*`)

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
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain0` | 0 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain1` | 1 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain2` | 2 |
| `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0023 | 18.0 | 17.2 | 5000 | 90.1254 | 859.47 | 873.842 | 14.3721 |
| f_margin | 1.0014 | 18.8 | 17.4 | 5000 | 94.0762 | -0.241553 | -0.348471 | -0.106919 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6577 | 20666.4 | 119219 | 98552.1 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 800000 | 866.601 | 95.753 | 839.388 | 890.693 | 869.629 | 859.355 | 873.846 | 14.4906 |
| f_margin | 800000 | -0.294841 | 0.253016 | -0.175527 | -0.364568 | -0.342971 | -0.24121 | -0.348473 | -0.107263 |
| ce_mean_train | 800000 | 1.69202 | 0.186968 | 1.63915 | 1.73904 | 1.6977 | 1.67806 | 1.70598 | 0.0279132 |
| margin_probe | 800000 | -0.294841 | 0.253016 | -0.175527 | -0.364568 | -0.342971 | -0.24121 | -0.348473 | -0.107263 |
| pmax_mean | 800000 | 0.336434 | 0.0424498 | 0.331507 | 0.328161 | 0.349244 | 0.327731 | 0.345136 | 0.0174049 |
| U_train | 800000 | 11511.7 | 8610.05 | 2429.42 | 9833.88 | 21955.3 | 4114.56 | 18908.8 | 14794.3 |
| grad_norm | 800000 | 960.101 | 1300.28 | 1806.69 | 573.035 | 514.094 | 1395.18 | 525.024 | -870.153 |
| nll_probe_mean | 800000 | 1.69258 | 0.187017 | 1.63943 | 1.73963 | 1.69849 | 1.67843 | 1.70673 | 0.028302 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 800000 | 233.251 | 124.575 | 85.2236 | 238.544 | 371.786 | 125.698 | 340.803 | 215.104 |
| dist_to_ref_sq_over_d | 800000 | 0.0587274 | 0.0481457 | 0.00802538 | 0.0491934 | 0.117192 | 0.0173477 | 0.100107 | 0.0827593 |
| dist_to_ref_over_sqrt_d | 800000 | 0.213761 | 0.114166 | 0.0781025 | 0.218612 | 0.34072 | 0.115195 | 0.312326 | 0.197131 |
| dist_to_ref_over_ou_radius | 800000 | 0.117081 | 0.0625311 | 0.0427785 | 0.119739 | 0.18662 | 0.063095 | 0.171068 | 0.107973 |
| theta_norm | 800000 | 237.329 | 121.012 | 93.876 | 240.802 | 373.193 | 132.288 | 342.371 | 210.084 |
| v_norm | 800000 | 1085.04 | 3.32593 | 1089.02 | 1084.8 | 1081.41 | 1087.99 | 1082.1 | -5.89088 |
| kinetic_energy | 800000 | 588665 | 3610.08 | 592988 | 588397 | 584729 | 591861 | 585468 | -6392.76 |
| theta_v_cosine | 800000 | 0.740647 | 0.120674 | 0.761057 | 0.78298 | 0.67975 | 0.77742 | 0.703875 | -0.0735453 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 799992 | 0.286984 | 0.120658 | 0.144062 | 0.30366 | 0.409517 | 0.187144 | 0.386823 | 0.199679 |
| noise_step_norm | 800000 | 5.97472 | 0.00386726 | 5.97471 | 5.97474 | 5.97471 | 5.97472 | 5.97472 | 7.20935e-06 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 800000 | 10645.4 | 8598.16 | 1590.17 | 8943.49 | 21086.1 | 3255.39 | 18035.4 | 14780 |
| U_data | 800000 | 866.314 | 95.7278 | 839.244 | 890.39 | 869.22 | 859.168 | 873.46 | 14.2916 |
| ce_mean_train | 800000 | 1.69202 | 0.186968 | 1.63915 | 1.73904 | 1.6977 | 1.67806 | 1.70598 | 0.0279132 |
| U_data_minus_ce | 800000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 800000 / 800000

### Stability gates

- **max U_train** (iter_metrics): 29185.6 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 434.472

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; otherwise those cells are **—**.


- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.4399 | 3.6393 | 5.3703 | — | — | 0.2499 | 99960 | nan | nan |
| f_margin | 1.4542 | 1.2922 | 2.1563 | — | — | 0.2499 | 99960 | nan | nan |
| f_dist | 0.9998 | 3.2025 | 3.2031 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9998 | 3.2025 | 3.2031 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6579 | 3.6588 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6579 | 3.6588 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj1 | 3.3967 | 1.7227 | 2.1096 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj2 | 8.4983 | 4.1645 | 6.8872 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc1 | 2.2265 | 1.6520 | 2.3856 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc2 | 1.6023 | 0.9363 | 2.1984 | — | — | 0.2499 | 99960 | nan | nan |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 2.6425 | 1.4762 | 1.7801 | — | — | 0.1249 | 49960 | nan | nan |
| f_margin | 2.4964 | 0.2665 | 0.4211 | — | — | 0.1249 | 49960 | nan | nan |
| f_dist | 0.9996 | 3.3588 | 3.3601 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9996 | 3.3588 | 3.3601 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9996 | 3.5554 | 3.5567 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9996 | 3.5554 | 3.5567 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj1 | 4.1339 | 2.7623 | 3.2461 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj2 | 18.2881 | 4.0405 | 4.7938 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc1 | 3.3460 | 1.3039 | 1.9263 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc2 | 1.4178 | 2.6763 | 3.3109 | — | — | 0.1249 | 49960 | nan | nan |
