# Chain convergence and diagnostics report

Runs discovered: **16** in `experiments/runs` (glob `w2_*_ul_initI*_chain*`)

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
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain0` | 0 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain1` | 1 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain2` | 2 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0849 | 11.0 | 9.4 | 5000 | 54.9485 | 873.731 | 872.867 | -0.864095 |
| f_margin | 1.0669 | 31.4 | 10.4 | 5000 | 156.7602 | -0.335858 | -0.346293 | -0.0104355 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6596 | 5334.6 | 30190.7 | 24856.1 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 400000 | 873.324 | 52.9498 | 859.461 | 892.246 | 868.415 | 873.777 | 872.872 | -0.904385 |
| f_margin | 400000 | -0.341148 | 0.104498 | -0.315574 | -0.361669 | -0.346053 | -0.336001 | -0.346296 | -0.010295 |
| ce_mean_train | 400000 | 1.70558 | 0.103419 | 1.67859 | 1.74252 | 1.69592 | 1.70652 | 1.70464 | -0.00187899 |
| margin_probe | 400000 | -0.341148 | 0.104498 | -0.315574 | -0.361669 | -0.346053 | -0.336001 | -0.346296 | -0.010295 |
| pmax_mean | 400000 | 0.349045 | 0.0529621 | 0.373752 | 0.327954 | 0.345536 | 0.356601 | 0.341489 | -0.0151127 |
| U_train | 400000 | 3630.79 | 2173.1 | 1331.01 | 3220.24 | 6261.39 | 1765.83 | 5495.75 | 3729.92 |
| grad_norm | 400000 | 1180.39 | 1785.76 | 2457.7 | 572.904 | 530.251 | 1834.82 | 525.949 | -1308.87 |
| nll_probe_mean | 400000 | 1.70571 | 0.103418 | 1.67863 | 1.74267 | 1.69612 | 1.70659 | 1.70483 | -0.00176638 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 400000 | 118.172 | 61.5926 | 45.0602 | 120.322 | 187.045 | 64.8238 | 171.519 | 106.695 |
| dist_to_ref_sq_over_d | 400000 | 0.0591099 | 0.0481293 | 0.00842811 | 0.0495779 | 0.117553 | 0.0177475 | 0.100472 | 0.0827248 |
| dist_to_ref_over_sqrt_d | 400000 | 0.215598 | 0.112372 | 0.0822099 | 0.219521 | 0.341254 | 0.118268 | 0.312928 | 0.19466 |
| dist_to_ref_over_ou_radius | 400000 | 0.118088 | 0.0615489 | 0.0450282 | 0.120237 | 0.186913 | 0.0647779 | 0.171398 | 0.10662 |
| theta_norm | 400000 | 122.176 | 58.7927 | 52.7845 | 122.968 | 188.758 | 70.9434 | 173.409 | 102.465 |
| v_norm | 400000 | 545.12 | 1.76812 | 547.27 | 544.793 | 543.35 | 546.555 | 543.684 | -2.87117 |
| kinetic_energy | 400000 | 148579 | 964.7 | 149753 | 148400 | 147615 | 149362 | 147797 | -1565.76 |
| theta_v_cosine | 400000 | 0.705529 | 0.130993 | 0.672719 | 0.769538 | 0.675246 | 0.713218 | 0.69784 | -0.0153781 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 399996 | 0.0682783 | 0.0980611 | 0.0250213 | 0.07572 | 0.10304 | 0.0393748 | 0.0971818 | 0.0578069 |
| noise_step_norm | 400000 | 3.00117 | 0.00386527 | 3.00118 | 3.00118 | 3.00115 | 3.00118 | 3.00117 | -9.78425e-06 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 400000 | 2757.53 | 2170.54 | 471.577 | 2328.07 | 5393.08 | 892.092 | 4622.97 | 3730.88 |
| U_data | 400000 | 873.256 | 52.9505 | 859.436 | 892.17 | 868.313 | 873.737 | 872.775 | -0.962045 |
| ce_mean_train | 400000 | 1.70558 | 0.103419 | 1.67859 | 1.74252 | 1.69592 | 1.70652 | 1.70464 | -0.00187899 |
| U_data_minus_ce | 400000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 400000 / 400000

### Stability gates

- **max U_train** (iter_metrics): 8084.6 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 219.35

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; otherwise those cells are **—**.


- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.5144 | 2.4030 | 4.5298 | — | — | 0.2499 | 99960 | nan | nan |
| f_margin | 1.2001 | 0.5868 | 1.0246 | — | — | 0.2499 | 99960 | nan | nan |
| f_dist | 0.9998 | 3.2005 | 3.2023 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9998 | 3.2005 | 3.2023 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6536 | 3.6560 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6536 | 3.6560 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj1 | 4.0064 | 3.5104 | 4.0684 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj2 | 2.9543 | 6.8874 | 9.6716 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc1 | 1.3380 | 1.5179 | 2.0207 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc2 | 1.9193 | 2.6871 | 7.5506 | — | — | 0.2499 | 99960 | nan | nan |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.7614 | 3.0098 | 7.2470 | — | — | 0.1249 | 49960 | nan | nan |
| f_margin | 1.2113 | 1.4733 | 2.6579 | — | — | 0.1249 | 49960 | nan | nan |
| f_dist | 0.9996 | 3.3585 | 3.3605 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9996 | 3.3585 | 3.3605 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9996 | 3.5546 | 3.5568 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9996 | 3.5546 | 3.5568 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj1 | 9.1453 | 3.3873 | 4.1154 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj2 | 11.5759 | 4.6217 | 7.7099 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc1 | 1.3400 | 2.1780 | 4.3517 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc2 | 2.4285 | 2.0675 | 3.9769 | — | — | 0.1249 | 49960 | nan | nan |

## Group (12 chains)

**sampler=underdamped γ=3.0 h=5e-06 α=0.3 β=1.0 T=100000 B=0 S=20 n_train=512 arch=small_resnet_ln nb=1**

| run_dir | chain_id |
|---------|----------|
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain0` | 0 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2375_chain0` | 0 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain0` | 0 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain1` | 1 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2375_chain1` | 1 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain1` | 1 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain2` | 2 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2375_chain2` | 2 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain2` | 2 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain3` | 3 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2375_chain3` | 3 |
| `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0030 | 8.9 | 8.6 | 5000 | 44.4473 | 716.645 | 807.935 | 91.2902 |
| f_margin | 1.0016 | 9.7 | 9.1 | 5000 | 48.4805 | 0.0515334 | -0.261234 | -0.312768 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6614 | 5197.12 | 29987.8 | 24790.7 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 1200000 | 762.225 | 135.218 | 664.656 | 814.304 | 806.377 | 716.514 | 807.936 | 91.4226 |
| f_margin | 1200000 | -0.104603 | 0.426997 | 0.202803 | -0.245936 | -0.265792 | 0.0520254 | -0.261231 | -0.313257 |
| ce_mean_train | 1200000 | 1.48857 | 0.264087 | 1.29805 | 1.59029 | 1.57475 | 1.39933 | 1.57781 | 0.178483 |
| margin_probe | 1200000 | -0.104603 | 0.426997 | 0.202803 | -0.245936 | -0.265792 | 0.0520254 | -0.261231 | -0.313257 |
| pmax_mean | 1200000 | 0.370749 | 0.0529936 | 0.390934 | 0.352882 | 0.368498 | 0.376484 | 0.365013 | -0.0114709 |
| U_train | 1200000 | 3496.01 | 2225.04 | 1117.19 | 3119.01 | 6170.76 | 1588.55 | 5403.46 | 3814.91 |
| grad_norm | 1200000 | 603.275 | 420.738 | 986.49 | 414.018 | 415.022 | 795.579 | 410.971 | -384.608 |
| nll_probe_mean | 1200000 | 1.48872 | 0.264098 | 1.29816 | 1.59044 | 1.57496 | 1.39944 | 1.578 | 0.17856 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 1200000 | 116.98 | 62.4786 | 42.7521 | 119.612 | 186.471 | 63.041 | 170.92 | 107.879 |
| dist_to_ref_sq_over_d | 1200000 | 0.0585435 | 0.0480076 | 0.00800211 | 0.04902 | 0.116842 | 0.0172899 | 0.0997971 | 0.0825073 |
| dist_to_ref_over_sqrt_d | 1200000 | 0.213425 | 0.113989 | 0.077999 | 0.218226 | 0.340207 | 0.115015 | 0.311834 | 0.196819 |
| dist_to_ref_over_ou_radius | 1200000 | 0.116897 | 0.0624343 | 0.0427218 | 0.119527 | 0.186339 | 0.0629963 | 0.170799 | 0.107802 |
| theta_norm | 1200000 | 121.332 | 59.196 | 51.3951 | 122.324 | 188.25 | 69.7897 | 172.875 | 103.086 |
| v_norm | 1200000 | 544.594 | 1.44176 | 546.313 | 544.335 | 543.176 | 545.736 | 543.451 | -2.28536 |
| kinetic_energy | 1200000 | 148292 | 785.818 | 149229 | 148151 | 147520 | 148915 | 147670 | -1245.03 |
| theta_v_cosine | 1200000 | 0.713113 | 0.128781 | 0.691265 | 0.772801 | 0.676387 | 0.72692 | 0.699307 | -0.0276131 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 1199988 | 0.0776898 | 0.0256563 | 0.0526552 | 0.0764012 | 0.103239 | 0.0579095 | 0.0974702 | 0.0395607 |
| noise_step_norm | 1200000 | 3.00117 | 0.00386526 | 3.00118 | 3.00118 | 3.00115 | 3.00118 | 3.00117 | -9.78425e-06 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 1200000 | 2733.86 | 2166.41 | 452.59 | 2304.78 | 5364.48 | 872.094 | 4595.62 | 3723.53 |
| U_data | 1200000 | 762.147 | 135.212 | 664.603 | 814.228 | 806.274 | 716.456 | 807.839 | 91.3832 |
| ce_mean_train | 1200000 | 1.48857 | 0.264087 | 1.29805 | 1.59029 | 1.57475 | 1.39933 | 1.57781 | 0.178483 |
| U_data_minus_ce | 1200000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 1200000 / 1200000

### Stability gates

- **max U_train** (iter_metrics): 8009.19 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 218.915

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; otherwise those cells are **—**.


- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 2.3560 | 2.4683 | 5.0961 | — | — | 0.2499 | 99960 | nan | nan |
| f_margin | 1.2971 | 1.4221 | 2.9492 | — | — | 0.2499 | 99960 | nan | nan |
| f_dist | 0.9998 | 3.1993 | 3.2010 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9998 | 3.1993 | 3.2010 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6545 | 3.6568 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6545 | 3.6568 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj1 | 3.5762 | 3.4999 | 4.0323 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj2 | 2.9773 | 6.6583 | 8.7400 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc1 | 1.4574 | 2.2154 | 5.2142 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc2 | 1.3822 | 2.1103 | 4.4090 | — | — | 0.2499 | 99960 | nan | nan |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 4.1877 | 0.7731 | 3.2903 | — | — | 0.1249 | 49960 | nan | nan |
| f_margin | 1.8061 | 1.0320 | 2.1588 | — | — | 0.1249 | 49960 | nan | nan |
| f_dist | 0.9996 | 3.3580 | 3.3602 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9996 | 3.3580 | 3.3602 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9996 | 3.5549 | 3.5573 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9996 | 3.5549 | 3.5573 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj1 | 8.0478 | 3.4384 | 4.1359 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj2 | 11.5807 | 4.9023 | 8.3803 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc1 | 1.5828 | 3.7040 | 8.1481 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc2 | 1.2345 | 2.4984 | 9.2589 | — | — | 0.1249 | 49960 | nan | nan |
