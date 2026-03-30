# Chain convergence and diagnostics report

Runs discovered: **15** in `experiments/runs` (glob `w1_*_ul_initI*_chain*`)

## Method

- **R̂**: Gelman–Rubin on **parallel chains** (same config, different `chain_id`) using aligned `samples_metrics` traces; if only one chain, split-chain R̂ is used.

- **ESS (bulk)**: Autocorrelation ESS per chain; table shows **mean** and **min** across chains.

- **ESS rate**: mean ESS divided by approximate **post-burn-in gradient evaluations** (×2 for underdamped BAOAB).

- **iter_metrics**: pooled early/mid/late means by step tertiles across logged rows.

- **Half split**: for `iter_metrics`, records are sorted by `step` and split at the midpoint **by count** (first half vs second half of pooled rows). **Δ** = mean(2nd half) − mean(1st half). For `samples_metrics`, each chain’s trace is split at its midpoint index; reported means are averaged across chains, then **Δ** = mean(2nd) − mean(1st).

- **Late-window analytics** (below): use only the **last *f*** fraction of saved samples per chain; **R̂** and **multi-chain ESS** (ArviZ bulk/tail) are computed on that window. **drift_z** = |mean(2nd half)−mean(1st half)|/std(2nd half) **within** the late window, per chain (mean/max across chains). **ESS/T_phys** = ESS_bulk / *T*_analysis; **ESS/(1e6 grad)** uses the same step span.


## Group (3 chains)

**sampler=underdamped γ=3.0 h=5e-06 α=0.3 β=1.0 T=100000 B=0 S=20 n_train=512 arch=small_resnet_ln nb=1**

| run_dir | chain_id |
|---------|----------|
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain1` | 1 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain2` | 2 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI3_sigma0p02_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0726 | 7.4 | 6.3 | 5000 | 36.9617 | 745.944 | 801.222 | 55.2777 |
| f_margin | 1.0293 | 8.9 | 7.6 | 5000 | 44.7146 | -0.102238 | -0.264586 | -0.162347 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6557 | 1354.07 | 7664.56 | 6310.49 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 300000 | 773.59 | 73.857 | 714.325 | 806.091 | 799.566 | 745.957 | 801.223 | 55.2662 |
| f_margin | 300000 | -0.183399 | 0.211435 | -0.0174713 | -0.25803 | -0.27201 | -0.102218 | -0.26458 | -0.162362 |
| ce_mean_train | 300000 | 1.51088 | 0.144245 | 1.39516 | 1.57436 | 1.5616 | 1.45693 | 1.56484 | 0.107913 |
| margin_probe | 300000 | -0.183399 | 0.211435 | -0.0174713 | -0.25803 | -0.27201 | -0.102218 | -0.26458 | -0.162362 |
| pmax_mean | 300000 | 0.398856 | 0.0692008 | 0.456516 | 0.372082 | 0.368879 | 0.430688 | 0.367024 | -0.0636641 |
| U_train | 300000 | 1515.88 | 589.331 | 873.768 | 1439.14 | 2213.59 | 1012.71 | 2019.05 | 1006.34 |
| grad_norm | 300000 | 1311.25 | 1834.29 | 2930.22 | 604.339 | 426.021 | 2165.81 | 456.693 | -1709.12 |
| nll_probe_mean | 300000 | 1.51092 | 0.144252 | 1.39517 | 1.5744 | 1.56165 | 1.45695 | 1.56489 | 0.107942 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 300000 | 59.5438 | 31.0277 | 22.7192 | 60.6134 | 94.247 | 32.6664 | 86.4211 | 53.7547 |
| dist_to_ref_sq_over_d | 300000 | 0.0589381 | 0.0479875 | 0.00841101 | 0.0494184 | 0.117219 | 0.0176933 | 0.100183 | 0.0824896 |
| dist_to_ref_over_sqrt_d | 300000 | 0.215295 | 0.112188 | 0.0821469 | 0.219163 | 0.340773 | 0.118113 | 0.312477 | 0.194363 |
| dist_to_ref_over_ou_radius | 300000 | 0.117922 | 0.0614479 | 0.0449937 | 0.12004 | 0.186649 | 0.0646934 | 0.171151 | 0.106457 |
| theta_norm | 300000 | 64.4874 | 28.1085 | 31.5767 | 64.2296 | 96.6802 | 39.8958 | 89.0789 | 49.1831 |
| v_norm | 300000 | 274.833 | 1.35273 | 276.173 | 274.885 | 273.481 | 275.746 | 273.92 | -1.82646 |
| kinetic_energy | 300000 | 37767.5 | 372.208 | 38136.5 | 37781.1 | 37396.1 | 38018.6 | 37516.3 | -502.276 |
| theta_v_cosine | 300000 | 0.660878 | 0.141507 | 0.57348 | 0.742974 | 0.666024 | 0.63603 | 0.685726 | 0.0496956 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 299997 | 0.017322 | 0.0782997 | 0.00574271 | 0.019753 | 0.0262014 | 0.00993791 | 0.0247061 | 0.0147682 |
| noise_step_norm | 300000 | 1.51433 | 0.00385985 | 1.51432 | 1.51434 | 1.51432 | 1.51432 | 1.51433 | 1.15027e-05 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 300000 | 742.306 | 553.214 | 159.448 | 633.068 | 1414.05 | 266.761 | 1217.85 | 951.09 |
| U_data | 300000 | 773.573 | 73.8534 | 714.32 | 806.072 | 799.54 | 745.947 | 801.199 | 55.2515 |
| ce_mean_train | 300000 | 1.51088 | 0.144245 | 1.39516 | 1.57436 | 1.5616 | 1.45693 | 1.56484 | 0.107913 |
| U_data_minus_ce | 300000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 300000 / 300000

### Stability gates

- **max U_train** (iter_metrics): 2694.02 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 111.981

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; otherwise those cells are **—**.


- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 3.0978 | 1.8775 | 3.4064 | — | — | 0.2499 | 99960 | nan | nan |
| f_margin | 1.3411 | 1.5131 | 2.8406 | — | — | 0.2499 | 99960 | nan | nan |
| f_dist | 0.9999 | 3.2071 | 3.2159 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9999 | 3.2071 | 3.2159 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9999 | 3.6613 | 3.6711 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9999 | 3.6613 | 3.6711 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj1 | 1.6169 | 6.4874 | 6.7972 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj2 | 2.5052 | 2.6940 | 2.8361 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc1 | 1.0210 | 1.8846 | 2.6884 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc2 | 1.0843 | 1.3459 | 2.6777 | — | — | 0.2499 | 99960 | nan | nan |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 4.0542 | 1.0648 | 2.5650 | — | — | 0.1249 | 49960 | nan | nan |
| f_margin | 1.5577 | 1.5747 | 1.9958 | — | — | 0.1249 | 49960 | nan | nan |
| f_dist | 1.0001 | 3.3596 | 3.3651 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_sq_over_d | 1.0001 | 3.3596 | 3.3651 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_sqrt_d | 1.0001 | 3.5553 | 3.5604 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_ou_radius | 1.0001 | 3.5553 | 3.5604 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj1 | 5.1568 | 4.4913 | 4.7017 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj2 | 4.0907 | 3.1715 | 3.3612 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc1 | 1.1398 | 3.0172 | 5.7800 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc2 | 1.5733 | 2.1688 | 2.6766 | — | — | 0.1249 | 49960 | nan | nan |

## Group (12 chains)

**sampler=underdamped γ=3.0 h=5e-06 α=0.3 β=1.0 T=100000 B=0 S=20 n_train=512 arch=small_resnet_ln nb=1**

| run_dir | chain_id |
|---------|----------|
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain0` | 0 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain0` | 0 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain0` | 0 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain1` | 1 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain1` | 1 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain1` | 1 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain2` | 2 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain2` | 2 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain2` | 2 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI1_chain3` | 3 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain3` | 3 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0219 | 6.3 | 5.5 | 5000 | 31.3572 | 617.337 | 753.607 | 136.27 |
| f_margin | 1.0123 | 7.2 | 6.6 | 5000 | 35.7852 | 0.268147 | -0.183313 | -0.45146 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6595 | 1313.1 | 7576.64 | 6263.53 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 1200000 | 685.413 | 141.565 | 554.549 | 743.979 | 755.584 | 617.221 | 753.605 | 136.384 |
| f_margin | 1200000 | 0.0426531 | 0.481873 | 0.46049 | -0.127029 | -0.198202 | 0.268603 | -0.183297 | -0.4519 |
| ce_mean_train | 1200000 | 1.33865 | 0.276502 | 1.08305 | 1.45305 | 1.4757 | 1.20546 | 1.47184 | 0.266373 |
| margin_probe | 1200000 | 0.0426531 | 0.481873 | 0.46049 | -0.127029 | -0.198202 | 0.268603 | -0.183297 | -0.4519 |
| pmax_mean | 1200000 | 0.409889 | 0.0563842 | 0.450542 | 0.384818 | 0.394766 | 0.427658 | 0.392121 | -0.0355368 |
| U_train | 1200000 | 1415.7 | 639.708 | 705.089 | 1365.58 | 2154.05 | 874.497 | 1956.9 | 1082.4 |
| grad_norm | 1200000 | 740.65 | 462.839 | 1199.84 | 529.48 | 499.929 | 984.636 | 496.665 | -487.971 |
| nll_probe_mean | 1200000 | 1.3387 | 0.276493 | 1.0831 | 1.45308 | 1.47575 | 1.20551 | 1.47188 | 0.266376 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 1200000 | 58.8052 | 31.3957 | 21.5128 | 60.1118 | 93.7325 | 31.6977 | 85.9126 | 54.215 |
| dist_to_ref_sq_over_d | 1200000 | 0.0580957 | 0.0476377 | 0.00795232 | 0.0486284 | 0.115953 | 0.0171578 | 0.0990335 | 0.0818758 |
| dist_to_ref_over_sqrt_d | 1200000 | 0.212624 | 0.113519 | 0.0777849 | 0.217349 | 0.338913 | 0.114611 | 0.310638 | 0.196027 |
| dist_to_ref_over_ou_radius | 1200000 | 0.116459 | 0.0621769 | 0.0426045 | 0.119047 | 0.18563 | 0.0627749 | 0.170144 | 0.107369 |
| theta_norm | 1200000 | 63.7706 | 28.3202 | 30.5631 | 63.6263 | 96.1415 | 39.0104 | 88.5308 | 49.5204 |
| v_norm | 1200000 | 274.034 | 0.969178 | 274.854 | 274.148 | 273.128 | 274.606 | 273.463 | -1.14352 |
| kinetic_energy | 1200000 | 37547.9 | 265.895 | 37772.9 | 37578.6 | 37299.6 | 37704.6 | 37391.1 | -313.538 |
| theta_v_cosine | 1200000 | 0.669038 | 0.139632 | 0.591411 | 0.748177 | 0.66757 | 0.650263 | 0.687813 | 0.0375507 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 1199988 | 0.0242338 | 0.0123749 | 0.026125 | 0.0201634 | 0.0263487 | 0.0234931 | 0.0249745 | 0.00148148 |
| noise_step_norm | 1200000 | 1.51433 | 0.00386018 | 1.51432 | 1.51434 | 1.51432 | 1.51432 | 1.51433 | 1.52501e-05 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 1200000 | 730.308 | 550.312 | 150.566 | 621.619 | 1398.49 | 257.3 | 1203.32 | 946.017 |
| U_data | 1200000 | 685.389 | 141.569 | 554.523 | 743.959 | 755.557 | 617.197 | 753.58 | 136.383 |
| ce_mean_train | 1200000 | 1.33865 | 0.276502 | 1.08305 | 1.45305 | 1.4757 | 1.20546 | 1.47184 | 0.266373 |
| U_data_minus_ce | 1200000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 1200000 / 1200000

### Stability gates

- **max U_train** (iter_metrics): 2641.91 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 111.547

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

*Note:* install **`arviz`** for multi-chain **ESS_bulk** / **ESS_tail**; otherwise those cells are **—**.


- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 2.9384 | 2.8715 | 6.2349 | — | — | 0.2499 | 99960 | nan | nan |
| f_margin | 1.5063 | 2.8852 | 5.7118 | — | — | 0.2499 | 99960 | nan | nan |
| f_dist | 0.9998 | 3.2032 | 3.2126 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9998 | 3.2032 | 3.2126 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6592 | 3.6699 | — | — | 0.2499 | 99960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6592 | 3.6699 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj1 | 4.1378 | 6.7111 | 7.8422 | — | — | 0.2499 | 99960 | nan | nan |
| f_proj2 | 2.2646 | 2.7322 | 3.0677 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc1 | 1.2006 | 2.8531 | 9.2841 | — | — | 0.2499 | 99960 | nan | nan |
| f_pc2 | 1.6721 | 1.2362 | 2.7843 | — | — | 0.2499 | 99960 | nan | nan |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 4.2084 | 1.7753 | 3.7105 | — | — | 0.1249 | 49960 | nan | nan |
| f_margin | 2.0162 | 2.3267 | 4.8319 | — | — | 0.1249 | 49960 | nan | nan |
| f_dist | 0.9998 | 3.3583 | 3.3631 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_sq_over_d | 0.9998 | 3.3583 | 3.3631 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.5549 | 3.5593 | — | — | 0.1249 | 49960 | nan | nan |
| dist_to_ref_over_ou_radius | 0.9998 | 3.5549 | 3.5593 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj1 | 15.6781 | 3.9597 | 5.6049 | — | — | 0.1249 | 49960 | nan | nan |
| f_proj2 | 3.6896 | 3.1506 | 3.4395 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc1 | 1.5580 | 1.9305 | 3.8893 | — | — | 0.1249 | 49960 | nan | nan |
| f_pc2 | 2.2856 | 3.0947 | 7.0854 | — | — | 0.1249 | 49960 | nan | nan |
