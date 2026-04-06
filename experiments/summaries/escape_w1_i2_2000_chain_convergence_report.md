# Chain convergence and diagnostics report

Runs discovered: **4** in `experiments/runs` (glob `w1_*_T100000_*ul_initI2_step2000_chain*`)

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
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain0` | 0 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain1` | 1 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain2` | 2 |
| `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0012 | 6.2 | 5.9 | 5000 | 31.1049 | 611.427 | 752.274 | 140.847 |
| f_margin | 1.0007 | 7.2 | 6.9 | 5000 | 36.0526 | 0.283981 | -0.181048 | -0.465029 |
| f_dist | 0.9999 | 3.5 | 3.5 | 5000 | 17.6596 | 1312.62 | 7574.43 | 6261.81 |

*Approx. post-burn grad evals per chain: **200000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 800000 | 681.789 | 141.76 | 547.597 | 741.192 | 754.377 | 611.306 | 752.271 | 140.965 |
| f_margin | 800000 | 0.0517172 | 0.488706 | 0.481432 | -0.120451 | -0.198255 | 0.284464 | -0.181029 | -0.465493 |
| ce_mean_train | 800000 | 1.33157 | 0.276884 | 1.06947 | 1.4476 | 1.47334 | 1.19391 | 1.46923 | 0.27532 |
| margin_probe | 800000 | 0.0517172 | 0.488706 | 0.481432 | -0.120451 | -0.198255 | 0.284464 | -0.181029 | -0.465493 |
| pmax_mean | 800000 | 0.411939 | 0.0543913 | 0.4531 | 0.385839 | 0.397321 | 0.430023 | 0.393854 | -0.0361693 |
| U_train | 800000 | 1412.1 | 641.845 | 698.296 | 1362.83 | 2152.73 | 868.714 | 1955.49 | 1086.77 |
| grad_norm | 800000 | 707.104 | 379.766 | 1109.19 | 514.255 | 504.017 | 916.133 | 498.076 | -418.056 |
| nll_probe_mean | 800000 | 1.33162 | 0.276875 | 1.06953 | 1.44764 | 1.47339 | 1.19396 | 1.46928 | 0.275322 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 800000 | 58.796 | 31.3916 | 21.509 | 60.1015 | 93.7192 | 31.6919 | 85.9 | 54.2081 |
| dist_to_ref_sq_over_d | 800000 | 0.0580781 | 0.0476249 | 0.00794933 | 0.048612 | 0.11592 | 0.0171515 | 0.0990048 | 0.0818532 |
| dist_to_ref_over_sqrt_d | 800000 | 0.212591 | 0.113504 | 0.0777709 | 0.217312 | 0.338865 | 0.11459 | 0.310592 | 0.196002 |
| dist_to_ref_over_ou_radius | 800000 | 0.116441 | 0.0621686 | 0.0425969 | 0.119027 | 0.185604 | 0.0627635 | 0.170118 | 0.107355 |
| theta_norm | 800000 | 63.7777 | 28.3076 | 30.5854 | 63.6292 | 96.1378 | 39.0269 | 88.5284 | 49.5015 |
| v_norm | 800000 | 274.007 | 0.962668 | 274.813 | 274.123 | 273.111 | 274.57 | 273.443 | -1.12713 |
| kinetic_energy | 800000 | 37540.3 | 264.086 | 37761.6 | 37571.9 | 37294.8 | 37694.8 | 37385.8 | -309.014 |
| theta_v_cosine | 800000 | 0.668787 | 0.13963 | 0.590875 | 0.748023 | 0.667503 | 0.649838 | 0.687737 | 0.0378984 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 799992 | 0.0244632 | 0.0115977 | 0.0267198 | 0.0202462 | 0.0263658 | 0.0239155 | 0.0250109 | 0.00109542 |
| noise_step_norm | 800000 | 1.51433 | 0.00386018 | 1.51432 | 1.51434 | 1.51432 | 1.51432 | 1.51433 | 1.52501e-05 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 800000 | 730.337 | 550.185 | 150.726 | 621.663 | 1398.38 | 257.432 | 1203.24 | 945.81 |
| U_data | 800000 | 681.764 | 141.765 | 547.571 | 741.172 | 754.351 | 611.282 | 752.246 | 140.964 |
| ce_mean_train | 800000 | 1.33157 | 0.276884 | 1.06947 | 1.4476 | 1.47334 | 1.19391 | 1.46923 | 0.27532 |
| U_data_minus_ce | 800000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 800000 / 800000

### Stability gates

- **max U_train** (iter_metrics): 2622.47 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 111.433

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.5934 | 2.4466 | 5.2674 | 5.6781 | 11.2918 | 0.2499 | 99960 | 22.7213 | 56.8032 |
| f_margin | 1.0645 | 3.1674 | 5.7115 | 6.8950 | 22.8667 | 0.2499 | 99960 | 27.591 | 68.9775 |
| f_dist | 0.9998 | 3.2031 | 3.2125 | 6.0874 | 62.7016 | 0.2499 | 99960 | 24.3592 | 60.898 |
| dist_to_ref_sq_over_d | 0.9998 | 3.2031 | 3.2125 | 6.0874 | 62.7016 | 0.2499 | 99960 | 24.3592 | 60.898 |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6591 | 3.6697 | 6.0874 | 62.7016 | 0.2499 | 99960 | 24.3592 | 60.898 |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6591 | 3.6697 | 6.0874 | 62.7016 | 0.2499 | 99960 | 24.3592 | 60.898 |
| f_proj1 | 4.4588 | 6.6761 | 7.7780 | 4.3668 | 11.3895 | 0.2499 | 99960 | 17.4743 | 43.6857 |
| f_proj2 | 2.4043 | 2.7353 | 2.9827 | 4.3641 | 11.3604 | 0.2499 | 99960 | 17.4633 | 43.6583 |
| f_pc1 | 1.1624 | 1.9229 | 2.9149 | 7.8496 | 15.3166 | 0.2499 | 99960 | 31.411 | 78.5275 |
| f_pc2 | 1.7193 | 1.3197 | 2.7846 | 5.7262 | 11.0585 | 0.2499 | 99960 | 22.9139 | 57.2847 |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.9412 | 1.7794 | 2.7410 | 5.0129 | 12.4534 | 0.1249 | 49960 | 40.1353 | 100.338 |
| f_margin | 1.0903 | 1.5181 | 2.9828 | 6.3912 | 11.5153 | 0.1249 | 49960 | 51.1706 | 127.926 |
| f_dist | 0.9998 | 3.3583 | 3.3630 | 6.0924 | 62.3867 | 0.1249 | 49960 | 48.7783 | 121.946 |
| dist_to_ref_sq_over_d | 0.9998 | 3.3583 | 3.3630 | 6.0924 | 62.3867 | 0.1249 | 49960 | 48.7783 | 121.946 |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.5549 | 3.5592 | 6.0924 | 62.3867 | 0.1249 | 49960 | 48.7783 | 121.946 |
| dist_to_ref_over_ou_radius | 0.9998 | 3.5549 | 3.5592 | 6.0924 | 62.3867 | 0.1249 | 49960 | 48.7783 | 121.946 |
| f_proj1 | 16.7907 | 4.6131 | 5.6049 | 4.3341 | 11.4093 | 0.1249 | 49960 | 34.7007 | 86.7516 |
| f_proj2 | 3.9471 | 3.1348 | 3.4370 | 4.2703 | 11.4093 | 0.1249 | 49960 | 34.1894 | 85.4736 |
| f_pc1 | 1.7473 | 1.9798 | 3.8853 | 5.6345 | 11.4603 | 0.1249 | 49960 | 45.1117 | 112.779 |
| f_pc2 | 3.4251 | 1.6656 | 2.8013 | 4.6004 | 11.4093 | 0.1249 | 49960 | 36.8328 | 92.082 |
