# Chain convergence and diagnostics report

Runs discovered: **4** in `experiments/runs` (glob `w1_*_T200000_*ul_initI2_step2000_chain*`)

## Method

- **R̂**: Gelman–Rubin on **parallel chains** (same config, different `chain_id`) using aligned `samples_metrics` traces; if only one chain, split-chain R̂ is used.

- **ESS (bulk)**: Autocorrelation ESS per chain; table shows **mean** and **min** across chains.

- **ESS rate**: mean ESS divided by approximate **post-burn-in gradient evaluations** (×2 for underdamped BAOAB).

- **iter_metrics**: pooled early/mid/late means by step tertiles across logged rows.

- **Half split**: for `iter_metrics`, records are sorted by `step` and split at the midpoint **by count** (first half vs second half of pooled rows). **Δ** = mean(2nd half) − mean(1st half). For `samples_metrics`, each chain’s trace is split at its midpoint index; reported means are averaged across chains, then **Δ** = mean(2nd) − mean(1st).

- **Late-window analytics** (below): use only the **last *f*** fraction of saved samples per chain; **R̂** and **multi-chain ESS** (ArviZ bulk/tail) are computed on that window. **drift_z** = |mean(2nd half)−mean(1st half)|/std(2nd half) **within** the late window, per chain (mean/max across chains). **ESS/T_phys** = ESS_bulk / *T*_analysis; **ESS/(1e6 grad)** uses the same step span.


## Group (4 chains)

**sampler=underdamped γ=3.0 h=5e-06 α=0.3 β=1.0 T=200000 B=0 S=20 n_train=512 arch=small_resnet_ln nb=1**

| run_dir | chain_id |
|---------|----------|
| `w1_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain0` | 0 |
| `w1_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain1` | 1 |
| `w1_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain2` | 2 |
| `w1_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2000_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0024 | 11.6 | 11.0 | 10000 | 29.0593 | 681.851 | 760.091 | 78.2407 |
| f_margin | 1.0007 | 12.9 | 12.1 | 10000 | 32.3446 | 0.0514661 | -0.223637 | -0.275103 |
| f_dist | 1.0000 | 5.8 | 5.8 | 10000 | 14.5276 | 4443.53 | 22122.3 | 17678.8 |

*Approx. post-burn grad evals per chain: **400000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 800000 | 720.94 | 107.82 | 644.395 | 755.916 | 761.286 | 681.789 | 760.091 | 78.3016 |
| f_margin | 800000 | -0.0859601 | 0.372409 | 0.180491 | -0.207447 | -0.22666 | 0.0517164 | -0.223637 | -0.275353 |
| ce_mean_train | 800000 | 1.40803 | 0.21059 | 1.25854 | 1.47634 | 1.48682 | 1.33157 | 1.48449 | 0.152919 |
| margin_probe | 800000 | -0.0859601 | 0.372409 | 0.180491 | -0.207447 | -0.22666 | 0.0517164 | -0.223637 | -0.275353 |
| pmax_mean | 800000 | 0.413273 | 0.0394322 | 0.419469 | 0.402353 | 0.417859 | 0.411938 | 0.414608 | 0.00266989 |
| U_train | 800000 | 2782.37 | 1581.04 | 1030.57 | 2605.51 | 4654.3 | 1412.1 | 4152.63 | 2740.53 |
| grad_norm | 800000 | 589.824 | 302.516 | 811.733 | 485.551 | 475.648 | 707.112 | 472.537 | -234.575 |
| nll_probe_mean | 800000 | 1.40809 | 0.210586 | 1.25858 | 1.4764 | 1.48689 | 1.33162 | 1.48455 | 0.152933 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 800000 | 103.039 | 51.6178 | 40.8052 | 107.639 | 158.976 | 58.796 | 147.281 | 88.4852 |
| dist_to_ref_sq_over_d | 800000 | 0.173635 | 0.132754 | 0.0282807 | 0.155083 | 0.332721 | 0.0580781 | 0.289192 | 0.231114 |
| dist_to_ref_over_sqrt_d | 800000 | 0.372561 | 0.186637 | 0.147541 | 0.389195 | 0.574818 | 0.212591 | 0.532531 | 0.31994 |
| dist_to_ref_over_ou_radius | 800000 | 0.20406 | 0.102225 | 0.0808117 | 0.213171 | 0.314841 | 0.116441 | 0.291679 | 0.175238 |
| theta_norm | 800000 | 106.382 | 49.2533 | 47.1073 | 109.835 | 160.562 | 63.7777 | 148.986 | 85.2088 |
| v_norm | 800000 | 273.5 | 0.947166 | 274.468 | 272.934 | 273.11 | 274.007 | 272.994 | -1.01288 |
| kinetic_energy | 800000 | 37401.6 | 259.441 | 37666.7 | 37246.5 | 37294.9 | 37540.3 | 37263 | -277.31 |
| theta_v_cosine | 800000 | 0.595458 | 0.128635 | 0.669449 | 0.628842 | 0.491241 | 0.668787 | 0.522128 | -0.146659 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 799996 | 0.0279564 | 0.0109182 | 0.023483 | 0.0281039 | 0.0321551 | 0.0244632 | 0.0314496 | 0.00698642 |
| noise_step_norm | 800000 | 1.51434 | 0.00386559 | 1.51433 | 1.51435 | 1.51435 | 1.51433 | 1.51436 | 3.18301e-05 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 800000 | 2061.46 | 1528.95 | 386.194 | 1849.63 | 3893.04 | 730.337 | 3392.57 | 2662.24 |
| U_data | 800000 | 720.912 | 107.822 | 644.371 | 755.888 | 761.254 | 681.765 | 760.059 | 78.2948 |
| ce_mean_train | 800000 | 1.40803 | 0.21059 | 1.25854 | 1.47634 | 1.48682 | 1.33157 | 1.48449 | 0.152919 |
| U_data_minus_ce | 800000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 800000 / 800000

### Stability gates

- **max U_train** (iter_metrics): 5793.76 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 182.793

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.7991 | 1.1541 | 2.0866 | 5.4789 | 11.3725 | 0.4999 | 199960 | 10.96 | 27.3999 |
| f_margin | 1.2253 | 0.8109 | 1.4644 | 10.8685 | 33.0322 | 0.4999 | 199960 | 21.7413 | 54.3533 |
| f_dist | 0.9999 | 3.3539 | 3.3570 | 6.0783 | 62.1090 | 0.4999 | 199960 | 12.159 | 30.3976 |
| dist_to_ref_sq_over_d | 0.9999 | 3.3539 | 3.3570 | 6.0783 | 62.1090 | 0.4999 | 199960 | 12.159 | 30.3976 |
| dist_to_ref_over_sqrt_d | 0.9999 | 3.7697 | 3.7741 | 6.0783 | 62.1090 | 0.4999 | 199960 | 12.159 | 30.3976 |
| dist_to_ref_over_ou_radius | 0.9999 | 3.7697 | 3.7741 | 6.0783 | 62.1090 | 0.4999 | 199960 | 12.159 | 30.3976 |
| f_proj1 | 6.2309 | 0.8529 | 2.4867 | 4.5908 | 11.7983 | 0.4999 | 199960 | 9.18334 | 22.9584 |
| f_proj2 | 1.5632 | 7.7772 | 8.4826 | 4.7144 | 11.3766 | 0.4999 | 199960 | 9.43076 | 23.5769 |
| f_pc1 | 1.3524 | 0.6591 | 1.0262 | 9.5639 | 20.9753 | 0.4999 | 199960 | 19.1317 | 47.8292 |
| f_pc2 | 1.0857 | 2.2464 | 3.3733 | 6.0065 | 11.3099 | 0.4999 | 199960 | 12.0154 | 30.0385 |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.7451 | 2.4320 | 3.6760 | 4.8523 | 14.0531 | 0.2499 | 99960 | 19.4171 | 48.5427 |
| f_margin | 1.2562 | 1.5494 | 3.2331 | 6.8920 | 20.8727 | 0.2499 | 99960 | 27.579 | 68.9475 |
| f_dist | 1.0001 | 3.4380 | 3.4421 | 6.0793 | 60.7599 | 0.2499 | 99960 | 24.3271 | 60.8178 |
| dist_to_ref_sq_over_d | 1.0001 | 3.4380 | 3.4421 | 6.0793 | 60.7599 | 0.2499 | 99960 | 24.3271 | 60.8178 |
| dist_to_ref_over_sqrt_d | 1.0001 | 3.6102 | 3.6148 | 6.0793 | 60.7599 | 0.2499 | 99960 | 24.3271 | 60.8178 |
| dist_to_ref_over_ou_radius | 1.0001 | 3.6102 | 3.6148 | 6.0793 | 60.7599 | 0.2499 | 99960 | 24.3271 | 60.8178 |
| f_proj1 | 7.4423 | 2.8186 | 3.3135 | 4.2637 | 11.3895 | 0.2499 | 99960 | 17.0617 | 42.6543 |
| f_proj2 | 6.0936 | 2.1447 | 2.2013 | 4.2582 | 11.3895 | 0.2499 | 99960 | 17.0397 | 42.5991 |
| f_pc1 | 1.3169 | 1.5672 | 2.1139 | 6.2942 | 12.3170 | 0.2499 | 99960 | 25.1867 | 62.9668 |
| f_pc2 | 1.3468 | 2.4704 | 5.6275 | 6.4605 | 12.0346 | 0.2499 | 99960 | 25.8524 | 64.6309 |
