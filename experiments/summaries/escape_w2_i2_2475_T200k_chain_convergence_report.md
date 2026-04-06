# Chain convergence and diagnostics report

Runs discovered: **4** in `experiments/runs` (glob `w2_*_T200000_*ul_initI2_step2475_chain*`)

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
| `w2_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain0` | 0 |
| `w2_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain1` | 1 |
| `w2_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain2` | 2 |
| `w2_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2475_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0027 | 17.2 | 16.8 | 10000 | 42.9458 | 758.248 | 809.423 | 51.1754 |
| f_margin | 1.0010 | 17.9 | 17.4 | 10000 | 44.8311 | -0.0943916 | -0.295109 | -0.200717 |
| f_dist | 1.0000 | 5.8 | 5.8 | 10000 | 14.5164 | 17590.7 | 87452.1 | 69861.4 |

*Approx. post-burn grad evals per chain: **400000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 800000 | 783.802 | 102.163 | 733.977 | 806.062 | 810.555 | 758.181 | 809.422 | 51.2416 |
| f_margin | 800000 | -0.194615 | 0.334919 | -0.00704163 | -0.274192 | -0.299435 | -0.0941263 | -0.295104 | -0.200978 |
| ce_mean_train | 800000 | 1.53067 | 0.199519 | 1.43342 | 1.57412 | 1.58287 | 1.48067 | 1.58066 | 0.0999938 |
| margin_probe | 800000 | -0.194615 | 0.334919 | -0.00704163 | -0.274192 | -0.299435 | -0.0941263 | -0.295104 | -0.200978 |
| pmax_mean | 800000 | 0.383769 | 0.0403556 | 0.374191 | 0.380165 | 0.396564 | 0.372614 | 0.394924 | 0.0223097 |
| U_train | 800000 | 8759.58 | 6050.48 | 2112.69 | 7951.37 | 15995.4 | 3491.95 | 14027.2 | 10535.3 |
| grad_norm | 800000 | 538.242 | 319.329 | 712.499 | 438.191 | 466.22 | 613.616 | 462.868 | -150.748 |
| nll_probe_mean | 800000 | 1.53086 | 0.199538 | 1.43355 | 1.57434 | 1.58312 | 1.48082 | 1.5809 | 0.100081 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 800000 | 204.919 | 102.584 | 81.1778 | 214.181 | 316.031 | 116.974 | 292.864 | 175.889 |
| dist_to_ref_sq_over_d | 800000 | 0.174803 | 0.133443 | 0.0285079 | 0.156339 | 0.334715 | 0.0585375 | 0.291068 | 0.23253 |
| dist_to_ref_over_sqrt_d | 800000 | 0.373864 | 0.187159 | 0.148105 | 0.390762 | 0.576582 | 0.213414 | 0.534314 | 0.320901 |
| dist_to_ref_over_ou_radius | 800000 | 0.204774 | 0.102511 | 0.0811202 | 0.214029 | 0.315807 | 0.116891 | 0.292656 | 0.175765 |
| theta_norm | 800000 | 207.686 | 100.194 | 86.8662 | 215.771 | 317.107 | 121.336 | 294.037 | 172.702 |
| v_norm | 800000 | 542.764 | 2.19375 | 545.303 | 542.728 | 540.334 | 544.577 | 540.95 | -3.62782 |
| kinetic_energy | 800000 | 147299 | 1191.64 | 148678 | 147277 | 145981 | 148283 | 146314 | -1969.59 |
| theta_v_cosine | 800000 | 0.618313 | 0.13733 | 0.731924 | 0.635718 | 0.491151 | 0.713033 | 0.523593 | -0.189441 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 799996 | 0.100557 | 0.0303349 | 0.0649431 | 0.110334 | 0.125635 | 0.0779646 | 0.12315 | 0.0451855 |
| noise_step_norm | 800000 | 3.00118 | 0.00386688 | 3.00118 | 3.00119 | 3.00118 | 3.00117 | 3.00119 | 1.97251e-05 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 800000 | 7975.88 | 6016.39 | 1378.78 | 7145.42 | 15185 | 2733.85 | 13217.9 | 10484.1 |
| U_data | 800000 | 783.701 | 102.154 | 733.912 | 805.952 | 810.43 | 758.103 | 809.3 | 51.1968 |
| ce_mean_train | 800000 | 1.53067 | 0.199519 | 1.43342 | 1.57412 | 1.58287 | 1.48067 | 1.58066 | 0.0999938 |
| U_data_minus_ce | 800000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 800000 / 800000

### Stability gates

- **max U_train** (iter_metrics): 20340.4 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 360.85

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 2.0471 | 1.3444 | 4.0365 | 5.1158 | 20.0974 | 0.4999 | 199960 | 10.2336 | 25.584 |
| f_margin | 1.5762 | 0.7520 | 1.4889 | 6.6237 | 26.9171 | 0.4999 | 199960 | 13.25 | 33.125 |
| f_dist | 0.9999 | 3.3892 | 3.3904 | 6.0793 | 62.4594 | 0.4999 | 199960 | 12.1611 | 30.4027 |
| dist_to_ref_sq_over_d | 0.9999 | 3.3892 | 3.3904 | 6.0793 | 62.4594 | 0.4999 | 199960 | 12.1611 | 30.4027 |
| dist_to_ref_over_sqrt_d | 0.9999 | 3.8078 | 3.8092 | 6.0793 | 62.4594 | 0.4999 | 199960 | 12.1611 | 30.4027 |
| dist_to_ref_over_ou_radius | 0.9999 | 3.8078 | 3.8092 | 6.0793 | 62.4594 | 0.4999 | 199960 | 12.1611 | 30.4027 |
| f_proj1 | 2.2982 | 1.3566 | 1.6691 | 4.8558 | 16.2748 | 0.4999 | 199960 | 9.71361 | 24.284 |
| f_proj2 | 2.1788 | 2.5969 | 2.6490 | 4.4338 | 11.3766 | 0.4999 | 199960 | 8.86941 | 22.1735 |
| f_pc1 | 1.5020 | 1.7151 | 3.1315 | 5.5047 | 17.8236 | 0.4999 | 199960 | 11.0116 | 27.5291 |
| f_pc2 | 1.3740 | 3.7973 | 6.9001 | 5.3397 | 22.1605 | 0.4999 | 199960 | 10.6815 | 26.7038 |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 2.5471 | 1.3729 | 2.3871 | 4.8557 | 17.7950 | 0.2499 | 99960 | 19.4306 | 48.5766 |
| f_margin | 1.6112 | 2.1160 | 4.8025 | 5.4019 | 14.7869 | 0.2499 | 99960 | 21.6163 | 54.0407 |
| f_dist | 0.9999 | 3.4463 | 3.4476 | 6.0846 | 62.0459 | 0.2499 | 99960 | 24.3482 | 60.8705 |
| dist_to_ref_sq_over_d | 0.9999 | 3.4463 | 3.4476 | 6.0846 | 62.0459 | 0.2499 | 99960 | 24.3482 | 60.8705 |
| dist_to_ref_over_sqrt_d | 0.9999 | 3.6165 | 3.6181 | 6.0846 | 62.0459 | 0.2499 | 99960 | 24.3482 | 60.8705 |
| dist_to_ref_over_ou_radius | 0.9999 | 3.6165 | 3.6181 | 6.0846 | 62.0459 | 0.2499 | 99960 | 24.3482 | 60.8705 |
| f_proj1 | 2.5497 | 3.9579 | 4.2462 | 4.4489 | 11.3895 | 0.2499 | 99960 | 17.8026 | 44.5064 |
| f_proj2 | 3.1162 | 3.0709 | 3.3437 | 4.3117 | 11.3895 | 0.2499 | 99960 | 17.2538 | 43.1345 |
| f_pc1 | 2.1855 | 0.6644 | 1.1556 | 5.8979 | 27.6994 | 0.2499 | 99960 | 23.6012 | 59.0031 |
| f_pc2 | 4.1117 | 2.3041 | 3.6724 | 4.6702 | 14.1429 | 0.2499 | 99960 | 18.6884 | 46.721 |
