# Chain convergence and diagnostics report

Runs discovered: **4** in `experiments/runs` (glob `w4_*_T200000_*ul_initI2_step1600_chain*`)

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
| `w4_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain0` | 0 |
| `w4_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain1` | 1 |
| `w4_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain2` | 2 |
| `w4_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step1600_chain3` | 3 |

### Convergence from `samples_metrics.npz`

| probe | R̂ | ESS mean | ESS min | n samples / chain | ESS rate (×1e6 grads⁻¹) | mean 1st half | mean 2nd half | Δ(2nd−1st) |
|-------|---|----------|---------|-------------------|-------------------------|---------------|---------------|------------|
| f_nll | 1.0035 | 31.2 | 28.5 | 10000 | 77.9626 | 866.645 | 846.91 | -19.7343 |
| f_margin | 1.0037 | 37.4 | 32.9 | 10000 | 93.5920 | -0.294992 | -0.339744 | -0.0447515 |
| f_dist | 1.0000 | 5.8 | 5.8 | 10000 | 14.5171 | 69942.5 | 347301 | 277358 |

*Approx. post-burn grad evals per chain: **400000***

### `iter_metrics.jsonl` — primary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| f_nll | 800000 | 856.751 | 68.8218 | 865.03 | 863.484 | 842.182 | 866.59 | 846.913 | -19.6766 |
| f_margin | 800000 | -0.317283 | 0.181392 | -0.270028 | -0.340976 | -0.340151 | -0.294822 | -0.339744 | -0.0449221 |
| ce_mean_train | 800000 | 1.67259 | 0.134413 | 1.68907 | 1.68564 | 1.64392 | 1.692 | 1.65318 | -0.0388204 |
| margin_probe | 800000 | -0.317283 | 0.181392 | -0.270028 | -0.340976 | -0.340151 | -0.294822 | -0.339744 | -0.0449221 |
| pmax_mean | 800000 | 0.360696 | 0.0413669 | 0.329836 | 0.357356 | 0.393892 | 0.336435 | 0.384958 | 0.0485225 |
| U_train | 800000 | 32299.7 | 23866.2 | 6131.64 | 29007.5 | 60893.4 | 11511.7 | 53087.7 | 41576 |
| grad_norm | 800000 | 795.347 | 938.162 | 1189.89 | 550.783 | 649.779 | 960.066 | 630.627 | -329.439 |
| nll_probe_mean | 800000 | 1.67334 | 0.134418 | 1.68951 | 1.68649 | 1.64489 | 1.69256 | 1.65413 | -0.0384309 |

### `iter_metrics.jsonl` — secondary diagnostics

| key | n | mean | std | early | mid | late | 1st half | 2nd half | Δ(2nd−1st) |
|-----|-----|------|-----|-------|-----|------|----------|----------|------------|
| dist_to_ref | 800000 | 408.436 | 204.394 | 161.884 | 426.927 | 629.791 | 233.251 | 583.622 | 350.372 |
| dist_to_ref_sq_over_d | 800000 | 0.175194 | 0.133697 | 0.0286094 | 0.156719 | 0.335398 | 0.0587273 | 0.29166 | 0.232933 |
| dist_to_ref_over_sqrt_d | 800000 | 0.374308 | 0.187315 | 0.148357 | 0.391254 | 0.577167 | 0.213761 | 0.534856 | 0.321096 |
| dist_to_ref_over_ou_radius | 800000 | 0.205017 | 0.102597 | 0.0812585 | 0.214299 | 0.316127 | 0.117081 | 0.292953 | 0.175871 |
| theta_norm | 800000 | 410.916 | 201.916 | 167.339 | 428.164 | 630.589 | 237.329 | 584.503 | 347.173 |
| v_norm | 800000 | 1081.01 | 4.80171 | 1086.91 | 1080.21 | 1076.05 | 1085.04 | 1076.97 | -8.06838 |
| kinetic_energy | 800000 | 584302 | 5196.67 | 590692 | 583427 | 578947 | 588665 | 579938 | -8726.27 |
| theta_v_cosine | 800000 | 0.632886 | 0.143184 | 0.772019 | 0.638186 | 0.492701 | 0.740647 | 0.525124 | -0.215523 |
| snr | 0 | — | — | — | — | — | — | — | — |
| delta_U | 799996 | 0.387543 | 0.132849 | 0.223862 | 0.437065 | 0.498345 | 0.286985 | 0.488101 | 0.201117 |
| noise_step_norm | 800000 | 5.97473 | 0.00386611 | 5.97473 | 5.97473 | 5.97473 | 5.97472 | 5.97474 | 1.63379e-05 |
| drift_step_norm | 0 | — | — | — | — | — | — | — | — |
| U_prior | 800000 | 31443.3 | 23875 | 5266.83 | 28144.5 | 60051.7 | 10645.4 | 52241.2 | 41595.9 |
| U_data | 800000 | 856.365 | 68.8195 | 864.806 | 863.048 | 841.686 | 866.303 | 846.427 | -19.876 |
| ce_mean_train | 800000 | 1.67259 | 0.134413 | 1.68907 | 1.68564 | 1.64392 | 1.692 | 1.65318 | -0.0388204 |
| U_data_minus_ce | 800000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Quick interpretation

- **abort_suggested** ever: False
- **bad_locality** flags (count): 0
- **U_train** finite records: 800000 / 800000

### Stability gates

- **max U_train** (iter_metrics): 77995.7 — flag if blow-up vs typical scale.

- **max ||θ||** (iter_metrics): 717.212

- **NaNs in last 50% of saved f_nll** (chain0 ref): **False**


### Late-window analytics (stationarity + mixing)

- **f_nll**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).

- **f_margin**: no suffix found with R̂≤1.05 and max drift_z≤0.5 (heuristic).


#### Last **50%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.2103 | 4.7130 | 10.5748 | 6.0021 | 66.6939 | 0.4999 | 199960 | 12.0066 | 30.0165 |
| f_margin | 1.4271 | 0.8926 | 1.0660 | 8.7450 | 17.1991 | 0.4999 | 199960 | 17.4935 | 43.7338 |
| f_dist | 0.9999 | 3.3900 | 3.3907 | 6.0800 | 62.6081 | 0.4999 | 199960 | 12.1623 | 30.4058 |
| dist_to_ref_sq_over_d | 0.9999 | 3.3900 | 3.3907 | 6.0800 | 62.6081 | 0.4999 | 199960 | 12.1623 | 30.4058 |
| dist_to_ref_over_sqrt_d | 0.9999 | 3.8086 | 3.8090 | 6.0800 | 62.6081 | 0.4999 | 199960 | 12.1623 | 30.4058 |
| dist_to_ref_over_ou_radius | 0.9999 | 3.8086 | 3.8090 | 6.0800 | 62.6081 | 0.4999 | 199960 | 12.1623 | 30.4058 |
| f_proj1 | 9.1507 | 1.7409 | 3.1170 | 4.5799 | 18.9347 | 0.4999 | 199960 | 9.16154 | 22.9038 |
| f_proj2 | 9.1512 | 7.1425 | 10.5456 | 4.2751 | 13.4184 | 0.4999 | 199960 | 8.55187 | 21.3797 |
| f_pc1 | 1.6125 | 1.1934 | 1.3769 | 6.0022 | 21.8327 | 0.4999 | 199960 | 12.0068 | 30.0169 |
| f_pc2 | 1.2286 | 2.9587 | 3.9317 | 5.7733 | 24.0609 | 0.4999 | 199960 | 11.549 | 28.8725 |

#### Last **25%** of saved samples per chain

| probe | R̂ (late) | drift_z mean | drift_z max | ESS_bulk | ESS_tail | T_analysis (phys) | grad evals (span) | ESS/T_phys | ESS/(1e6 grad) |
|-------|----------|-------------|-------------|----------|----------|---------------------|-------------------|----------|----------------|
| f_nll | 1.5413 | 0.7748 | 2.1374 | 7.7016 | 45.7158 | 0.2499 | 99960 | 30.8188 | 77.0469 |
| f_margin | 1.5332 | 0.6738 | 0.9184 | 8.5070 | 30.2252 | 0.2499 | 99960 | 34.0416 | 85.1041 |
| f_dist | 0.9998 | 3.4530 | 3.4536 | 6.0877 | 62.7254 | 0.2499 | 99960 | 24.3605 | 60.9011 |
| dist_to_ref_sq_over_d | 0.9998 | 3.4530 | 3.4536 | 6.0877 | 62.7254 | 0.2499 | 99960 | 24.3605 | 60.9011 |
| dist_to_ref_over_sqrt_d | 0.9998 | 3.6236 | 3.6244 | 6.0877 | 62.7254 | 0.2499 | 99960 | 24.3605 | 60.9011 |
| dist_to_ref_over_ou_radius | 0.9998 | 3.6236 | 3.6244 | 6.0877 | 62.7254 | 0.2499 | 99960 | 24.3605 | 60.9011 |
| f_proj1 | 18.9945 | 0.8372 | 1.5232 | 4.4720 | 11.3895 | 0.2499 | 99960 | 17.8952 | 44.7381 |
| f_proj2 | 37.7250 | 3.7864 | 5.6825 | 4.4937 | 13.3165 | 0.2499 | 99960 | 17.982 | 44.9551 |
| f_pc1 | 1.6015 | 1.0382 | 2.2017 | 6.4800 | 15.9934 | 0.2499 | 99960 | 25.9303 | 64.8257 |
| f_pc2 | 1.5683 | 1.0772 | 1.8204 | 6.5421 | 27.6068 | 0.2499 | 99960 | 26.1791 | 65.4476 |
