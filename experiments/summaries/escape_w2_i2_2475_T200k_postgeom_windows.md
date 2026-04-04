# Event-aligned window diagnostics
- Window A: `post_geom_d0p05` start at `τ_geom(0.05)`
- Window B: `post_pred_nll1p45_given_geom_d0p05` start at `τ_NLL|geom(0.05, 1.45)`
- Window length (saves): `0` (<=0 means full suffix)
- **Suffix scan** (last fraction of each aligned window): 80%, 60%, 50%, 40%, 30%, 25%; stabilization pick: first suffix with R̂≤1.1 and max drift_z≤0.5 (largest listed fraction tried first).

## Group `w2_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2475`
| window_kind | n_chains_used | step_first_mean | step_last_mean | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_geom_d0p05` | 4 | 10285.0 | 199985.0 | 1.0757 | 1.649 | 2.77 | 7.897 | 39.37 | 0.9485 | 3.794e+05 | 8.325 | 20.81 |

#### `post_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 1897 | 7589 | 1.6306 | 1.678 | 2.747 | 5.245 | 15.07 | 0.7588 | 3.035e+05 | 6.912 | no |
| 0.60 | 3794 | 5692 | 1.7924 | 1.738 | 4.527 | 5.016 | 11.94 | 0.5691 | 2.276e+05 | 8.813 | no |
| 0.50 | 4743 | 4743 | 2.1417 | 1.207 | 3.427 | 5.208 | 26.37 | 0.4742 | 1.897e+05 | 10.98 | no |
| 0.40 | 5691 | 3795 | 2.5020 | 0.4843 | 1.366 | 5.222 | 33.95 | 0.3794 | 1.518e+05 | 13.76 | no |
| 0.30 | 6640 | 2846 | 2.6412 | 0.7616 | 1.303 | 4.996 | 20.9 | 0.2845 | 1.138e+05 | 17.56 | no |
| 0.25 | 7114 | 2372 | 2.4886 | 1.594 | 2.731 | 4.802 | 13.24 | 0.2371 | 9.484e+04 | 20.25 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
| `post_pred_nll1p45_given_geom_d0p05` | 4 | 14950.0 | 199450.0 | 1.2855 | 1.283 | 1.808 | 7.321 | 41.28 | 0.9225 | 3.69e+05 | 7.936 | 19.84 |

#### `post_pred_nll1p45_given_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 1845 | 7381 | 1.6214 | 1.799 | 3.059 | 5.187 | 14.51 | 0.738 | 2.952e+05 | 7.029 | no |
| 0.60 | 3690 | 5536 | 1.8129 | 1.683 | 4.395 | 5.044 | 12.34 | 0.5535 | 2.214e+05 | 9.113 | no |
| 0.50 | 4613 | 4613 | 2.1704 | 1.143 | 3.13 | 5.251 | 26.94 | 0.4612 | 1.845e+05 | 11.39 | no |
| 0.40 | 5535 | 3691 | 2.5697 | 0.4673 | 1.163 | 5.191 | 39.67 | 0.369 | 1.476e+05 | 14.07 | no |
| 0.30 | 6458 | 2768 | 2.6498 | 0.8106 | 1.262 | 4.992 | 22.78 | 0.2767 | 1.107e+05 | 18.04 | no |
| 0.25 | 6919 | 2307 | 2.4979 | 1.683 | 2.873 | 4.803 | 15.11 | 0.2306 | 9.224e+04 | 20.83 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
