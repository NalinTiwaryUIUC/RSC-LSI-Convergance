# Event-aligned window diagnostics
- Window A: `post_geom_d0p05` start at `τ_geom(0.05)`
- Window B: `post_pred_nll1p45_given_geom_d0p05` start at `τ_NLL|geom(0.05, 1.45)`
- Window length (saves): `0` (<=0 means full suffix)
- **Suffix scan** (last fraction of each aligned window): 80%, 60%, 50%, 40%, 30%, 25%; stabilization pick: first suffix with R̂≤1.1 and max drift_z≤0.5 (largest listed fraction tried first).

## Group `w1_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step2000`
| window_kind | n_chains_used | step_first_mean | step_last_mean | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_geom_d0p05` | 4 | 10295.0 | 199995.0 | 1.0194 | 5.977 | 6.579 | 6.908 | 17.23 | 0.9485 | 3.794e+05 | 7.283 | 18.21 |

#### `post_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 1897 | 7589 | 1.5095 | 1.315 | 1.476 | 6.33 | 14.48 | 0.7588 | 3.035e+05 | 8.342 | no |
| 0.60 | 3794 | 5692 | 1.7433 | 1.055 | 1.842 | 5.726 | 11.48 | 0.5691 | 2.276e+05 | 10.06 | no |
| 0.50 | 4743 | 4743 | 1.8030 | 1.221 | 2.35 | 5.406 | 11.25 | 0.4742 | 1.897e+05 | 11.4 | no |
| 0.40 | 5691 | 3795 | 1.7136 | 1.363 | 2.147 | 5.35 | 11.51 | 0.3794 | 1.518e+05 | 14.1 | no |
| 0.30 | 6640 | 2846 | 1.7049 | 1.919 | 2.932 | 5.033 | 11.61 | 0.2845 | 1.138e+05 | 17.69 | no |
| 0.25 | 7114 | 2372 | 1.7714 | 2.645 | 4.509 | 4.812 | 17.05 | 0.2371 | 9.484e+04 | 20.3 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
| `post_pred_nll1p45_given_geom_d0p05` | 4 | 52040.0 | 177320.0 | 1.3708 | 3.002 | 5.874 | 6.119 | 18.97 | 0.6264 | 2.506e+05 | 9.768 | 24.42 |

#### `post_pred_nll1p45_given_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 1252 | 5013 | 1.6119 | 2.077 | 4.637 | 5.949 | 24.89 | 0.5012 | 2.005e+05 | 11.87 | no |
| 0.60 | 2506 | 3759 | 1.8409 | 1.56 | 2.289 | 5.547 | 10.71 | 0.3758 | 1.503e+05 | 14.76 | no |
| 0.50 | 3132 | 3133 | 1.9824 | 1.122 | 1.854 | 5.624 | 17.87 | 0.3132 | 1.253e+05 | 17.96 | no |
| 0.40 | 3759 | 2506 | 1.9614 | 0.7662 | 1.318 | 6.263 | 15.19 | 0.2505 | 1.002e+05 | 25 | no |
| 0.30 | 4385 | 1880 | 1.9400 | 1 | 1.69 | 5.92 | 38.67 | 0.1879 | 7.516e+04 | 31.51 | no |
| 0.25 | 4698 | 1567 | 1.8854 | 1.169 | 2.187 | 5.702 | 55.47 | 0.1566 | 6.264e+04 | 36.41 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
