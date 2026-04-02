# Event-aligned window diagnostics
- Window A: `post_geom_d0p05` start at `τ_geom(0.05)`
- Window B: `post_pred_nll1p45_given_geom_d0p05` start at `τ_NLL|geom(0.05, 1.45)`
- Window length (saves): `0` (<=0 means full suffix)
- **Suffix scan** (last fraction of each aligned window): 80%, 60%, 50%, 40%, 30%, 25%; stabilization pick: first suffix with R̂≤1.1 and max drift_z≤0.5 (largest listed fraction tried first).

## Group `w1_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2000`
| window_kind | n_chains_used | step_first_mean | step_last_mean | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_geom_d0p05` | 4 | 10295.0 | 99995.0 | 1.0115 | 10.06 | 11.89 | 5.981 | 38.64 | 0.4485 | 1.794e+05 | 13.34 | 33.34 |

#### `post_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 897 | 3589 | 1.1975 | 4.767 | 10.79 | 5.982 | 30.06 | 0.3588 | 1.435e+05 | 16.67 | no |
| 0.60 | 1794 | 2692 | 1.4943 | 2.868 | 6.544 | 5.632 | 13.46 | 0.2691 | 1.076e+05 | 20.93 | no |
| 0.50 | 2243 | 2243 | 1.7173 | 2.11 | 4.257 | 5.547 | 11.37 | 0.2242 | 8.968e+04 | 24.74 | no |
| 0.40 | 2691 | 1795 | 2.0534 | 2.384 | 4.501 | 4.903 | 12.46 | 0.1794 | 7.176e+04 | 27.33 | no |
| 0.30 | 3140 | 1346 | 1.9470 | 1.906 | 2.381 | 5.016 | 24.84 | 0.1345 | 5.38e+04 | 37.29 | no |
| 0.25 | 3364 | 1122 | 1.9446 | 1.564 | 3.277 | 4.925 | 18.31 | 0.1121 | 4.484e+04 | 43.94 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
| `post_pred_nll1p45_given_geom_d0p05` | 4 | 52040.0 | 77320.0 | 1.1484 | 3.216 | 6.868 | 6.684 | 11.81 | 0.1264 | 5.056e+04 | 52.88 | 132.2 |

#### `post_pred_nll1p45_given_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 252 | 1013 | 1.2711 | 3.207 | 6.16 | 5.69 | 15.89 | 0.1012 | 4.048e+04 | 56.22 | no |
| 0.60 | 506 | 759 | 1.6156 | 3.743 | 5.852 | 4.875 | 17.38 | 0.0758 | 3.032e+04 | 64.31 | no |
| 0.50 | 632 | 633 | 2.2132 | 3.202 | 5.727 | 4.724 | 12.29 | 0.0632 | 2.528e+04 | 74.74 | no |
| 0.40 | 759 | 506 | 3.6477 | 2.023 | 4.595 | 4.678 | 18.37 | 0.0505 | 2.02e+04 | 92.64 | no |
| 0.30 | 885 | 380 | 5.1670 | 2.361 | 5.432 | 4.589 | 19.68 | 0.0379 | 1.516e+04 | 121.1 | no |
| 0.25 | 948 | 317 | 5.6877 | 3.115 | 5.963 | 4.543 | 15.39 | 0.0316 | 1.264e+04 | 143.8 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
