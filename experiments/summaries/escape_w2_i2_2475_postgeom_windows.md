# Event-aligned window diagnostics
- Window A: `post_geom_d0p05` start at `τ_geom(0.05)`
- Window B: `post_pred_nll1p45_given_geom_d0p05` start at `τ_NLL|geom(0.05, 1.45)`
- Window length (saves): `0` (<=0 means full suffix)
- **Suffix scan** (last fraction of each aligned window): 80%, 60%, 50%, 40%, 30%, 25%; stabilization pick: first suffix with R̂≤1.1 and max drift_z≤0.5 (largest listed fraction tried first).

## Group `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475`
| window_kind | n_chains_used | step_first_mean | step_last_mean | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_geom_d0p05` | 4 | 10285.0 | 99985.0 | 1.0325 | 3.761 | 6.096 | 15.18 | 15.32 | 0.4485 | 1.794e+05 | 33.85 | 84.62 |

#### `post_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 897 | 3589 | 1.5626 | 2.362 | 4.382 | 5.611 | 11.43 | 0.3588 | 1.435e+05 | 15.64 | no |
| 0.60 | 1794 | 2692 | 1.9361 | 2.154 | 3.195 | 5.027 | 13.1 | 0.2691 | 1.076e+05 | 18.68 | no |
| 0.50 | 2243 | 2243 | 2.3450 | 2.335 | 3.294 | 5.267 | 13.64 | 0.2242 | 8.968e+04 | 23.49 | no |
| 0.40 | 2691 | 1795 | 2.8693 | 1.599 | 2.139 | 5.386 | 15.95 | 0.1794 | 7.176e+04 | 30.02 | no |
| 0.30 | 3140 | 1346 | 3.7365 | 0.7891 | 1.692 | 5.54 | 25.53 | 0.1345 | 5.38e+04 | 41.19 | no |
| 0.25 | 3364 | 1122 | 4.3897 | 0.312 | 0.5088 | 5.358 | 23.05 | 0.1121 | 4.484e+04 | 47.8 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
| `post_pred_nll1p45_given_geom_d0p05` | 4 | 14950.0 | 99450.0 | 1.1448 | 1.508 | 1.868 | 11.49 | 14.73 | 0.4225 | 1.69e+05 | 27.2 | 67.99 |

#### `post_pred_nll1p45_given_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 845 | 3381 | 1.5772 | 2.431 | 4.57 | 5.623 | 11.73 | 0.338 | 1.352e+05 | 16.64 | no |
| 0.60 | 1690 | 2536 | 2.0580 | 2.614 | 3.38 | 4.889 | 12.18 | 0.2535 | 1.014e+05 | 19.29 | no |
| 0.50 | 2113 | 2113 | 2.5152 | 2.403 | 3.131 | 5.159 | 13.17 | 0.2112 | 8.448e+04 | 24.43 | no |
| 0.40 | 2535 | 1691 | 2.9962 | 1.605 | 1.855 | 5.358 | 20.8 | 0.169 | 6.76e+04 | 31.7 | no |
| 0.30 | 2958 | 1268 | 4.1519 | 0.5063 | 0.8302 | 5.406 | 25.62 | 0.1267 | 5.068e+04 | 42.67 | no |
| 0.25 | 3169 | 1057 | 4.6437 | 0.4088 | 0.6711 | 5.528 | 23.6 | 0.1056 | 4.224e+04 | 52.35 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
