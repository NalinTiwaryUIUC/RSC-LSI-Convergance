# Event-aligned window diagnostics
- Window A: `post_geom_d0p05` start at `τ_geom(0.05)`
- Window B: `post_pred_nll1p45_given_geom_d0p05` start at `τ_NLL|geom(0.05, 1.45)`
- Window length (saves): `0` (<=0 means full suffix)
- **Suffix scan** (last fraction of each aligned window): 80%, 60%, 50%, 40%, 30%, 25%; stabilization pick: first suffix with R̂≤1.1 and max drift_z≤0.5 (largest listed fraction tried first).

## Group `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600`
| window_kind | n_chains_used | step_first_mean | step_last_mean | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_geom_d0p05` | 4 | 10275.0 | 99995.0 | 1.0633 | 4.581 | 4.975 | 5.499 | 11.07 | 0.4486 | 1.794e+05 | 12.26 | 30.64 |

#### `post_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 897 | 3590 | 1.1257 | 4.547 | 7.43 | 5.252 | 11.02 | 0.3589 | 1.436e+05 | 14.63 | no |
| 0.60 | 1794 | 2693 | 1.3515 | 3.496 | 5.774 | 5.178 | 10.76 | 0.2692 | 1.077e+05 | 19.23 | no |
| 0.50 | 2243 | 2244 | 1.5527 | 3.461 | 4.464 | 4.937 | 11.1 | 0.2243 | 8.972e+04 | 22.01 | no |
| 0.40 | 2692 | 1795 | 1.9130 | 2.952 | 3.422 | 4.661 | 11.92 | 0.1794 | 7.176e+04 | 25.98 | no |
| 0.30 | 3140 | 1347 | 2.3109 | 1.893 | 2.345 | 4.825 | 11.37 | 0.1346 | 5.384e+04 | 35.85 | no |
| 0.25 | 3365 | 1122 | 2.8655 | 1.112 | 1.503 | 4.832 | 17.19 | 0.1121 | 4.484e+04 | 43.11 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
| `post_pred_nll1p45_given_geom_d0p05` | 4 | 10275.0 | 99995.0 | 1.0633 | 4.581 | 4.975 | 5.499 | 11.07 | 0.4486 | 1.794e+05 | 12.26 | 30.64 |

#### `post_pred_nll1p45_given_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 897 | 3590 | 1.1257 | 4.547 | 7.43 | 5.252 | 11.02 | 0.3589 | 1.436e+05 | 14.63 | no |
| 0.60 | 1794 | 2693 | 1.3515 | 3.496 | 5.774 | 5.178 | 10.76 | 0.2692 | 1.077e+05 | 19.23 | no |
| 0.50 | 2243 | 2244 | 1.5527 | 3.461 | 4.464 | 4.937 | 11.1 | 0.2243 | 8.972e+04 | 22.01 | no |
| 0.40 | 2692 | 1795 | 1.9130 | 2.952 | 3.422 | 4.661 | 11.92 | 0.1794 | 7.176e+04 | 25.98 | no |
| 0.30 | 3140 | 1347 | 2.3109 | 1.893 | 2.345 | 4.825 | 11.37 | 0.1346 | 5.384e+04 | 35.85 | no |
| 0.25 | 3365 | 1122 | 2.8655 | 1.112 | 1.503 | 4.832 | 17.19 | 0.1121 | 4.484e+04 | 43.11 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
