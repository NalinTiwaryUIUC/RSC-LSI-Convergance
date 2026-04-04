# Event-aligned window diagnostics
- Window A: `post_geom_d0p05` start at `τ_geom(0.05)`
- Window B: `post_pred_nll1p45_given_geom_d0p05` start at `τ_NLL|geom(0.05, 1.45)`
- Window length (saves): `0` (<=0 means full suffix)
- **Suffix scan** (last fraction of each aligned window): 80%, 60%, 50%, 40%, 30%, 25%; stabilization pick: first suffix with R̂≤1.1 and max drift_z≤0.5 (largest listed fraction tried first).

## Group `w4_n512_h5e-06_T200000_a0.3_b1p0_g3p0_ul_initI2_step1600`
| window_kind | n_chains_used | step_first_mean | step_last_mean | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_geom_d0p05` | 4 | 10275.0 | 199995.0 | 1.0261 | 4.977 | 6.039 | 5.891 | 47.17 | 0.9486 | 3.794e+05 | 6.211 | 15.53 |

#### `post_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 1897 | 7590 | 1.0710 | 4.083 | 4.869 | 5.733 | 19.69 | 0.7589 | 3.036e+05 | 7.555 | no |
| 0.60 | 3794 | 5693 | 1.1477 | 4.529 | 9.079 | 5.873 | 25.99 | 0.5692 | 2.277e+05 | 10.32 | no |
| 0.50 | 4743 | 4744 | 1.2505 | 4.643 | 10.7 | 5.939 | 20.57 | 0.4743 | 1.897e+05 | 12.52 | no |
| 0.40 | 5692 | 3795 | 1.3807 | 2.914 | 5.754 | 6.493 | 11.53 | 0.3794 | 1.518e+05 | 17.11 | no |
| 0.30 | 6640 | 2847 | 1.5292 | 1.162 | 2.747 | 6.958 | 22.18 | 0.2846 | 1.138e+05 | 24.45 | no |
| 0.25 | 7115 | 2372 | 1.5471 | 0.6448 | 1.759 | 8.061 | 44.96 | 0.2371 | 9.484e+04 | 34 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
| `post_pred_nll1p45_given_geom_d0p05` | 4 | 10275.0 | 199995.0 | 1.0261 | 4.977 | 6.039 | 5.891 | 47.17 | 0.9486 | 3.794e+05 | 6.211 | 15.53 |

#### `post_pred_nll1p45_given_geom_d0p05` — tail fractions (last *F* of **aligned** window)

| last_frac | start_idx | n_draws | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_phys | grad_span | ESS/T_phys | passes_gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.80 | 1897 | 7590 | 1.0710 | 4.083 | 4.869 | 5.733 | 19.69 | 0.7589 | 3.036e+05 | 7.555 | no |
| 0.60 | 3794 | 5693 | 1.1477 | 4.529 | 9.079 | 5.873 | 25.99 | 0.5692 | 2.277e+05 | 10.32 | no |
| 0.50 | 4743 | 4744 | 1.2505 | 4.643 | 10.7 | 5.939 | 20.57 | 0.4743 | 1.897e+05 | 12.52 | no |
| 0.40 | 5692 | 3795 | 1.3807 | 2.914 | 5.754 | 6.493 | 11.53 | 0.3794 | 1.518e+05 | 17.11 | no |
| 0.30 | 6640 | 2847 | 1.5292 | 1.162 | 2.747 | 6.958 | 22.18 | 0.2846 | 1.138e+05 | 24.45 | no |
| 0.25 | 7115 | 2372 | 1.5471 | 0.6448 | 1.759 | 8.061 | 44.96 | 0.2371 | 9.484e+04 | 34 | no |

*Stabilized pick:* **none** on this grid / gates (R̂≤1.1, max drift_z≤0.5).
