# Event-aligned window diagnostics
- Window A: `post_geom_d0p05` start at `τ_geom(0.05)`
- Window B: `post_pred_nll1p45_given_geom_d0p05` start at `τ_NLL|geom(0.05, 1.45)`
- Window length (saves): `0` (<=0 means full suffix)

## Group `w4_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step1600`
| window_kind | n_chains_used | step_first_mean | step_last_mean | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_geom_d0p05` | 4 | 10275.0 | 99995.0 | 1.0633 | 4.581 | 4.975 | 5.499 | 11.07 | 0.4486 | 1.794e+05 | 12.26 | 30.64 |
| `post_pred_nll1p45_given_geom_d0p05` | 4 | 10275.0 | 99995.0 | 1.0633 | 4.581 | 4.975 | 5.499 | 11.07 | 0.4486 | 1.794e+05 | 12.26 | 30.64 |
