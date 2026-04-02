# Event-aligned window diagnostics
- Window A: `post_geom_d0p05` start at `τ_geom(0.05)`
- Window B: `post_pred_nll1p45_given_geom_d0p05` start at `τ_NLL|geom(0.05, 1.45)`
- Window length (saves): `0` (<=0 means full suffix)

## Group `w2_n512_h5e-06_T100000_a0.3_b1p0_g3p0_ul_initI2_step2475`
| window_kind | n_chains_used | step_first_mean | step_last_mean | R̂ | drift_z_mean | drift_z_max | ESS_bulk | ESS_tail | T_analysis | grad_evals_span | ESS/T_analysis | ESS/(1e6 grad) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `post_geom_d0p05` | 4 | 10285.0 | 99985.0 | 1.0325 | 3.761 | 6.096 | nan | nan | 0.4485 | 1.794e+05 | nan | nan |
| `post_pred_nll1p45_given_geom_d0p05` | 4 | 14950.0 | 99450.0 | 1.1448 | 1.508 | 1.868 | nan | nan | 0.4225 | 1.69e+05 | nan | nan |
