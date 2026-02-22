# Experiment Summary: ULA Sampling for Bayesian Posterior over Neural Nets on CIFAR-10

A comprehensive summary of our setup, metrics, diagnostics, and known issues. Intended for external reviewers or collaborators who need to understand the codebase without reading the code.

---

## 1. What We Are Trying to Show

Empirically test:

1. **Convergence vs width**: Whether convergence efficiency (ESS, ESS-rate, R̂) degrades with network width.
2. **Proxy LSI constant**: Whether a dimension-free proxy of the log-Sobolev inequality constant stabilizes beyond a width threshold.
3. **Discretization sanity**: Whether results are stable under step-size halving (ULA discretization check).

The proxy LSI is defined as:
\[
\widehat{\rho}_f = \frac{\widehat{\mathbb{E}}\|\nabla_\theta f(\theta)\|^2}{\widehat{\mathrm{Var}}(f(\theta))}
\]

We run ULA (Unadjusted Langevin Algorithm) to sample from the posterior \(p(\theta \mid \mathcal{D}) \propto \exp(-U(\theta))\).

---

## 2. Model Makeup

- **Architecture**: ResNet-18 for CIFAR-10 (CIFAR-native variant).
  - 3×3 conv stem, stride 1, padding 1 (no 7×7 conv, no maxpool).
  - 4 stages, 2 residual blocks each.
  - Global average pooling, linear head to 10 classes.
  - Stage widths: `[base, 2*base, 4*base, 8*base]` with `base = int(64 * width_multiplier)`.
- **Width multipliers**: Typically `w ∈ {0.5, 1, 2, 4}`; also `0.1`, `0.01` for narrow sweeps.
- **Components**: Uses **BatchNorm2d** in all conv blocks. **No dropout** anywhere.
- **Parameter count** \(d\) is logged per run (e.g. ~100k for width 0.1).

---

## 3. Dataset and Data Usage

### 3.1 Data Subsets

We use a **subset of CIFAR-10**, not the full 50k training set. The default and typical choice is **1024 training samples**.

| Subset | Source | Size | Indices | Persisted |
|--------|--------|------|---------|-----------|
| **Train subset** | CIFAR-10 train split | \(n\) (default 1024; e.g. 512, 1024, 2048) | First \(n\) of seeded permutation of [0..49999] | `train_subset_indices.json` keyed by \(n\) |
| **Probe set** | CIFAR-10 test split | `probe_size` (e.g. 512) | First probe_size of seeded permutation of [0..9999] | `probe_indices.json` keyed by probe_size |

- **Seeds**: Train subset uses `dataset_seed` (default 42). Probe set uses `dataset_seed + 1` (e.g. 43).
- **Indices**: Generated once per (n, seed) or (probe_size, seed), cached in `experiments/data/` so all runs share the same subsets.

### 3.2 Batch and Loading

| Loader | Subset | Batch size | Shuffle |
|--------|--------|------------|---------|
| **Train loader** | Train subset | **Full-batch** (batch_size = n_train) | No |
| **Probe loader** | Probe set | **Full-batch** (batch_size = probe_size) | No |

- **Full-batch everywhere**: Both loaders yield a single batch each; no minibatching.
- **Logged in run_config.yaml**: `effective_batch_size`, `num_microbatches`, `microbatch_size` (prevents "we thought full-batch, but actually microbatch" confusion).
- **Pre-loading**: At chain start, `(x_train, y_train)` and `(x_probe, y_probe)` are loaded to device once. All ULA steps and probes use these same tensors.

### 3.3 Transforms and Augmentations

| Transform | Pipeline | Augmentation |
|-----------|----------|--------------|
| **TRAIN_TRANSFORM** | RandomCrop(32, padding=4), RandomHorizontalFlip(0.5), ToTensor, Normalize | Yes (random crop, flip) |
| **TEST_TRANSFORM** | ToTensor, Normalize | No (deterministic) |

- **Train loader**: Uses **TEST_TRANSFORM** when `eval_transform=True` (default for pretrain and chain). No augmentation.
- **Probe loader**: Always uses **TEST_TRANSFORM** (test split, no augmentation).
- **Normalization** (both): mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010).

**Critical**: Both pretrain and ULA chain use `eval_transform=True` so they see **identical, deterministic training inputs**. This fixed a mismatch where pretrain used TRAIN_TRANSFORM (augmentation) but chain used TEST_TRANSFORM.

### 3.4 Where Each Subset Is Used

| Usage | Subset | Data | Purpose |
|-------|--------|------|---------|
| **Potential U** | Train | (x_train, y_train) | Full-batch CE + prior |
| **ULA step** | Train | (x_train, y_train) | \(\nabla U\) |
| **f_nll** | Train | (x_train, y_train) | CE with same reduction as U |
| **f_margin** | Train | (x_train, y_train) | Mean margin on train batch |
| **f_pc1, f_pc2** | Probe | (x_probe, y_probe) | Random projection of flattened logits |
| **f_proj1, f_proj2** | — | \(\theta - \theta_0\) | Param projection (no data) |
| **f_dist** | — | \(\theta - \theta_0\) | \(\|\theta - \theta_0\|^2\) (no data) |
| **nll_probe** (diagnostic) | Train | (x_train, y_train) | CE on train batch (must match U_data) |
| **margin_probe** (diagnostic) | Train | (x_train, y_train) | Mean margin on train batch |
| **logit_max_abs, pmax_mean, logits_finite** (diagnostic) | Probe | (x_probe, y_probe) | Logit scale on probe set |

**Summary**: U, f_nll, f_margin, nll_probe, margin_probe, and grad_norm_sq for f_nll/f_margin all use the **train batch**. f_pc1, f_pc2, and logit-scale diagnostics (logit_max_abs, pmax_mean, etc.) use the **probe set**. f_proj1, f_proj2, f_dist use only \(\theta\) (no data).

---

## 4. Target Distribution (Potential \(U\))

\[
U(\theta) = \text{CE}(\theta; \mathcal{D}_{\text{train}}) + \frac{\alpha}{2}\|\theta\|^2
\]

- **CE**: Cross-entropy on the full training batch. Reduction is configurable: **"mean"** or **"sum"** (see §7).
- **Prior**: L2 Gaussian prior with precision \(\alpha\) (default 0.01; we also use 0.1, 0.3, 1.0, 3.0 for alpha sweeps).
- **Temperature**: 1 (absorbed into ULA step).

---

## 5. ULA Algorithm

\[
\theta_{t+1} = \theta_t - h\,\nabla U(\theta_t) + \sqrt{2h}\,\sigma\,\xi_t,\quad \xi_t \sim \mathcal{N}(0, I)
\]

- **Step size** \(h\): configurable; typical range 1e-9 to 1e-4.
- **Noise scale** \(\sigma\): default 1.0 (standard ULA); adjustable via `--noise-scale`.
- **Full-batch**: One full forward/backward on the entire training subset each step.

---

## 6. Metrics (What We Measure and How)

### 6.1 Per-Iteration Metrics (iter_metrics.jsonl)

| Metric | Formula / Definition | Purpose |
|--------|----------------------|---------|
| `U_train` | \(U(\theta)\) from potential | Negative log-posterior (up to const). |
| `grad_norm` | \(\|\nabla U(\theta)\|\) | Gradient magnitude for SNR. |
| `theta_norm` | \(\|\theta\|\) | Parameter scale. |
| `f_nll` | CE on **train batch** with same reduction as U | Should equal U_data term (consistency check). |
| `f_margin` | Mean over train batch of (logit_true − max_other) | Classification margin (always mean). |
| `snr` | \((h \cdot \|\nabla U\|) / (\sqrt{2hd} \cdot \sigma)\) | Signal-to-noise ratio; &lt;1e-3 ⇒ near random walk, ≫1 ⇒ near GD. |
| `delta_U` | \(U_t - U_{t-1}\) | U change; fluctuating ⇒ exploring. |
| `drift_step_norm` | \(\|h\,\nabla U\|\) | Drift magnitude. |
| `noise_step_norm` | \(\|\sqrt{2h}\,\sigma\,\xi\|\) | Noise magnitude. |
| `delta_theta_norm` | \(\|\theta_{t+1} - \theta_t\|\) | Total step size. |

**U decomposition** (logged for sanity):

- `U_prior` = \((\alpha/2)\|\theta\|^2\)
- `U_data` = \(U - U_\text{prior}\) = CE term (sum or mean per config)
- `ce_mean_train`, `ce_sum_train`: derived so both are available
- `U_data_minus_ce`: should be ≈0 (consistency)

### 6.2 Probe Functions \(f(\theta)\) (saved in samples_metrics.npz)

| Probe | Data | Calculation | Used For |
|-------|------|-------------|----------|
| `f_nll` | Train batch | CE on (x_train, y_train), same reduction (mean/sum) as U | Matches U_data; consistency check |
| `f_margin` | Train batch | Mean over train batch of (logit_true − max_other) | Classification margin probe |
| `f_pc1`, `f_pc2` | Probe set | Random projection of flattened logits: `logit_proj[i] @ logits_probe.reshape(-1)` | Logit-space probe |
| `f_proj1`, `f_proj2` | — | Random projection of \(\theta - \theta_0\): `v[i] @ diff` | Param-space probe |
| `f_dist` | — | \(\|\theta - \theta_0\|^2\) | Distance from init |

**Calculation details**:
- **f_nll, f_margin**: One forward pass on train batch; CE and margin computed per-sample then reduced (sum/mean for CE, mean for margin).
- **f_pc1, f_pc2**: One forward pass on probe set; logits reshaped to (probe_size × 10), projected by fixed random vectors.
- **f_proj1, f_proj2, f_dist**: No forward pass; use flattened params \(\theta\) and reference \(\theta_0\).

### 6.3 Convergence Metrics (from compute_convergence.py)

- **R̂ (split-Rhat)**: Split each chain in half, compute potential scale reduction.
- **ESS (bulk)**: \(n / (1 + 2\sum_k \rho_k)\) from autocorrelation.
- **ESS-rate**: ESS per gradient evaluation.

### 6.4 Proxy LSI (from compute_lsi_proxy.py)

- **rho_hat_f** = \(\widehat{\mathbb{E}}\|\nabla f\|^2 / \widehat{\mathrm{Var}}(f)\) for each probe \(f\) with gradient estimates.
- Uses post-burn-in samples; `grad_norm_sq` is computed every G saved samples.
- **grad_norm_sq** uses same data as each probe: train batch for f_nll/f_margin, probe set for f_pc1/f_pc2, no data for f_proj1/f_proj2/f_dist.

---

## 7. Diagnostics (What We Compute and Why)

### 7.1 Param/Grad Sanity

| Diagnostic | How Computed | Purpose |
|------------|--------------|---------|
| `theta_max_abs`, `finite_params`, `nan_count_params` | Max over all params, finite check | Detect explosion / NaN. |
| `gradU_max_abs`, `finite_grad`, `nan_count_grads` | Same for gradients | Gradient sanity. |
| `logit_max_abs`, `logsumexp_max`, `pmax_mean`, `logits_finite` | On probe batch | Logit scale (from forward on probe set). |
| `nll_probe`, `margin_probe` | On **train** batch | CE and mean margin on train (must match U_data). |

### 7.2 BN / Activation Hooks

| Diagnostic | How Computed | Purpose |
|------------|--------------|---------|
| `bn_runmean_maxabs`, `bn_runvar_maxabs`, `bn_buffers_finite` | Max over BN running_mean/var | BN buffer sanity (logged every 500 steps). |
| `act_max_abs` | Max activation from BasicBlock/Bottleneck hooks | Activation scale. |

### 7.3 Locality and OU Test

| Diagnostic | How Computed | Purpose |
|------------|--------------|---------|
| `dist_to_ref` | \(\sqrt{f_\text{dist}}\) = \(\|\theta - \theta_0\|\) | Distance from pretrained init. |
| `theta_norm_over_ou` | \(\|\theta\| / \sqrt{d/\alpha}\) | OU stationary radius; &lt;1 ⇒ inside prior scale. |
| `theta_norm_sq_over_pred_ou` | \(\|\theta\|^2 / \text{predicted}\) from OU process | Pure prior diffusion test. |

### 7.4 Stop-Early Flags

| Flag | Condition | Meaning |
|------|-----------|---------|
| `bad_locality_raw` | dist / √(t) &gt; 5√(2d) | Diffusion-normalized drift above 5× expected. |
| `bad_locality` | `bad_locality_raw` for ≥ K consecutive log intervals **and** past burn-in | Persistent excessive drift. |
| `bad_prediction_raw` | nll_probe_mean &gt; 2 × nll_probe_mean@step1 + 2 | Probe NLL (mean) degraded. |
| `bad_prediction` | `bad_prediction_raw` for ≥ K consecutive log intervals **and** past burn-in | Persistent prediction degradation. |
| `abort_suggested` | bad_locality OR bad_prediction | Suggest early stop (debounced). |

- **nll_probe_mean** = nll_probe / n_train when ce_reduction=sum; both logged.
- **Config**: `abort_consecutive_intervals` (K=3), `abort_after_burnin_only` (True).

### 7.5 diagnose_ula.py

- Decomposes gradient: \(\nabla U = \nabla\text{NLL} + \alpha\theta\).
- Computes SNR, drift coefficient, restorative coefficient, balance ratio.
- Uses config `ce_reduction` (or `--ce-reduction`) so it matches runs.
- **Determinism**: cudnn.benchmark=False, cudnn.deterministic=True for diagnostic runs.

---

## 8. CE Reduction: Sum vs Mean

**Current default**: `ce_reduction = "sum"` (in config.py).

### Where CE Reduction Is Used

| Location | CE Reduction | Notes |
|----------|--------------|-------|
| **Potential U** (ula/potential.py) | config.ce_reduction | Full U for ULA. |
| **ULA step** (ula/step.py) | config.ce_reduction | Same as U. |
| **Probes** f_nll (probes/probes.py) | config.ce_reduction | Must match U for f_nll = U_data. |
| **probe_metrics** (run/diagnostics.py) | config.ce_reduction | nll_probe on train batch. |
| **Pretrain** (run/chain.py, scripts/pretrain.py) | **always "mean"** | SGD loss. |
| **eval_checkpoint.py**, **diagnose_sampler_load.py** | **always "mean"** | One-off evals. |
| **diagnose_ula.py** | config.ce_reduction or --ce-reduction | Matches runs. |

### Sum vs Mean: Tradeoffs

- **Sum CE**: Matches true log-posterior scaling \(-\log p(\theta|\mathcal{D}) \propto \sum_i \text{CE}_i + \frac{\alpha}{2}\|\theta\|^2\). Gradients are ~\(n\) times larger than mean ⇒ requires much smaller \(h\) (e.g. 1e-9) for stability.
- **Mean CE**: Smaller gradients ⇒ can use larger \(h\) (e.g. 1e-5) ⇒ better SNR and stability, but U and gradients are scaled by \(1/n\) relative to the Bayesian posterior.

---

## 9. model.train() vs model.eval() and BN / Dropout

### 9.1 Pretrain

- `model.train()` throughout.
- Full-batch SGD with mean CE.
- BN uses batch statistics; running stats are updated.

### 9.2 ULA Chain (bn_mode)

**Default**: `bn_mode = "eval"` (deterministic, partition-invariant target).

Two modes, set via `--bn-mode`:

| Mode | Effect |
|------|--------|
| **eval** | `model.eval()`: BN uses running_mean/var; deterministic, partition-invariant U(θ). Optional **BN calibration** (`--bn-calibration-steps N`): N forward passes (train mode, no grad) over subset to populate running stats before switching to eval for sampling. |
| **batchstat_frozen** | `model.train()` for BN (batch stats). Running_mean/var frozen (momentum=0). **Must use single batch** (batch_size=n_train) for fixed target; partition-sensitive. |

**Note**: ResNet CIFAR has **no Dropout**. The dropout handling is defensive for future models.

### 9.3 Other Scripts

- **eval_checkpoint.py**: `model.eval()`, no dropout, BN running stats.
- **diagnose_ula.py**: User chooses bn_mode (eval or batchstat_frozen) before gradient computation.
- **diagnose_sampler_load.py**: Tests train vs eval for load/mode/transform checks.

---

## 10. Problems Encountered and Fixes

### 10.1 Transform Mismatch (FIXED)

- **Problem**: Pretrain used TRAIN_TRANSFORM (augmentation), chain used TEST_TRANSFORM. Different inputs ⇒ f_nll and U_data could not match.
- **Fix**: Use `eval_transform=True` in `get_train_loader` for pretrain and chain so both see identical, deterministic data.

### 10.2 f_nll / U_data Mismatch (FIXED)

- **Problem**: f_nll was computed on probe set; U_data on train batch.
- **Fix**: f_nll and nll_probe now use the **train batch** (same as U_data) when `nll_data=train_data` is passed.

### 10.3 U, grad_norm, f_nll Blowing Up (PARTIALLY SOLVED)

- **Observation**: At h=1e-9, alpha=0.1, n_train=1024, U/f_nll/grad_norm increase over time (e.g. f_nll 0.24 → 27.6 over 30k steps).
- **Cause**: At h=1e-9, drift ≈ 1e-6 vs noise ≈ 0.014 ⇒ chain is almost pure diffusion and drifts away from the mode.
- **Attempted**: Larger h (e.g. 5e-7) causes faster blow-up.
- **Interpretation**: With **sum CE**, gradients are ~n_train larger. h=1e-9 is near the stability limit; larger h is unstable. Mean CE would allow larger h but we default to sum for posterior fidelity.

### 10.4 SNR Too Low or Too High

- **Too low** (&lt;1e-3): Drift negligible vs noise ⇒ near random walk.
- **Too high** (&gt;0.1–1): Chain behaves like gradient descent, little exploration.
- **Tools**: diagnose_ula.py, iter_metrics `snr`, `drift_step_norm`, `noise_step_norm`.

### 10.5 Prior vs Data Gradient Cancellation

- **Problem**: Near a mode, \(\nabla\text{NLL}\) and \(\alpha\theta\) can nearly cancel ⇒ \(\|\nabla U\| \ll \|\nabla\text{NLL}\|\) or \(\|\alpha\theta\|\).
- **Effect**: SNR can be misleading; chain can appear stable while drifting.
- **Mitigation**: Reduced alpha from 0.05 to 0.01 (or 0.1) to improve SNR; monitor U decomposition.

### 10.6 Prior Precision (alpha) and Step Size (h)

- **Alpha**: Higher alpha ⇒ stronger pull toward origin, less drift. We sweep alpha ∈ {0.1, 0.3, 1.0, 3.0}.
- **h**: With sum CE, h must be very small (e.g. 1e-9). Experiment plan suggested h ∈ {1e-6, 5e-6, 1e-5, 2e-5} for mean CE.
- **ou_radius_pred** = \(\sqrt{d/\alpha}\): Predicted OU stationary std; used for theta_norm_over_ou diagnostic.

### 10.7 BN Mode During Sampling

- **batchstat_frozen**: BN uses batch stats (train-like) but running buffers frozen. More faithful to “training-time” behavior.
- **eval**: BN uses running stats from pretrain. Deterministic; can differ from batchstat if data distribution shifts.

---

## 11. File Layout (Key Paths)

| Path | Purpose |
|------|---------|
| `config.py` | RunConfig: h, alpha, ce_reduction, bn_mode, T, B, S, etc. |
| `ula/potential.py` | compute_U (CE + prior). |
| `ula/step.py` | ula_step. |
| `run/chain.py` | Main chain loop, logging, probes. |
| `run/diagnostics.py` | param/grad stats, probe_metrics, BN stats. |
| `run/bn_mode.py` | set_bn_batchstats_freeze_buffers. |
| `run/persistence.py` | write_iter_metrics, write_samples_metrics, dump_failure. |
| `probes/probes.py` | evaluate_probes, get_probe_value_for_grad. |
| `data/cifar.py` | get_train_loader, get_probe_loader, TRAIN_TRANSFORM, TEST_TRANSFORM. |
| `data/indices.py` | get_train_subset_indices, get_probe_indices (persisted to JSON). |
| `probes/random_projections.py` | get_or_create_param_projections, get_or_create_logit_projection (persisted to .pt). |
| `scripts/run_single_chain.py` | Entry point for one chain. |
| `scripts/pretrain.py` | Pretrain with fixed seed. |
| `scripts/diagnose_ula.py` | Gradient decomposition, SNR, balance. |
| `scripts/diagnose_sampler_load.py` | Load checkpoint, test train vs eval mode, transform sensitivity. |
| `scripts/eval_checkpoint.py` | One-off eval of checkpoint (mean CE, accuracy). |
| `scripts/test_mean_vs_sum_ce.py` | Compare mean vs sum at given h. |
| `experiments/runs/<run_name>/` | run_config.yaml, iter_metrics.jsonl, samples_metrics.npz. |
| `experiments/analysis/compute_convergence.py` | R̂, ESS, ESS-rate. |
| `experiments/analysis/compute_lsi_proxy.py` | rho_hat per probe. |

---

## 12. Implementation Notes and Potential Pitfalls

### 12.1 Config Default h vs ce_reduction

- **config.py** default `h = 1e-4` but with `ce_reduction = "sum"` we typically need `h ~ 1e-9` for stability.
- The default h in config is tuned for mean CE. With sum CE, explicitly pass `--h 1e-9` (or smaller).

### 12.2 Projection Files Are Not Keyed by Width / Probe Size

- **Param projections** (`v1.pt`, `v2.pt`): Saved in `experiments/data/` with fixed filenames. Shape must match `d_param` (parameter count).
- **Logit projection** (`logit_proj.pt`): Shape must match `probe_size × num_classes`.
- **Pitfall**: If you run width 0.1 (d ~ 100k) then width 1 (d ~ 11M), the second run overwrites v1.pt/v2.pt. A later run of width 0.1 will load mismatched shapes, reject them, recreate, and overwrite again. **Workaround**: Run each width in isolation, or use a separate `data_dir` per width.

### 12.3 Pretrain vs Chain: Matching Seeds

- Pretrain uses `--seed` (default 42) for both dataset indices and SGD.
- Chain uses `--seed` for `dataset_seed` (data indices).
- **Critical**: Use the same `--seed` for pretrain and chain so they see identical data. Mismatched seeds ⇒ different train subsets ⇒ pretrain checkpoint does not match chain data.

### 12.4 theta0 Reference for Probes

- **With pretrain_path**: `theta0` = loaded checkpoint (post-pretrain).
- **Without pretrain_path**: `theta0` = model state after per-chain pretrain.
- `f_dist`, `f_proj1`, `f_proj2` measure distance from this init. `dist_to_ref`, `bad_locality` use the same reference.

### 12.5 cuDNN and Reproducibility

- Chain sets `torch.backends.cudnn.benchmark = True` for speed. This can cause non-determinism across runs.
- Pretrain sets `cudnn.deterministic = True` and `benchmark = False` for reproducibility.
- **For strict ULA reproducibility**: Disable cuDNN benchmark (or set deterministic); expect slowdown.

### 12.6 Failure Handling

- On first non-finite (U, params, or grads): `dump_failure` writes `FAIL_step{N}.pt`, appends final iter_metrics, writes samples_metrics, and returns.
- No further ULA steps run. You get partial output; use FAIL_step{N}.pt to debug.

### 12.7 Seed Schedule

- **chain_seed** = `args.seed + args.chain * 1000` (e.g. chain 0: 42, chain 1: 1042).
- ULA step noise: `config.chain_seed + chain_id * 1000 + step`. Chains are independent.
- Init noise (when no pretrain): `config.chain_seed + chain_id * 1000`.

### 12.8 Gradients Retained After ula_step

- `ula_step` leaves model parameters with `.grad` set. `probe_metrics` and `evaluate_probes` run after the step; they use `torch.no_grad()` where appropriate, but param_vector_stats / grad_vector_stats read the current grads. No extra backward needed.

### 12.9 diagnose_ula.py ce_reduction Mismatch

- `diagnose_ula.py` does not pass `ce_reduction` to `compute_U` (uses default "mean"). Cross-check uses `reduction="sum"`.
- If runs use `ce_reduction="sum"`, the diagnostic gradient decomposition may not match. Consider adding `--ce-reduction` to the script.

### 12.10 Temperature Unused

- Config has `temperature: float = 1.0` but it is not used in the ULA step. Noise std is \(\sqrt{2h} \cdot \sigma\); temperature would scale the noise as \(\sqrt{2hT}\). Effective temperature is 1.

### 12.11 Indices Keyed by n and probe_size Only (Not by Seed)

- `train_subset_indices.json`: key = str(n). Different n_train ⇒ different keys.
- `probe_indices.json`: key = str(probe_size). Different probe_size ⇒ different keys.
- **dataset_seed is not part of the key**: Indices are created with the given seed only when the key is missing. Once indices for n=1024 exist, any later call with n=1024 returns the cached indices regardless of dataset_seed. To get different subsets for different seeds, you would need a different data_dir or to delete the JSON before running.

---

## 13. Quick Sanity Checklist

When diagnosing a run:

1. **Finite**: `finite_params`, `finite_grad`, `finite_loss` all true.
2. **SNR**: `snr` in [1e-3, 0.1] band (not pure walk, not pure GD).
3. **U decomposition**: `U_data_minus_ce` ≈ 0; `U_prior` and `U_data` in plausible range.
4. **Locality**: `dist_to_ref` not exploding; `bad_locality`, `bad_prediction`, `abort_suggested` false if run should be healthy.
5. **Consistency**: f_nll matches ce_sum_train (sum) or ce_mean_train (mean) depending on ce_reduction.
6. **Transform**: Pretrain and chain both use eval_transform=True.
7. **Seeds**: Same `--seed` for pretrain and chain; pretrain checkpoint matches chain (width, n_train).
8. **Projections**: If running multiple widths, use separate `data_dir` or run one width at a time to avoid overwriting v1.pt / v2.pt / logit_proj.pt.
