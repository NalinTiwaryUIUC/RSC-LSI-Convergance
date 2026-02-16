# Experimental Plan: Convergence and Proxy LSI Scaling in Wide ResNet-18 on CIFAR-10 (PyTorch)

This document specifies **all experimental details** needed to reproduce:

1. **Convergence vs width** analysis (ESS, IACT, R̂, etc.)
2. **Proxy LSI constant vs width** analysis  
   \[
   \widehat{\rho}_f(m) = \frac{\widehat{\mathbb{E}}\|\nabla_\theta f(\theta)\|^2}{\widehat{\mathrm{Var}}(f(\theta))}
   \]

We use **normal ULA with small step sizes**, monitor domain fidelity for \(B_t\), and ensure fair compute normalization.  
Implementation assumptions: **PyTorch**, CIFAR-10, ResNet-18 CIFAR variant.

---

## 1. Goals

Empirically test:

- Whether convergence efficiency **degrades with width**
- Whether a **proxy LSI constant stabilizes (dimension-free behavior)** beyond a width threshold
- Whether results are stable under step-size halving (ULA discretization sanity)

---

## 2. Dataset

### 2.1 CIFAR-10

- Train: 50,000 images
- Test: 10,000 images
- Input size: 32×32 RGB

### 2.2 Theory-faithful Track (Recommended)

To better align with **full-batch ULA** assumptions and make “wide regime” plausibly testable, use **subsampled CIFAR-10**:

- \( n \in \{512, 1024, 2048\} \)

Implementation detail:
- Fix a global permutation seed
- Take the first `n` indices of the permuted training set
- Save indices to disk (e.g., `train_subset_indices.json`) so the subset is identical across widths/chains.

> If you also want a “realistic” track, run on full 50k, but interpret results as “empirical stabilization,” not literally crossing \(m \sim n^2\).

### 2.3 Preprocessing (exact)

Train transforms:

```python
import torchvision.transforms as T

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)),
])
```

Test transforms:

```python
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)),
])
```

### 2.4 Probe set \(S_{\text{probe}}\)

- Fixed set of **512 test examples** (or 256 if compute is tight).
- Fix indices once and reuse for every width/chain.
- Save to `probe_indices.json`.

---

## 3. Architecture

### 3.1 Model

Use **ResNet-18 for CIFAR**:

- 3×3 conv stem, stride 1, padding 1
- No 7×7 conv
- No maxpool
- 4 stages, 2 residual blocks each
- Global average pooling
- Linear head to 10 classes

If you don’t already have a CIFAR ResNet-18 implementation, use `torchvision.models.resnet18` **only if** you modify the stem (torchvision’s default is ImageNet-style). Prefer a CIFAR-native ResNet.

### 3.2 Width grid (exact)

Parameterize the model by a **base width**:

- `base_width = int(64 * w)`
- Stage widths: `[base_width, 2*base_width, 4*base_width, 8*base_width]`

Width multipliers:

```
w ∈ {0.5, 1, 2, 4}
```

Optional: add `w = 8` if compute allows.

Record parameter count:
- `d = sum(p.numel() for p in model.parameters())`

---

## 4. Target Distribution (Posterior)

Define potential (negative log posterior up to constant):

\[
U(\theta) = \sum_{(x_i,y_i)\in \mathcal D_{\text{train}}} \text{CE}(\theta;x_i,y_i)
+ \frac{\alpha}{2}\|\theta\|^2
\]

Use:
- Prior precision: `α = 1e-2` (main width sweep)
- Temperature: 1

**Important:** For classic ULA targeting this \(U\), use **full-batch gradients**. If you use minibatches, you are closer to SGLD and interpretations change.

---

## 5. ULA Algorithm (Normal ULA)

### 5.1 Update rule (exact)

For step size `h`:

```python
theta = theta - h * grad_U(theta) + (2*h)**0.5 * torch.randn_like(theta)
```

Implementation detail in PyTorch:
- Treat model parameters as the state.
- After computing `U` (full batch), call `U.backward()` to get gradients.
- Apply update to each parameter tensor.

### 5.2 Step sizes (exact shortlist)

Candidates:

```
h ∈ {1e-6, 5e-6, 1e-5, 2e-5}
```

For each width:
1) pick the **largest** `h` such that the \(B_t\) violation rate is < 1% (see §6)
2) also run `h/2` (discretization sanity check)

---

## 6. Domain \(B_t\) Monitoring (No Projection)

Since you cannot run projected ULA, you **must** monitor whether the chain stays in a region consistent with the theorem’s domain.

Use a practical Euclidean ball around initialization:

\[
B_t := \{ \theta : \|\theta - \theta_0\|_2 \le \rho_2 \}
\]

### 6.1 Radius choice (exact starting point)

Let:
- `r0 = ||theta0||_2` (flatten all parameters)
- `rho2 = 0.05 * r0` (start)

If too restrictive (violation is always huge), use:
- `rho2 = 0.10 * r0`

Fix `rho2` across widths.

### 6.2 Logged domain metrics (every iteration)

- `theta_dist = ||θ - θ0||_2`
- `bt_margin = theta_dist - rho2`
- `inside_bt = (bt_margin <= 0)`

**Acceptance criterion for “small step stays in ball”:**
- post-burn-in violation rate < 1%

If not achievable:
- proceed, but compute all summary statistics both:
  - (a) using all samples
  - (b) using only samples with `inside_bt == True`
and show they match qualitatively.

---

## 7. Chains and Run Schedule

Per width:

- **K = 4 chains**
- independent seeds
- initialize each chain as:
  \[
  \theta^{(0)} = \theta_0 + \varepsilon,\quad \varepsilon\sim\mathcal N(0, \sigma_{\text{init}}^2 I)
  \]
  with `σ_init = 1e-4` times parameter std scale (tiny, just to decorrelate chains)

### 7.1 Length / saving schedule (exact)

Recommended defaults (Track A, full-batch on subsampled training set):

- Total steps: `T = 200_000`
- Burn-in: `B = 50_000`
- Save stride: `S = 200`  (save probes every 200 steps)

This yields ≈750 saved samples per chain post burn-in.

### 7.2 Bias sanity (exact)

For each width, run two step sizes:
- `h`
- `h/2`

(You can do 1 chain for `h/2` if compute is tight, but 4 is better.)

---

## 8. Probe Functions \(f(\theta)\) (used for both threads)

Use the **same probe set** for:
- convergence diagnostics (R̂/ESS/IACT)
- proxy LSI computation

### 8.1 Function-space probes (primary)

Evaluate on the fixed probe set \(S_{\text{probe}}\):

1) **Probe NLL**
- `f_nll(θ)` = mean cross-entropy on probe set

2) **Probe margin**
- `f_margin(θ)` = mean over probe examples:
  - `logit[y] - max_{k≠y} logit[k]`

3) **Logit PCA coordinates** (recommended)
Procedure:
- Run a short pilot chain at width `w=1`
- Collect logits on probe set for ~200 samples
- Flatten to a vector length `|probe|*10`
- Fit PCA; keep top 2 vectors `u1,u2`
- Freeze them and reuse across all widths
Then:
- `f_pc1(θ) = <u1, vec(logits(θ))>`
- `f_pc2(θ) = <u2, vec(logits(θ))>`

If you don’t want PCA:
- replace with 2 fixed random projections of the logit vector.

### 8.2 Parameter-space probes (controls)

4) `f_proj1(θ) = v1 · (θ - θ0)`  
5) `f_proj2(θ) = v2 · (θ - θ0)`  
where `v1,v2` are fixed Gaussian random vectors (seeded) in parameter space.

6) `f_dist(θ) = ||θ - θ0||^2`

---

## 9. What to compute, store, and how

### 9.1 Run manifest (one per chain)

Save `run_config.yaml` with:
- width multiplier `w`, base_width
- dataset subset size `n_train` and subset indices file
- step size `h`, temperature, α
- seeds (chain seed, dataset seed)
- T, B, S
- rho2 definition
- probe indices file
- git commit hash (if you can)

### 9.2 Per-iteration log (cheap, JSONL)

File: `iter_metrics.jsonl` (one JSON per step)

Fields:
- `step`
- `grad_evals` (equals `step` for full-batch)
- `theta_dist`
- `bt_margin`
- `inside_bt` (0/1)
- optional: `U_train` (full-batch value; expensive but doable on small n)

### 9.3 Saved-sample log (every S steps)

File: `samples_metrics.npz` (or parquet)

Arrays for each saved sample:
- `step`, `grad_evals`
- `inside_bt`, `bt_margin`
- `f_values`: arrays for each probe `f_*`

### 9.4 Gradient norms for proxy LSI (expensive; subsample)

For proxy LSI you need:
- `grad_norm_sq[f] = ||∇θ f(θ)||^2`

Compute gradient norms only for:
- `f_nll`, `f_margin`, `f_pc1`, `f_proj1`, `f_dist`  (5 probes is enough)

And compute them on a subsampled set of saved points:
- every `G=5` saved samples (i.e., every `5*S` steps)

Store in `samples_metrics.npz` as:
- `grad_norm_sq__f_nll`, etc.

> Tip: compute ∇f using `torch.autograd.grad(f_scalar, params, retain_graph=False, create_graph=False)`.

---

## 10. Thread 1: Convergence vs Width

For each width `w` and probe `f`, compute (post burn-in):

### 10.1 Split-R̂ (multi-chain)
Compute split-R̂ across K chains.

Store:
- `rhat_final[w,f,h]`

Threshold (suggested):
- `R̂ ≤ 1.05` acceptable
- `R̂ ≤ 1.01` strong

### 10.2 ESS (bulk)
Compute bulk ESS per probe.

Store:
- `ess_bulk[w,f,h]`

### 10.3 ESS per compute (main metric)
For full-batch ULA:
- 1 step = 1 full gradient evaluation

Define:
\[
\text{ESS\_rate}(w,f,h) = \frac{\text{ESS\_bulk}(w,f,h)}{T_{\text{post}}}
\]
where \(T_{\text{post}}\) is post-burn-in steps used.

Report as:
- ESS per 1e6 grad-evals:
  - `ess_rate_1e6 = ess_rate * 1e6`

### 10.4 Time to R̂ threshold (optional)
Compute earliest saved index where `R̂ ≤ 1.05` and remains ≤ 1.05 for a window of e.g. 50 saved points.

Store:
- `t_rhat_105[w,f,h]` in grad-evals

---

## 11. Thread 2: Proxy LSI vs Width

For each width `w` and probe `f` (those with grad norms):

Compute:
\[
\widehat{\rho}_f(w,h) = \frac{\widehat{\mathbb{E}}\|\nabla f\|^2}{\widehat{\mathrm{Var}}(f)}.
\]

Procedure:
1) Pool post-burn-in samples across chains
2) Optionally restrict to `inside_bt == True`
3) Compute:
   - `var_f = Var(f_values)`
   - `Egrad = Mean(grad_norm_sq)`
4) Proxy:
   - `rho_hat = Egrad / var_f`

Uncertainty:
- compute `rho_hat` per chain, then report mean ± (std / sqrt(K))

Also compute summary:
- `rho_hat_min[w,h] = min_f rho_hat[w,f,h]` over your chosen probe set

---

## 12. Plots (exact)

### 12.1 Domain fidelity plots (sanity)
**Plot D1: bt_margin vs iteration**
- x: step (or grad-evals)
- y: `bt_margin`
- show one representative chain per width (or overlay transparently)

**Plot D2: violation rate vs width**
- x: width multiplier `w`
- y: % steps with `inside_bt == False` (post burn-in)
- show for both `h` and `h/2`

---

### 12.2 Convergence vs width
**Plot C1: ESS-rate vs width**
- x: width `w` (or parameter count)
- y: ESS per 1e6 grad-evals
- curves: `f_nll`, `f_margin`, `f_pc1`
- markers: `h` and `h/2`
- error bars: chain-to-chain SE

**Plot C2: time-to-R̂≤1.05 vs width**
- x: width
- y: grad-evals to reach threshold
- curves: same probes

Expected shape if “dimension-free” behavior holds:
- ESS-rate should **not degrade** with width in the wide regime (plateau)
- time-to-R̂ should **not blow up** with width

---

### 12.3 Proxy LSI vs width
**Plot L1: rho_hat vs width**
- x: width
- y: `rho_hat`
- curves: up to 5 probes
- show both `h` and `h/2`
- error bars: chain-to-chain SE

**Plot L2: rho_hat_min vs width**
- x: width
- y: `min_f rho_hat`
- single curve summary

Expected:
- rho_hat curves should **plateau** with width once wide.

---

## 13. Sanity checks (non-negotiable)

1) **Step-size halving**: compare `h` vs `h/2`. If results shift materially, discretization bias dominates.
2) **Inside-Bt restriction**: recompute summary metrics using only inside-Bt samples; results should be qualitatively consistent.
3) **Adequate ESS**: aim for ESS ≥ 100 per key probe before making strong claims.

---

## 14. Suggested directory structure

```
experiments/
  runs/
    w1_n1024_alpha1e-2_h1e-5_chain0/
      run_config.yaml
      iter_metrics.jsonl
      samples_metrics.npz
      stdout.log
analysis/
  compute_convergence.py
  compute_lsi_proxy.py
  make_plots.py
summaries/
  bt_fidelity.csv
  convergence.csv
  lsi_proxy.csv
figures/
  bt_violation_vs_width.png
  ess_rate_vs_width.png
  time_to_rhat_vs_width.png
  rho_hat_vs_width.png
  rho_hat_min_vs_width.png
```

---

## 15. Expected outcomes (what “good” looks like)

If your theory aligns with practice:

- **Bt violation rate** stays near zero (or tiny) across widths with chosen small `h`
- **ESS-rate** does not trend down with width once wide enough (plateau)
- **rho_hat** does not trend down with width once wide enough (plateau)
- Results are stable between `h` and `h/2`

---

## 16. Notes for implementation in PyTorch

- For flattening parameters:
  - create helpers `flatten_params(model)` and `unflatten_like(vector, model)`
- For random parameter projections:
  - draw `v` once with fixed seed, store it to disk (`v1.pt`, `v2.pt`) so it’s identical across widths/chains
- For gradient norms:
  - use `torch.autograd.grad(f, params, ...)`, then sum squared norms of returned tensors
- Always run evaluation probes in `model.eval()` and with `torch.no_grad()` **except** when computing gradients of probes.

---
