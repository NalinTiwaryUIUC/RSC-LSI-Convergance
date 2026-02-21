# Running the LSI / ULA experiment

## Does everything work?

Yes. The pipeline is implemented and tested:

- **Unit tests**: `python3 -m unittest discover tests -v` (data, model, ULA, probes, chain persistence).
- **Smoke run**: `python3 scripts/smoke_run.py` runs a short chain (T=500) and writes `experiments/runs/smoke_w0.5_n128_h1e-5_chain0/`.

You can re-run the smoke and then analysis/plots to confirm end-to-end before launching the full experiment.

---

## How to run the actual experiment

### Plan summary (from EXPERIMENT_PLAN_LSI_PYTORCH.md)

- **Widths**: `w ∈ {0.5, 1, 2, 4}` (optionally 8).
- **Step sizes**: Use small h (e.g. 1e-5); optionally run `h/2` for a discretization check.
- **Noise scale**: Default 1.0 (standard ULA). Use `--noise-scale` to override; run `scripts/diagnose_ula.py` to tune for balance.
- **Pretrain**: Default 2000 full-batch SGD steps before ULA so chains start near a mode. For standardized init across chains, run `scripts/pretrain.py` once per (width, n_train) and pass `--pretrain-path` to run and diagnose.
- **Chains**: K = 4 per (width, h).
- **Schedule**: T = 200_000, B = 50_000, S = 200 (≈750 saved samples per chain after burn-in).
- **Data**: Subsampled CIFAR-10 with `n_train ∈ {512, 1024, 2048}` (e.g. 1024); probe_size = 512.

### Optional: Pretrain once per (width, n_train) for standardized init

Use a fixed random seed so all chains start from the same pretrained checkpoint:

```bash
# Pretrain for width 1, n_train 1024 (fixed seed 42)
python3 scripts/pretrain.py --width 1 --n_train 1024 --pretrain-steps 2000
# Writes experiments/checkpoints/pretrain_w1_n1024.pt

# For other widths:
python3 scripts/pretrain.py --width 0.1 --n_train 1024
python3 scripts/pretrain.py --width 0.01 --n_train 1024
```

Then pass `--pretrain-path` to run and diagnose (see Option A/B below).

### Option A: Run one chain at a time (local or single job, with SGD warm-up)

```bash
# Example: width 1, h=1e-5, 4 chains, n_train=1024 (plan defaults)
# With shared pretrain (run pretrain.py first):
PRETRAIN=experiments/checkpoints/pretrain_w1_n1024.pt
for chain in 0 1 2 3; do
  python3 scripts/run_single_chain.py \
    --width 1 --h 1e-5 --chain $chain --n_train 1024 \
    --pretrain-path "$PRETRAIN"
done

# Or without shared pretrain (per-chain SGD warm-up):
for chain in 0 1 2 3; do
  python3 scripts/run_single_chain.py \
    --width 1 --h 1e-5 --chain $chain --n_train 1024 \
    --pretrain-steps 2000 --pretrain-lr 0.1
done
```

Repeat for each (width, h) you want. Run dirs will be like:

`experiments/runs/w1_n1024_h1e-5_chain0`, `..._chain1`, etc.

### Option B: Full grid (sequential, for reference)

```bash
# One width and h for illustration; extend to full grid as needed.
# Include SGD warm-up so chains start near a reasonable region.
for w in 0.5 1 2 4; do
  for chain in 0 1 2 3; do
    python3 scripts/run_single_chain.py \
      --width $w \
      --h 1e-5 \
      --chain $chain \
      --n_train 1024 \
      --pretrain-steps 1000 \
      --pretrain-lr 0.1
  done
done
```

### After all chains for a given (width, h)

Run analysis and plots on the run dirs for that (width, h):

```bash
# Convergence (Rhat, ESS, ESS-rate)
python3 experiments/analysis/compute_convergence.py experiments/runs/w1_n1024_h1e-5_chain{0,1,2,3} --B 50000 --S 200 -o experiments/summaries/convergence.csv

# Proxy LSI
python3 experiments/analysis/compute_lsi_proxy.py experiments/runs/w1_n1024_h1e-5_chain{0,1,2,3} --B 50000 --G 5 --S 200 -o experiments/summaries/lsi_proxy.csv

# Plots (use summary CSVs you produced)
python3 experiments/analysis/make_plots.py --convergence-csv experiments/summaries/convergence.csv --lsi-csv experiments/summaries/lsi_proxy.csv -o experiments/figures
```

You can aggregate summaries per (width, h) and then plot across widths (e.g. ESS-rate vs width, rho_hat vs width) by editing the plot script or merging CSVs with a `width`/`h` column.

---

## Running on Google Colab

1. **Enable GPU**  
   Runtime → Change runtime type → Hardware accelerator: **GPU** (e.g. T4).

2. **Get the project**  
   Either clone from git or upload the project (e.g. as a zip) and unzip in `/content/`:
   ```bash
   # Option A: clone (if you have a repo)
   !git clone https://github.com/YOUR_USER/RSC_Conv.git
   %cd RSC_Conv

   # Option B: upload RSC_Conv.zip then
   # !unzip -q RSC_Conv.zip && cd RSC_Conv
   ```

3. **Install dependencies**  
   In a cell:
   ```bash
   %pip install -q -r requirements.txt
   ```
   Colab already has PyTorch; this aligns torchvision and the rest.

4. **Run from project root**  
   Use the same commands as local, from the notebook (with `!` or `%run`):
   ```bash
   # Quick smoke (T=500) to check everything works
   !python3 scripts/smoke_run.py

   # Or one short chain with pretrain (e.g. T=2000 for a fast test)
   !python3 scripts/run_single_chain.py --width 1 --h 1e-5 --chain 0 --n_train 512 \
     --T 2000 --B 500 --S 100 --pretrain-steps 500 --pretrain-lr 0.1
   ```
   For a full single chain (T=200k), use the same args as in Option A above (no `--T`/`--B`/`--S`); runtime may be several hours and Colab can disconnect after ~12h, so keep that in mind.

5. **CIFAR-10 and data dir**  
   On first run, CIFAR-10 is downloaded under `./data`. Indices and projections are created under `experiments/data/` automatically when you run a chain.

6. **Keeping results across sessions**  
   To avoid losing runs if the runtime disconnects, mount Drive and point outputs there:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Then run with:
   ```bash
   !python3 scripts/run_single_chain.py ... --runs_dir /content/drive/MyDrive/RSC_Conv_runs
   ```
   (Create `RSC_Conv_runs` in Drive first, or use any folder you prefer.)

7. **Analysis and plots**  
   After a run, from the project root:
   ```bash
   !python3 experiments/analysis/compute_convergence.py experiments/runs/w1_n512_h1e-05_chain0 --B 500 --S 100 -o experiments/summaries/convergence.csv
   !python3 experiments/analysis/compute_lsi_proxy.py experiments/runs/w1_n512_h1e-05_chain0 --B 500 --G 5 --S 100 -o experiments/summaries/lsi_proxy.csv
   !python3 experiments/analysis/make_plots.py --convergence-csv experiments/summaries/convergence.csv --lsi-csv experiments/summaries/lsi_proxy.csv -o experiments/figures
   ```
   Adjust `--B` and `--S` to match the chain you ran. View figures in the Files panel or with `from IPython.display import Image; Image('experiments/figures/ess_rate_vs_width.png')`.

---

## External compute (SLURM, cloud, etc.)

### 1. Copy the project

- Clone or copy the repo (including `config.py`, `data/`, `models/`, `ula/`, `probes/`, `run/`, `scripts/`, `experiments/`, `tests/`) to the cluster or VM.
- Do **not** copy `data/` CIFAR downloads; re-download on the worker (see below) or copy only `experiments/data/*.json` and `*.pt` (indices and projections) if you want identical subsets across workers.

### 2. Environment

```bash
cd /path/to/RSC_Conv
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Optional: install a specific PyTorch build for your CUDA version (see [pytorch.org](https://pytorch.org)).

### 3. Data (CIFAR-10)

- On first run, CIFAR-10 is downloaded under `--root` (default `./data`). Use a shared path if all jobs use the same filesystem (e.g. `--root /shared/data/cifar10`).
- Indices and projections are under `experiments/data/` (or `--data_dir`). Generate them once (e.g. run one short chain or a tiny script that calls `get_train_subset_indices`, `get_probe_indices`, `get_or_create_param_projections`, `get_or_create_logit_projection`). Then copy `experiments/data/` to all workers so all use the same indices and projections.

### 4. One chain per job (recommended)

Submit one job per (width, h, chain). Each job runs a single chain and writes its own run dir.

A ready-to-use SLURM script is in `scripts/submit_chain.sh` (mail, logs, GPU, account/partition set for Illinois cluster). **Submit from the project directory**: `cd /path/to/RSC_Conv`, run `mkdir -p logs/lsi_ula` once, then `sbatch scripts/submit_chain.sh`. The script creates `.venv` and installs dependencies automatically if missing. It accepts `WIDTH H CHAIN N_TRAIN` as arguments.

Submit:

```bash
for w in 0.5 1 2 4; do
  for c in 0 1 2 3; do
    sbatch scripts/submit_chain.sh $w 1e-5 $c 1024
  done
done
```

Adjust SBATCH options in `scripts/submit_chain.sh` (time, mem, account, partition) for your cluster. Set `RSC_CONV_DIR` if you submit from a different directory.

### 5. After jobs finish

- Collect `experiments/runs/` (all run dirs) on one machine or a shared filesystem.
- Run the analysis scripts (compute_convergence, compute_lsi_proxy) with the appropriate run dirs for each (width, h).
- Run `make_plots.py` with the summary CSVs and run dirs to produce figures.

### 6. Optional: checkpointing / resume

The current implementation does not checkpoint mid-run. For very long runs (T=200k), consider adding checkpointing (save model state and step counter periodically and add a `--resume` path) or splitting into multiple segments.

---

## Quick reference

| What              | Command / path |
|-------------------|----------------|
| Colab             | See **Running on Google Colab** (GPU, pip install, then same commands as local). |
| Smoke run         | `python3 scripts/smoke_run.py` |
| One chain         | `python3 scripts/run_single_chain.py --width 1 --h 1e-5 --chain 0 --n_train 1024` |
| Unit tests        | `python3 -m unittest discover tests -v` |
| Run outputs       | `experiments/runs/<run_name>/` (run_config.yaml, iter_metrics.jsonl, samples_metrics.npz) |
| Mid-run checks    | `tail experiments/runs/<run_name>/iter_metrics.jsonl`; SLURM stdout shows progress every 10k steps |
| Summaries        | `experiments/summaries/*.csv` |
| Figures          | `experiments/figures/*.png` (from `make_plots.py`) |

### Single-chain diagnostics (from iter_metrics.jsonl + run_config.yaml)

From one chain you can sanity-check behaviour using these logged fields:

| Goal | What to check |
|------|----------------|
| **Not pure random walk** | `snr` in iter_metrics: should be in a meaningful band (e.g. > 1e-3). If snr ≪ 1e-3, drift is negligible vs noise. |
| **Not deterministic** | `snr` should not be huge (e.g. < 0.1–1). If snr ≫ 1, chain is almost gradient descent. |
| **Exploring (not stuck)** | `delta_U` and `U_train` over time: should fluctuate (delta_U positive and negative). Monotonic U or near-zero delta_U suggests stuck or purely drifting. |
| **Reasonable region** | `f_nll`, `f_margin`: finite and in a plausible range (f_nll not exploding, f_margin not absurd). |
| **Independent samples** | Run `compute_convergence.py` on the run dir(s) and check ESS; single-chain ESS is still informative. For R̂ you need multiple chains. |

`run_config.yaml` includes `param_count` (d) so you can recompute SNR or interpret scales from logs.
