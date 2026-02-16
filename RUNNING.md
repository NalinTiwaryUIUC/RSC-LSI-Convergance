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
- **Step sizes**: For each width, pick the largest `h ∈ {1e-6, 5e-6, 1e-5, 2e-5}` such that post burn-in B_t violation rate < 1%; then also run `h/2` (discretization check).
- **Chains**: K = 4 per (width, h).
- **Schedule**: T = 200_000, B = 50_000, S = 200 (≈750 saved samples per chain after burn-in).
- **Data**: Subsampled CIFAR-10 with `n_train ∈ {512, 1024, 2048}` (e.g. 1024); probe_size = 512.

### Option A: Run one chain at a time (local or single job)

```bash
# Example: width 1, h=1e-5, 4 chains, n_train=1024 (plan defaults)
for chain in 0 1 2 3; do
  python3 scripts/run_single_chain.py --width 1 --h 1e-5 --chain $chain --n_train 1024
done
```

Repeat for each (width, h) you want. Run dirs will be like:

`experiments/runs/w1_n1024_h1e-5_chain0`, `..._chain1`, etc.

### Option B: Full grid (sequential, for reference)

```bash
# One width and h for illustration; extend to full grid as needed
for w in 0.5 1 2 4; do
  for chain in 0 1 2 3; do
    python3 scripts/run_single_chain.py --width $w --h 1e-5 --chain $chain --n_train 1024
  done
done
```

### After all chains for a given (width, h)

Run analysis and plots on the run dirs for that (width, h):

```bash
# B_t fidelity (violation rate)
python3 experiments/analysis/compute_bt_fidelity.py experiments/runs/w1_n1024_h1e-5_chain0 experiments/runs/w1_n1024_h1e-5_chain1 experiments/runs/w1_n1024_h1e-5_chain2 experiments/runs/w1_n1024_h1e-5_chain3 --B 50000 -o experiments/summaries/bt_fidelity.csv

# Convergence (Rhat, ESS, ESS-rate)
python3 experiments/analysis/compute_convergence.py experiments/runs/w1_n1024_h1e-5_chain{0,1,2,3} --B 50000 --S 200 -o experiments/summaries/convergence.csv

# Proxy LSI
python3 experiments/analysis/compute_lsi_proxy.py experiments/runs/w1_n1024_h1e-5_chain{0,1,2,3} --B 50000 --G 5 --S 200 -o experiments/summaries/lsi_proxy.csv

# Plots (use run dirs and summary CSVs you produced)
python3 experiments/analysis/make_plots.py --run-dirs experiments/runs/w1_n1024_h1e-5_chain0 ... --bt-csv experiments/summaries/bt_fidelity.csv --convergence-csv experiments/summaries/convergence.csv --lsi-csv experiments/summaries/lsi_proxy.csv -o experiments/figures
```

You can aggregate summaries per (width, h) and then plot across widths (e.g. ESS-rate vs width, rho_hat vs width) by editing the plot script or merging CSVs with a `width`/`h` column.

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
- Run the analysis scripts (compute_bt_fidelity, compute_convergence, compute_lsi_proxy) with the appropriate run dirs for each (width, h).
- Run `make_plots.py` with the summary CSVs and run dirs to produce figures.

### 6. Optional: checkpointing / resume

The current implementation does not checkpoint mid-run. For very long runs (T=200k), consider adding checkpointing (save model state and step counter periodically and add a `--resume` path) or splitting into multiple segments.

---

## Quick reference

| What              | Command / path |
|-------------------|----------------|
| Smoke run         | `python3 scripts/smoke_run.py` |
| One chain         | `python3 scripts/run_single_chain.py --width 1 --h 1e-5 --chain 0 --n_train 1024` |
| Unit tests        | `python3 -m unittest discover tests -v` |
| Run outputs       | `experiments/runs/<run_name>/` (run_config.yaml, iter_metrics.jsonl, samples_metrics.npz) |
| Summaries        | `experiments/summaries/*.csv` |
| Figures          | `experiments/figures/*.png` (from `make_plots.py`) |
