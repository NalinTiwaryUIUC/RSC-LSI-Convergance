#!/bin/bash
#SBATCH --job-name=lsi_ula
#SBATCH --time=48:00:00                    # Job run time (hh:mm:ss) - 48h for full T=200k chain
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu"  # Email when job starts/finishes/fails
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/lsi_ula/lsi_ula_%j.out
#SBATCH --error=logs/lsi_ula/lsi_ula_%j.err

# Run one ULA chain. IMPORTANT: Submit from the project directory (cd /path/to/RSC_Conv first).
# Arguments: WIDTH  H  CHAIN  N_TRAIN
#   CHAIN = 0, 1, 2, or 3 (four chains per width/h)
# Examples:
#   sbatch scripts/submit_chain.sh              # defaults: width=1, h=1e-5, chain=0, n_train=1024
#   sbatch scripts/submit_chain.sh 1 1e-5 0 1024   # chain 0
#   sbatch scripts/submit_chain.sh 1 1e-5 2 1024   # chain 2
#   sbatch scripts/submit_chain.sh 1 1e-5 3 1024   # chain 3
#   for w in 0.5 1 2 4; do for c in 0 1 2 3; do sbatch scripts/submit_chain.sh $w 1e-5 $c 1024; done; done

WIDTH=${1:-1}
H=${2:-1e-5}
CHAIN=${3:-0}      # 0, 1, 2, or 3
N_TRAIN=${4:-1024}

# Project root: use SLURM submission dir (you must run sbatch from the project directory)
# Override with RSC_CONV_DIR if submitting from elsewhere
if [ -n "$RSC_CONV_DIR" ]; then
    PROJ_DIR="$RSC_CONV_DIR"
elif [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJ_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJ_DIR" || { echo "Failed to cd to $PROJ_DIR"; exit 1; }
# Ensure logs dir exists (output/error paths are relative to submission dir)
mkdir -p logs/lsi_ula

echo "=== Job started at $(date) ==="
echo "=== Job ID: $SLURM_JOB_ID ==="
echo "=== Parameters: width=$WIDTH h=$H chain=$CHAIN n_train=$N_TRAIN ==="
echo "=== Working directory: $PROJ_DIR ==="

echo "=== GPU Information ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo "=== Memory Information ==="
free -h 2>/dev/null || true
echo "=== Python Version ==="
python3 --version

echo "=== Setting up environment ==="
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No .venv or venv found in $PROJ_DIR"
    exit 1
fi

python3 -m pip install --upgrade pip --quiet
echo "=== Installing dependencies ==="
pip3 install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "pip install -r requirements.txt failed"
    exit 1
fi

echo "=== Verifying PyTorch and GPU ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"
if [ $? -ne 0 ]; then
    echo "PyTorch check failed"
    exit 1
fi

echo "=== Starting ULA chain ==="
python3 scripts/run_single_chain.py \
    --width "$WIDTH" \
    --h "$H" \
    --chain "$CHAIN" \
    --n_train "$N_TRAIN" \
    --data_dir experiments/data \
    --runs_dir experiments/runs \
    --root ./data

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Chain completed successfully at $(date) ==="
    echo "Output: experiments/runs/w${WIDTH}_n${N_TRAIN}_h${H}_chain${CHAIN}/"
else
    echo "=== Chain failed with exit code $EXIT_CODE at $(date) ==="
    exit $EXIT_CODE
fi
