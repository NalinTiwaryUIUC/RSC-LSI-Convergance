#!/bin/bash
#SBATCH --job-name=rsc_pretrain
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/pretrain/pretrain_%j.out
#SBATCH --error=logs/pretrain/pretrain_%j.err

#
# Run SGD pretraining (scripts/pretrain.py) with logging. Use from project root.
#
# Pass any pretrain.py arguments after the script; they override env defaults.
# Comma-separated flags (--snapshot-steps) are safest as trailing CLI args (Slurm --export
# breaks on commas inside values).
#
# Env defaults (optional):
#   WIDTH                 --width (default: 1.0)
#   N_TRAIN               --n_train (default: 1024)
#   ALPHA                 --alpha (pretrain.py default: 0.3; MAP uses WD=α when PRETRAIN_WEIGHT_DECAY=-1)
#   PRETRAIN_STEPS        --pretrain-steps (default: 2000)
#   PRETRAIN_LR           --pretrain-lr (pretrain.py default: 0.01; try 0.005 / 0.01 / 0.02)
#   PRETRAIN_WEIGHT_DECAY --pretrain-weight-decay (-1 = use α; 0 = no L2)
#   OUTPUT                -o/--output (default: experiments/checkpoints/pretrain_w{W}_n{N}_nb{NUM_BLOCKS}.pt)
#   ARCH                  --arch (default: resnet18)
#   NUM_BLOCKS            --num-blocks (default: 2)
#   SNAPSHOT_STEPS        --snapshot-steps (comma list; also pass as CLI if using sbatch --export)
#   SNAPSHOT_EVERY        --snapshot-every (pretrain.py default: 25; use 0 to disable periodic snaps)
#   SNAPSHOT_DIR          --snapshot-dir (intermediate *_step*.pt; default experiments/checkpoints)
#   DATA_DIR              --data_dir
#   ROOT                  --root (default: ./data)
#   DATASET_SEED          --dataset-seed (default: 42)
#   PRETRAIN_SEED         --pretrain-seed (default: 42)
#   BN_CALIBRATION_MB     --bn-calibration-microbatch (default: 256)
#   VERIFY                set to 1 for --verify
#   LOG_DIR               directory for log file (default: logs/pretrain)
#
# Examples (local):
#   ./scripts/run_pretrain.sh --width 0.1 --n_train 1024
#   ./scripts/run_pretrain.sh --snapshot-steps 500,1000,1500 --snapshot-dir experiments/checkpoints/my_snaps -o experiments/checkpoints/out.pt
#
# SBATCH — pretrain w=1 for escape diagnostic (snapshots for I2); run from project root:
#   mkdir -p logs/pretrain experiments/checkpoints/escape_diag_run01/snaps_w1
#   sbatch scripts/run_pretrain.sh --width 1 --n_train 512 --alpha 0.3 --pretrain-steps 4000 --pretrain-lr 0.01 \
#     --arch small_resnet_ln --num-blocks 1 --data_dir experiments/data --root ./data \
#     --snapshot-dir experiments/checkpoints/escape_diag_run01/snaps_w1 --snapshot-every 25 \
#     --snapshot-steps 800,1200,1600,2000,2400,3200 \
#     -o experiments/checkpoints/escape_diag_run01/pretrain_w1_n512_nb1_final.pt
#
# SBATCH — pretrain w=4 (submit after w=1 or in parallel if you have two GPUs):
#   mkdir -p experiments/checkpoints/escape_diag_run01/snaps_w4
#   sbatch scripts/run_pretrain.sh --width 4 --n_train 512 --alpha 0.3 --pretrain-steps 4000 --pretrain-lr 0.01 \
#     --arch small_resnet_ln --num-blocks 1 --data_dir experiments/data --root ./data \
#     --snapshot-dir experiments/checkpoints/escape_diag_run01/snaps_w4 --snapshot-every 25 \
#     --snapshot-steps 800,1200,1600,2000,2400,3200 \
#     -o experiments/checkpoints/escape_diag_run01/pretrain_w4_n512_nb1_final.pt
#
set -euo pipefail

# Project root
if [ -n "$RSC_CONV_DIR" ]; then
    PROJ_DIR="$RSC_CONV_DIR"
elif [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJ_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJ_DIR" || { echo "ERROR: Cannot cd to $PROJ_DIR"; exit 1; }

LOG_DIR="${LOG_DIR:-logs/pretrain}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pretrain_${TIMESTAMP}.log"
echo "Log file: $LOG_FILE" | tee "$LOG_FILE"

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

# Activate venv first so Python/PyTorch checks and all commands use it
if [ -d ".venv" ]; then
    source .venv/bin/activate
    log "Using .venv"
elif [ -d "venv" ]; then
    source venv/bin/activate
    log "Using venv"
fi

# Build base args from env (so command-line args override)
ARGS=()
[ -n "$WIDTH" ]              && ARGS+=(--width "$WIDTH")
[ -n "$N_TRAIN" ]            && ARGS+=(--n_train "$N_TRAIN")
[ -n "$ALPHA" ]              && ARGS+=(--alpha "$ALPHA")
[ -n "$PRETRAIN_STEPS" ]     && ARGS+=(--pretrain-steps "$PRETRAIN_STEPS")
[ -n "$PRETRAIN_LR" ]        && ARGS+=(--pretrain-lr "$PRETRAIN_LR")
[ -n "$PRETRAIN_WEIGHT_DECAY" ] && ARGS+=(--pretrain-weight-decay "$PRETRAIN_WEIGHT_DECAY")
[ -n "$OUTPUT" ]             && ARGS+=(--output "$OUTPUT")
[ -n "$ARCH" ]               && ARGS+=(--arch "$ARCH")
[ -n "$NUM_BLOCKS" ]         && ARGS+=(--num-blocks "$NUM_BLOCKS")
[ -n "$SNAPSHOT_STEPS" ]     && ARGS+=(--snapshot-steps "$SNAPSHOT_STEPS")
[ -n "$SNAPSHOT_EVERY" ]     && ARGS+=(--snapshot-every "$SNAPSHOT_EVERY")
[ -n "$SNAPSHOT_DIR" ]       && ARGS+=(--snapshot-dir "$SNAPSHOT_DIR")
[ -n "$DATA_DIR" ]           && ARGS+=(--data_dir "$DATA_DIR")
[ -n "$ROOT" ]               && ARGS+=(--root "$ROOT")
[ -n "$DATASET_SEED" ]       && ARGS+=(--dataset-seed "$DATASET_SEED")
[ -n "$PRETRAIN_SEED" ]      && ARGS+=(--pretrain-seed "$PRETRAIN_SEED")
[ -n "$BN_CALIBRATION_MB" ]  && ARGS+=(--bn-calibration-microbatch "$BN_CALIBRATION_MB")
[ "${VERIFY:-0}" = "1" ]     && ARGS+=(--verify)

log "=== Pretrain run started at $(date) ==="
log "=== Working directory: $PROJ_DIR ==="
log "=== Python ==="
python3 --version 2>&1 | tee -a "$LOG_FILE" || true
log "=== PyTorch / GPU ==="
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
" 2>&1 | tee -a "$LOG_FILE" || true

CMD=(python3 scripts/pretrain.py "${ARGS[@]}" "$@")
log ""
log "=============================================="
log "Command: ${CMD[*]}"
log "=============================================="

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
r=${PIPESTATUS[0]}

log ""
log "=== Pretrain finished at $(date) ==="
log "Exit code: $r"

if [ "$r" -ne 0 ]; then
    log "FAIL: pretrain.py exited with $r. Check $LOG_FILE for errors."
    exit "$r"
fi
log "SUCCESS: checkpoint written (see script output above)."
exit 0
