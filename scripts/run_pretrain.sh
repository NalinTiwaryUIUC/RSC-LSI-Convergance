#!/bin/bash
#
# Run SGD pretraining (scripts/pretrain.py) with logging. Use from project root.
#
# Pass any pretrain.py arguments after the script; they override env defaults.
#
# Env defaults (optional):
#   WIDTH              --width (default: 1.0)
#   N_TRAIN            --n_train (default: 1024)
#   ALPHA              --alpha (default: 0.1)
#   PRETRAIN_STEPS     --pretrain-steps (default: 2000)
#   PRETRAIN_LR        --pretrain-lr (default: 0.02)
#   OUTPUT             -o/--output (default: experiments/checkpoints/pretrain_w{W}_n{N}.pt)
#   ARCH               --arch (default: resnet18)
#   NUM_BLOCKS         --num-blocks (default: 2)
#   DATA_DIR           --data_dir
#   ROOT               --root (default: ./data)
#   DATASET_SEED       --dataset-seed (default: 42)
#   PRETRAIN_SEED      --pretrain-seed (default: 42)
#   BN_CALIBRATION_MB  --bn-calibration-microbatch (default: 256)
#   VERIFY             set to 1 for --verify
#   LOG_DIR            directory for log file (default: logs/pretrain)
#
# Examples:
#   ./scripts/run_pretrain.sh --width 0.1 --n_train 1024
#   ./scripts/run_pretrain.sh --width 0.1 --n_train 1024 --alpha 0.1 -o experiments/checkpoints/pretrain_w0.1_n1024.pt
#   WIDTH=0.1 N_TRAIN=1024 ./scripts/run_pretrain.sh
#   PRETRAIN_STEPS=500 ./scripts/run_pretrain.sh --width 0.5 --n_train 2048
#
set -e

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

# Activate venv if present
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
[ -n "$OUTPUT" ]             && ARGS+=(--output "$OUTPUT")
[ -n "$ARCH" ]               && ARGS+=(--arch "$ARCH")
[ -n "$NUM_BLOCKS" ]         && ARGS+=(--num-blocks "$NUM_BLOCKS")
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
