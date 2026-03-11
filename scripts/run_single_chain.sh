#!/bin/bash
#
# Wrapper around scripts/run_single_chain.py with logging. Use from project root.
#
# Pass any run_single_chain.py arguments after the script; they override env defaults.
#
# Env defaults (optional):
#   WIDTH              --width
#   H                  --h (step size)
#   CHAIN              --chain (chain id)
#   N_TRAIN            --n_train
#   PROBE_SIZE         --probe_size
#   T                  --T (total steps)
#   B                  --B (burn-in steps)
#   S                  --S (save stride)
#   LOG_EVERY          --log-every
#   PRETRAIN_STEPS     --pretrain-steps
#   PRETRAIN_LR        --pretrain-lr
#   PRETRAIN_PATH      --pretrain-path
#   BN_MODE            --bn-mode (eval | batchstat_frozen)
#   BN_CALIBRATION_STEPS  --bn-calibration-steps
#   DATA_DIR           --data_dir
#   RUNS_DIR           --runs_dir
#   ROOT               --root
#   DATASET_SEED       --dataset-seed
#   CHAIN_SEED         --chain-seed
#   PROBE_PROJECTION_SEED  --probe-projection-seed
#   DEVICE             --device (cuda, cuda:0, cpu, or empty for auto)
#   NOISE_SCALE        --noise-scale
#   ALPHA              --alpha
#   CE_REDUCTION       --ce-reduction (mean | sum)
#   CLIP_GRAD_NORM     --clip-grad-norm
#   MICROBATCH_SIZE    --microbatch-size
#   ARCH               --arch (resnet18 | small_resnet_ln)
#   NUM_BLOCKS         --num-blocks
#   LOG_DIR            directory for log file (default: logs/chain)
#
# Examples:
#   ./scripts/run_single_chain.sh --width 1 --h 1e-5 --chain 0 --n_train 1024
#   ./scripts/run_single_chain.sh --width 0.5 --chain 0 --pretrain-path experiments/checkpoints/pretrain_w0.5_n1024.pt
#   WIDTH=1 H=1e-5 CHAIN=0 ./scripts/run_single_chain.sh
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

LOG_DIR="${LOG_DIR:-logs/chain}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Include width and chain in log name when set (from env or we'll use placeholder)
LOG_NAME="chain_${TIMESTAMP}.log"
LOG_FILE="$LOG_DIR/$LOG_NAME"
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

# Build base args from env (command-line args override)
ARGS=()
[ -n "$WIDTH" ]                  && ARGS+=(--width "$WIDTH")
[ -n "$H" ]                      && ARGS+=(--h "$H")
[ -n "$CHAIN" ]                  && ARGS+=(--chain "$CHAIN")
[ -n "$N_TRAIN" ]                && ARGS+=(--n_train "$N_TRAIN")
[ -n "$PROBE_SIZE" ]             && ARGS+=(--probe_size "$PROBE_SIZE")
[ -n "$T" ]                      && ARGS+=(--T "$T")
[ -n "$B" ]                      && ARGS+=(--B "$B")
[ -n "$S" ]                      && ARGS+=(--S "$S")
[ -n "$LOG_EVERY" ]              && ARGS+=(--log-every "$LOG_EVERY")
[ -n "$PRETRAIN_STEPS" ]         && ARGS+=(--pretrain-steps "$PRETRAIN_STEPS")
[ -n "$PRETRAIN_LR" ]            && ARGS+=(--pretrain-lr "$PRETRAIN_LR")
[ -n "$PRETRAIN_PATH" ]          && ARGS+=(--pretrain-path "$PRETRAIN_PATH")
[ -n "$BN_MODE" ]                && ARGS+=(--bn-mode "$BN_MODE")
[ -n "$BN_CALIBRATION_STEPS" ]   && ARGS+=(--bn-calibration-steps "$BN_CALIBRATION_STEPS")
[ -n "$DATA_DIR" ]               && ARGS+=(--data_dir "$DATA_DIR")
[ -n "$RUNS_DIR" ]               && ARGS+=(--runs_dir "$RUNS_DIR")
[ -n "$ROOT" ]                   && ARGS+=(--root "$ROOT")
[ -n "$DATASET_SEED" ]           && ARGS+=(--dataset-seed "$DATASET_SEED")
[ -n "$CHAIN_SEED" ]             && ARGS+=(--chain-seed "$CHAIN_SEED")
[ -n "$PROBE_PROJECTION_SEED" ]  && ARGS+=(--probe-projection-seed "$PROBE_PROJECTION_SEED")
[ -n "$DEVICE" ]                 && ARGS+=(--device "$DEVICE")
[ -n "$NOISE_SCALE" ]            && ARGS+=(--noise-scale "$NOISE_SCALE")
[ -n "$ALPHA" ]                  && ARGS+=(--alpha "$ALPHA")
[ -n "$CE_REDUCTION" ]           && ARGS+=(--ce-reduction "$CE_REDUCTION")
[ -n "$CLIP_GRAD_NORM" ]         && ARGS+=(--clip-grad-norm "$CLIP_GRAD_NORM")
[ -n "$MICROBATCH_SIZE" ]        && ARGS+=(--microbatch-size "$MICROBATCH_SIZE")
[ -n "$ARCH" ]                   && ARGS+=(--arch "$ARCH")
[ -n "$NUM_BLOCKS" ]             && ARGS+=(--num-blocks "$NUM_BLOCKS")

log "=== Chain run started at $(date) ==="
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

CMD=(python3 scripts/run_single_chain.py "${ARGS[@]}" "$@")
log ""
log "=============================================="
log "Command: ${CMD[*]}"
log "=============================================="

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
r=${PIPESTATUS[0]}

log ""
log "=== Chain run finished at $(date) ==="
log "Exit code: $r"

if [ "$r" -ne 0 ]; then
    log "FAIL: run_single_chain.py exited with $r. Check $LOG_FILE for errors."
    exit "$r"
fi
log "SUCCESS: chain completed. See runs_dir for run_config.yaml, iter_metrics.jsonl, samples_metrics.npz."
exit 0
