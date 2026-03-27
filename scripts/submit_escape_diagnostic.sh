#!/bin/bash
#SBATCH --job-name=escape_diag
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/escape_diag/escape_diag_%j.out
#SBATCH --error=logs/escape_diag/escape_diag_%j.err
#
# Escape-time diagnostic wrapper (I1 / I2 / I3 init). Run from project root.
#
# Recommended: LOG_EVERY=1 (or 2) so τ_escape from iter_metrics is not quantized by coarse logging.
#
# mkdir -p logs/escape_diag experiments/checkpoints/escape_diag_run01
#
# One-line SBATCH (w=1, chain 0; bump CHAIN and WIDTH=4 + w4 paths for full grid). Replace STEP in I2.
# I1:
#   sbatch --export=ALL,INIT=I1,WIDTH=1,CHAIN=0,T=100000,B=0,S=20,LOG_EVERY=1,N_TRAIN=512,PROBE_SIZE=512,PRETRAIN_STEPS=0,PRETRAIN_LR=0.02,ALPHA=0.3,BETA=1.0,SAMPLER=underdamped,GAMMA=3.0,ARCH=small_resnet_ln,NUM_BLOCKS=1,DATA_DIR=experiments/data,RUNS_DIR=experiments/runs,ROOT=./data,CE_REDUCTION=sum,PRETRAIN_FINAL=experiments/checkpoints/escape_diag_run01/pretrain_w1_n512_nb1_final.pt scripts/submit_escape_diagnostic.sh
# I2:
#   sbatch --export=ALL,INIT=I2,WIDTH=1,CHAIN=0,T=100000,B=0,S=20,LOG_EVERY=1,N_TRAIN=512,PROBE_SIZE=512,PRETRAIN_STEPS=0,PRETRAIN_LR=0.02,ALPHA=0.3,BETA=1.0,SAMPLER=underdamped,GAMMA=3.0,ARCH=small_resnet_ln,NUM_BLOCKS=1,DATA_DIR=experiments/data,RUNS_DIR=experiments/runs,ROOT=./data,CE_REDUCTION=sum,PRETRAIN_EARLY=experiments/checkpoints/escape_diag_run01/snaps_w1/pretrain_w1_n512_nb1_stepSTEP.pt scripts/submit_escape_diagnostic.sh
# I3:
#   sbatch --export=ALL,INIT=I3,WIDTH=1,CHAIN=0,T=100000,B=0,S=20,LOG_EVERY=1,N_TRAIN=512,PROBE_SIZE=512,PRETRAIN_STEPS=0,PRETRAIN_LR=0.02,ALPHA=0.3,BETA=1.0,SAMPLER=underdamped,GAMMA=3.0,ARCH=small_resnet_ln,NUM_BLOCKS=1,DATA_DIR=experiments/data,RUNS_DIR=experiments/runs,ROOT=./data,CE_REDUCTION=sum,PRETRAIN_FINAL=experiments/checkpoints/escape_diag_run01/pretrain_w1_n512_nb1_final.pt,INIT_PERTURB_SIGMA=0.02,INIT_PERTURB_REFERENCE=checkpoint scripts/submit_escape_diagnostic.sh
#
# Export style:
#   export WIDTH=1 T=100000 B=0 S=20 LOG_EVERY=1 CHAIN=0 INIT=I1 CE_REDUCTION=sum
#   export PRETRAIN_FINAL=experiments/checkpoints/escape_diag_run01/pretrain_w1_n512_nb1_final.pt
#   sbatch scripts/submit_escape_diagnostic.sh
#
# Job array (3 inits × 4 chains):
#   #SBATCH --array=0-11
#   INIT_LIST=(I1 I1 I1 I1 I2 I2 I2 I2 I3 I3 I3 I3)
#   CHAIN_LIST=(0 1 2 3 0 1 2 3 0 1 2 3)
#   export INIT="${INIT_LIST[$SLURM_ARRAY_TASK_ID]}"
#   export CHAIN="${CHAIN_LIST[$SLURM_ARRAY_TASK_ID]}"

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

LOG_DIR="${LOG_DIR:-logs/escape_diag}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/escape_diag_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log}"

# --- Defaults (override via env) ---
WIDTH="${WIDTH:-1}"
H="${H:-5e-6}"
CHAIN="${CHAIN:-0}"
N_TRAIN="${N_TRAIN:-512}"
PROBE_SIZE="${PROBE_SIZE:-256}"
T="${T:-40000}"
B="${B:-0}"
S="${S:-20}"
LOG_EVERY="${LOG_EVERY:-1}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-0}"
PRETRAIN_LR="${PRETRAIN_LR:-0.02}"
DATA_DIR="${DATA_DIR:-experiments/data}"
RUNS_DIR="${RUNS_DIR:-experiments/runs}"
ROOT="${ROOT:-./data}"
DATASET_SEED="${DATASET_SEED:-42}"
CHAIN_SEED="${CHAIN_SEED:-}"
DEVICE="${DEVICE:-}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
ALPHA="${ALPHA:-0.3}"
BETA="${BETA:-1.0}"
SAMPLER="${SAMPLER:-underdamped}"
GAMMA="${GAMMA:-3.0}"
ARCH="${ARCH:-resnet18}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
BN_MODE="${BN_MODE:-eval}"
BN_CALIBRATION_STEPS="${BN_CALIBRATION_STEPS:-256}"
CE_REDUCTION="${CE_REDUCTION:-}"

# Init regime: I1 | I2 | I3
INIT="${INIT:-I1}"
PRETRAIN_FINAL="${PRETRAIN_FINAL:-}"
PRETRAIN_EARLY="${PRETRAIN_EARLY:-}"
INIT_PERTURB_SIGMA="${INIT_PERTURB_SIGMA:-0.0}"
INIT_PERTURB_REFERENCE="${INIT_PERTURB_REFERENCE:-checkpoint}"
RUN_SUFFIX_EXTRA="${RUN_SUFFIX_EXTRA:-}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

PRETRAIN_PATH=""
RUN_SUFFIX=""

case "$INIT" in
  I1)
    PRETRAIN_PATH="$PRETRAIN_FINAL"
    RUN_SUFFIX="${RUN_SUFFIX_EXTRA:+${RUN_SUFFIX_EXTRA}_}initI1"
    ;;
  I2)
    PRETRAIN_PATH="${PRETRAIN_EARLY:-}"
    STEP_TAG=""
    if [[ -n "$PRETRAIN_PATH" ]]; then
      base="$(basename "$PRETRAIN_PATH")"
      if [[ "$base" =~ _step([0-9]+)\.pt$ ]]; then
        STEP_TAG="step${BASH_REMATCH[1]}"
      fi
    fi
    RUN_SUFFIX="${RUN_SUFFIX_EXTRA:+${RUN_SUFFIX_EXTRA}_}initI2${STEP_TAG:+_${STEP_TAG}}"
    ;;
  I3)
    PRETRAIN_PATH="$PRETRAIN_FINAL"
    RUN_SUFFIX="${RUN_SUFFIX_EXTRA:+${RUN_SUFFIX_EXTRA}_}initI3_sigma$(echo "$INIT_PERTURB_SIGMA" | tr '.' 'p')"
    ;;
  *)
    log "ERROR: INIT must be I1, I2, or I3, got $INIT"
    exit 1
    ;;
esac

if [[ -z "$PRETRAIN_PATH" ]]; then
  log "ERROR: set PRETRAIN_FINAL (I1/I3) or PRETRAIN_EARLY / PRETRAIN_PATH (I2)"
  exit 1
fi

if [[ -d ".venv" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
elif [[ -d "venv" ]]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

ARGS=()
[[ -n "$WIDTH" ]] && ARGS+=(--width "$WIDTH")
[[ -n "$H" ]] && ARGS+=(--h "$H")
[[ -n "$CHAIN" ]] && ARGS+=(--chain "$CHAIN")
[[ -n "$N_TRAIN" ]] && ARGS+=(--n_train "$N_TRAIN")
[[ -n "$PROBE_SIZE" ]] && ARGS+=(--probe_size "$PROBE_SIZE")
[[ -n "$T" ]] && ARGS+=(--T "$T")
[[ -n "$B" ]] && ARGS+=(--B "$B")
[[ -n "$S" ]] && ARGS+=(--S "$S")
[[ -n "$LOG_EVERY" ]] && ARGS+=(--log-every "$LOG_EVERY")
[[ -n "$PRETRAIN_STEPS" ]] && ARGS+=(--pretrain-steps "$PRETRAIN_STEPS")
[[ -n "$PRETRAIN_LR" ]] && ARGS+=(--pretrain-lr "$PRETRAIN_LR")
[[ -n "$PRETRAIN_PATH" ]] && ARGS+=(--pretrain-path "$PRETRAIN_PATH")
[[ -n "$BN_MODE" ]] && ARGS+=(--bn-mode "$BN_MODE")
[[ -n "$BN_CALIBRATION_STEPS" ]] && ARGS+=(--bn-calibration-steps "$BN_CALIBRATION_STEPS")
[[ -n "$DATA_DIR" ]] && ARGS+=(--data_dir "$DATA_DIR")
[[ -n "$RUNS_DIR" ]] && ARGS+=(--runs_dir "$RUNS_DIR")
[[ -n "$ROOT" ]] && ARGS+=(--root "$ROOT")
[[ -n "$DATASET_SEED" ]] && ARGS+=(--dataset-seed "$DATASET_SEED")
[[ -n "$CHAIN_SEED" ]] && ARGS+=(--chain-seed "$CHAIN_SEED")
[[ -n "$DEVICE" ]] && ARGS+=(--device "$DEVICE")
[[ -n "$NOISE_SCALE" ]] && ARGS+=(--noise-scale "$NOISE_SCALE")
[[ -n "$ALPHA" ]] && ARGS+=(--alpha "$ALPHA")
[[ -n "$BETA" ]] && ARGS+=(--beta "$BETA")
[[ -n "$ARCH" ]] && ARGS+=(--arch "$ARCH")
[[ -n "$NUM_BLOCKS" ]] && ARGS+=(--num-blocks "$NUM_BLOCKS")
[[ -n "$SAMPLER" ]] && ARGS+=(--sampler "$SAMPLER")
[[ -n "$GAMMA" ]] && ARGS+=(--gamma "$GAMMA")
[[ -n "$CE_REDUCTION" ]] && ARGS+=(--ce-reduction "$CE_REDUCTION")
[[ -n "$INIT_PERTURB_SIGMA" ]] && ARGS+=(--init-perturb-sigma "$INIT_PERTURB_SIGMA")
[[ -n "$INIT_PERTURB_REFERENCE" ]] && ARGS+=(--init-perturb-reference "$INIT_PERTURB_REFERENCE")
[[ -n "$RUN_SUFFIX" ]] && ARGS+=(--run-suffix "$RUN_SUFFIX")

log "=== Escape diagnostic chain ==="
log "INIT=$INIT PRETRAIN_PATH=$PRETRAIN_PATH"
log "Command: python3 scripts/run_single_chain.py ${ARGS[*]} $*"

python3 scripts/run_single_chain.py "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
r=${PIPESTATUS[0]}
log "Exit code: $r"
exit "$r"
