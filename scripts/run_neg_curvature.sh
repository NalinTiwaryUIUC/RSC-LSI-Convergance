#!/bin/bash
#SBATCH --job-name=neg_curv
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/neg_curv/neg_curv_%j.out
#SBATCH --error=logs/neg_curv/neg_curv_%j.err

#
# Negative-curvature experiment (NME spectrum on small_resnet_ln).
# Implements minimal_negative_curvature_experiment.md.
#
# Submit from repo root or set RSC_CONV_DIR.
#
# Mode (default: pilot):
#   export MODE=pilot       # widths 1,2,4 seeds 0 final-only top-20 no-SLQ no-local
#   export MODE=main        # widths 1,2,4 seeds 0,1,2 ckpts {init,mid,final} SLQ + local
#   export MODE=plot_pilot
#   export MODE=plot_main
#
# Examples:
#   mkdir -p logs/neg_curv experiments/neg_curv
#   MODE=pilot sbatch scripts/run_neg_curvature.sh
#   MODE=main  sbatch scripts/run_neg_curvature.sh
#   MODE=plot_main sbatch scripts/run_neg_curvature.sh
#
set -euo pipefail

if [ -n "${RSC_CONV_DIR:-}" ]; then
    PROJ_DIR="$RSC_CONV_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJ_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJ_DIR" || { echo "ERROR: Cannot cd to $PROJ_DIR"; exit 1; }

LOG_DIR="${LOG_DIR:-logs/neg_curv}"
mkdir -p "$LOG_DIR" experiments/neg_curv

if [ -d ".venv" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
    echo "Using .venv"
elif [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
    echo "Using venv"
fi

MODE="${MODE:-pilot}"
echo "MODE=$MODE  PROJ_DIR=$PROJ_DIR  $(date)"

case "$MODE" in
    pilot)
        python3 scripts/run_neg_curvature.py \
            --widths 1,2,4 --seeds 0 \
            --arch small_resnet_ln --num-blocks 1 \
            --n-train 512 --n-curv 128 \
            --lr 0.02 --momentum 0.9 --weight-decay 0 --max-steps 1000 --mid-step 500 \
            --checkpoints final \
            --num-neg 20 --lanczos-steps 80 --ncv -1 \
            --no-slq --no-local-check \
            --dtype float32 \
            --out-dir experiments/neg_curv/pilot
        ;;
    main)
        python3 scripts/run_neg_curvature.py \
            --widths 1,2,4 --seeds 0,1,2 \
            --arch small_resnet_ln --num-blocks 1 \
            --n-train 512 --n-curv 128 \
            --lr 0.02 --momentum 0.9 --weight-decay 0 --max-steps 1000 --mid-step 500 \
            --checkpoints init,mid,final \
            --num-neg 20 --lanczos-steps 80 --ncv -1 \
            --slq --num-probes 8 --slq-steps 30 \
            --local-check --num-local 5 --eps-rel 0.01 --num-local-neg 10 \
            --dtype float32 \
            --out-dir experiments/neg_curv/main
        ;;
    plot_pilot)
        python3 scripts/plot_neg_curvature.py \
            --run-glob "experiments/neg_curv/pilot_seed*" \
            --plot-out-dir experiments/neg_curv/pilot_plots \
            --checkpoint final
        ;;
    plot_main)
        for CKPT in init mid final; do
            python3 scripts/plot_neg_curvature.py \
                --run-glob "experiments/neg_curv/main_seed*" \
                --plot-out-dir "experiments/neg_curv/main_plots_${CKPT}" \
                --checkpoint "$CKPT"
        done
        ;;
    *)
        echo "Unknown MODE=$MODE (use pilot|main|plot_pilot|plot_main)"
        exit 1
        ;;
esac

echo "Done $(date)"
