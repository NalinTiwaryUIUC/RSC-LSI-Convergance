#!/bin/bash
#SBATCH --job-name=bayeslin_lsi
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
#SBATCH --output=logs/bayeslin_lsi_width/bayeslin_lsi_%j.out
#SBATCH --error=logs/bayeslin_lsi_width/bayeslin_lsi_%j.err
#
# Bayesian linear LSI width experiment (CPU NumPy; GPU line kept so the job matches
# eng-research-gpu like run_single_chain.sh). Submit from repo root, or set RSC_CONV_DIR.
#
# Mode (default: pilot):
#   export MODE=pilot   # widths 32,64,128,256 seeds 0,1,2 T_phys=10 -> pilot_seed*
#   export MODE=main    # widths 32..512 seeds 0..9 sigma=10 T_phys=10 log_dt=0.005 -> main_seed*
#   export MODE=plot_pilot
#   export MODE=plot_main
#
# Examples:
#   mkdir -p logs/bayeslin_lsi_width experiments/bayeslin_lsi_width
#   sbatch scripts/run_bayeslin_lsi_width.sh
#   MODE=main sbatch scripts/run_bayeslin_lsi_width.sh
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

LOG_DIR="${LOG_DIR:-logs/bayeslin_lsi_width}"
mkdir -p "$LOG_DIR" experiments/bayeslin_lsi_width

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
        python3 scripts/bayeslin_lsi_width_convergence.py \
            --widths 32,64,128,256 --n-over-m 4 --alpha 0.3 --sigma 1.0 --teacher-scale 1.0 \
            --h-factor 0.05 --T-phys 10.0 --log-dt 0.02 \
            --seeds 0,1,2 --out-dir experiments/bayeslin_lsi_width/pilot
        ;;
    main)
        python3 scripts/bayeslin_lsi_width_convergence.py \
            --widths 32,64,128,256,512 --n-over-m 4 --alpha 0.3 --sigma 10.0 --teacher-scale 1.0 \
            --h-factor 0.05 --T-phys 10.0 --log-dt 0.005 \
            --seeds 0,1,2,3,4,5,6,7,8,9 --out-dir experiments/bayeslin_lsi_width/main
        ;;
    plot_pilot)
        python3 scripts/plot_bayeslin_lsi_width.py \
            --run-glob "experiments/bayeslin_lsi_width/pilot_seed*" \
            --plot-out-dir experiments/bayeslin_lsi_width/pilot_plots
        ;;
    plot_main)
        python3 scripts/plot_bayeslin_lsi_width.py \
            --run-glob "experiments/bayeslin_lsi_width/main_seed*" \
            --plot-out-dir experiments/bayeslin_lsi_width/main_plots
        ;;
    *)
        echo "Unknown MODE=$MODE (use pilot|main|plot_pilot|plot_main)"
        exit 1
        ;;
esac

echo "Done $(date)"
