#!/usr/bin/env bash
# Run on the cluster from the RSC_Conv project root (after module load / venv as you usually do).
# Produces:
#   - experiments/summaries/escape_w1_chain_convergence_report.md   (R̂, ESS, late-window probes, iter_metrics)
#   - experiments/summaries/escape_w1_chain_convergence_summary.csv
#   - experiments/summaries/escape_w1_chain_convergence_summary_late.csv
#   - experiments/summaries/escape_w1_tau_escape.csv                 (τ_escape + aligned R̂ from analyze_escape_diagnostic)
#   - experiments/summaries/escape_w1_init_comparison.md              (pooled I1 vs I2 vs I3 iter trends + prior ratios)
#   - experiments/summaries/escape_w1_init_comparison.csv
#
# Override globs if your run names differ (e.g. different h, T, n_train):
#   export RUN_GLOB='w1_n512_h5e-06_T100000*_ul_initI*_chain*'
#   ./scripts/cluster_summarize_escape_w1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

RUN_GLOB="${RUN_GLOB:-w1_*_ul_initI*_chain*}"
SUMMARY_DIR="${SUMMARY_DIR:-experiments/summaries}"
mkdir -p "$SUMMARY_DIR"

PY="${PYTHON:-python3}"

echo "=== 1/3 report_chain_convergence (samples_metrics + iter_metrics + late-window) ==="
"$PY" scripts/report_chain_convergence.py \
  --runs_dir experiments/runs \
  --glob "$RUN_GLOB" \
  --out_md "$SUMMARY_DIR/escape_w1_chain_convergence_report.md" \
  --out_csv "$SUMMARY_DIR/escape_w1_chain_convergence_summary.csv" \
  --late-out-csv "$SUMMARY_DIR/escape_w1_chain_convergence_summary_late.csv"

echo "=== 2/3 analyze_escape_diagnostic (τ_escape, aligned R̂) ==="
"$PY" scripts/analyze_escape_diagnostic.py \
  --runs-dir experiments/runs \
  --parent-glob "$RUN_GLOB" \
  --auto-group \
  --out-csv "$SUMMARY_DIR/escape_w1_tau_escape.csv"

echo "=== 3/3 summarize_escape_init_comparison (pooled trends + U_prior/U_data ratios) ==="
"$PY" scripts/summarize_escape_init_comparison.py \
  --runs-dir experiments/runs \
  --glob "$RUN_GLOB" \
  --out-md "$SUMMARY_DIR/escape_w1_init_comparison.md" \
  --out-csv "$SUMMARY_DIR/escape_w1_init_comparison.csv"

echo "Done. Open:"
echo "  $SUMMARY_DIR/escape_w1_chain_convergence_report.md"
echo "  $SUMMARY_DIR/escape_w1_init_comparison.md"
echo "  $SUMMARY_DIR/escape_w1_tau_escape.csv"
