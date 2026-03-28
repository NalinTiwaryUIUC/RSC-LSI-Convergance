#!/usr/bin/env bash
# Run on the cluster from the RSC_Conv project root (after module load / venv as you usually do).
# Produces:
#   - experiments/summaries/escape_w1_chain_convergence_report.md   (R̂, ESS, late-window probes, iter_metrics)
#   - experiments/summaries/escape_w1_chain_convergence_summary.csv
#   - experiments/summaries/escape_w1_chain_convergence_summary_late.csv
#   - experiments/summaries/escape_w1_tau_escape.csv                 (τ_escape + aligned R̂ from analyze_escape_diagnostic)
#   - experiments/summaries/escape_w1_tau_threshold_grid.csv         (optional: THRESHOLD_GRID=preset)
#   - experiments/summaries/escape_w1_init_comparison.md              (pooled I1 vs I2 vs I3 iter trends + prior ratios)
#   - experiments/summaries/escape_w1_init_comparison.csv
#
# Override globs if your run names differ (e.g. different h, T, n_train):
#   export RUN_GLOB='w1_n512_h5e-06_T100000*_ul_initI*_chain*'
#
# Escape τ policy (default: no arbitrary cutoffs — τ = extremal logged step per metric):
#   TAU_FROM=extremal  — τ_d/τ_ou = first argmax of dist metrics; τ_f = argmin f_margin; τ_nll = argmax nll
#   TAU_FROM=threshold — use thresh-d-sqrt / thresh-ou / f-margin-max / nll-rise-*; see analyze_escape_diagnostic.py -h
#   FILL_MISSING_TAU   — when TAU_FROM=threshold only: none | last | extremal (default extremal)
#   export TAU_FROM=extremal
#   ./scripts/cluster_summarize_escape_w1.sh
#
# Interpretable threshold grid (geometry + per-chain nll_0 / m_0); recommend FILL_MISSING_TAU=none:
#   export THRESHOLD_GRID=preset
#   export FILL_MISSING_TAU=none
#   ./scripts/cluster_summarize_escape_w1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

RUN_GLOB="${RUN_GLOB:-w1_*_ul_initI*_chain*}"
SUMMARY_DIR="${SUMMARY_DIR:-experiments/summaries}"
TAU_FROM="${TAU_FROM:-extremal}"
FILL_MISSING_TAU="${FILL_MISSING_TAU:-extremal}"
THRESHOLD_GRID="${THRESHOLD_GRID:-none}"
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
echo "    --tau-from $TAU_FROM --fill-missing-tau $FILL_MISSING_TAU"
"$PY" scripts/analyze_escape_diagnostic.py \
  --runs-dir experiments/runs \
  --parent-glob "$RUN_GLOB" \
  --auto-group \
  --tau-from "$TAU_FROM" \
  --fill-missing-tau "$FILL_MISSING_TAU" \
  --out-csv "$SUMMARY_DIR/escape_w1_tau_escape.csv"

if [[ "$THRESHOLD_GRID" == "preset" ]]; then
  echo "=== 2b/3 analyze_escape_diagnostic (preset threshold grid) ==="
  echo "    --threshold-grid preset --fill-missing-tau $FILL_MISSING_TAU"
  "$PY" scripts/analyze_escape_diagnostic.py \
    --runs-dir experiments/runs \
    --parent-glob "$RUN_GLOB" \
    --auto-group \
    --threshold-grid preset \
    --fill-missing-tau "$FILL_MISSING_TAU" \
    --out-csv "$SUMMARY_DIR/escape_w1_tau_threshold_grid.csv"
fi

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
if [[ "$THRESHOLD_GRID" == "preset" ]]; then
  echo "  $SUMMARY_DIR/escape_w1_tau_threshold_grid.csv"
fi
