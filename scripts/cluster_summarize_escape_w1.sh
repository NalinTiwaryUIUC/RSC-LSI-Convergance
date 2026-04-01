#!/usr/bin/env bash
# Run on the cluster from the RSC_Conv project root (after module load / venv as you usually do).
# Produces:
#   - experiments/summaries/${OUT_PREFIX}_chain_convergence_report.md   (R̂, ESS, late-window probes, iter_metrics)
#   - experiments/summaries/${OUT_PREFIX}_chain_convergence_summary.csv
#   - experiments/summaries/${OUT_PREFIX}_chain_convergence_summary_late.csv
#   - experiments/summaries/${OUT_PREFIX}_tau_escape.csv                 (τ_escape + aligned R̂ from analyze_escape_diagnostic)
#   - experiments/summaries/${OUT_PREFIX}_tau_threshold_grid.csv         (optional: THRESHOLD_GRID=preset)
#   - experiments/summaries/${OUT_PREFIX}_init_comparison.md             (pooled I1 vs I2 vs I3 iter trends + prior ratios)
#   - experiments/summaries/${OUT_PREFIX}_init_comparison.csv
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
#
# Optional absolute predictive levels (same c for all chains; pick from pooled iter range across inits):
#   export ABS_NLL_GE='1.45,1.50,1.55'
#   export ABS_F_MARGIN_LE='-0.2,-0.25,-0.3'
#
# Post-geometry predictive τ (τ_NLL|geom, τ_margin|geom) and Δτ = τ_pred − τ_geom — separate script:
#   python scripts/analyze_post_geom_predictive.py --runs-dir experiments/runs \\
#     --parent-glob 'w1_*_ul_initI*_chain*' --auto-group \\
#     --geom-d 0.05,0.10 --abs-nll-ge=1.45,1.55,1.70 --abs-f-margin-le=-0.20,-0.30,-0.38 \\
#     --out-csv experiments/summaries/postgeom_w1.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

RUN_GLOB="${RUN_GLOB:-w1_*_ul_initI*_chain*}"
SUMMARY_DIR="${SUMMARY_DIR:-experiments/summaries}"
OUT_PREFIX="${OUT_PREFIX:-escape_w1}"
TAU_FROM="${TAU_FROM:-extremal}"
FILL_MISSING_TAU="${FILL_MISSING_TAU:-extremal}"
THRESHOLD_GRID="${THRESHOLD_GRID:-none}"
RUN_POSTGEOM_WINDOWS="${RUN_POSTGEOM_WINDOWS:-0}"
POSTGEOM_GEOM_D="${POSTGEOM_GEOM_D:-0.05}"
POSTGEOM_NLL_THR="${POSTGEOM_NLL_THR:-1.45}"
POSTGEOM_WINDOW_LEN_SAVES="${POSTGEOM_WINDOW_LEN_SAVES:-0}"
mkdir -p "$SUMMARY_DIR"

PY="${PYTHON:-python3}"

echo "=== 1/3 report_chain_convergence (samples_metrics + iter_metrics + late-window) ==="
"$PY" scripts/report_chain_convergence.py \
  --runs_dir experiments/runs \
  --glob "$RUN_GLOB" \
  --out_md "$SUMMARY_DIR/${OUT_PREFIX}_chain_convergence_report.md" \
  --out_csv "$SUMMARY_DIR/${OUT_PREFIX}_chain_convergence_summary.csv" \
  --late-out-csv "$SUMMARY_DIR/${OUT_PREFIX}_chain_convergence_summary_late.csv"

echo "=== 2/3 analyze_escape_diagnostic (τ_escape, aligned R̂) ==="
echo "    --tau-from $TAU_FROM --fill-missing-tau $FILL_MISSING_TAU"
"$PY" scripts/analyze_escape_diagnostic.py \
  --runs-dir experiments/runs \
  --parent-glob "$RUN_GLOB" \
  --auto-group \
  --tau-from "$TAU_FROM" \
  --fill-missing-tau "$FILL_MISSING_TAU" \
  --out-csv "$SUMMARY_DIR/${OUT_PREFIX}_tau_escape.csv"

if [[ "$THRESHOLD_GRID" == "preset" ]]; then
  PRESET_GRID_EXTRA=()
  if [[ -n "${ABS_NLL_GE:-}" ]]; then
    PRESET_GRID_EXTRA+=(--abs-nll-ge="$ABS_NLL_GE")
  fi
  if [[ -n "${ABS_F_MARGIN_LE:-}" ]]; then
    # Equals form so values like -0.2,-0.25 are one argument (argparse rejects "--opt -0.2,...")
    PRESET_GRID_EXTRA+=(--abs-f-margin-le="$ABS_F_MARGIN_LE")
  fi
  echo "=== 2b/3 analyze_escape_diagnostic (preset threshold grid) ==="
  echo "    --threshold-grid preset --fill-missing-tau $FILL_MISSING_TAU"
  [[ -n "${ABS_NLL_GE:-}" ]] && echo "    --abs-nll-ge=$ABS_NLL_GE"
  [[ -n "${ABS_F_MARGIN_LE:-}" ]] && echo "    --abs-f-margin-le=$ABS_F_MARGIN_LE"
  "$PY" scripts/analyze_escape_diagnostic.py \
    --runs-dir experiments/runs \
    --parent-glob "$RUN_GLOB" \
    --auto-group \
    --threshold-grid preset \
    --fill-missing-tau "$FILL_MISSING_TAU" \
    "${PRESET_GRID_EXTRA[@]}" \
    --out-csv "$SUMMARY_DIR/${OUT_PREFIX}_tau_threshold_grid.csv"
fi

echo "=== 3/3 summarize_escape_init_comparison (pooled trends + U_prior/U_data ratios) ==="
"$PY" scripts/summarize_escape_init_comparison.py \
  --runs-dir experiments/runs \
  --glob "$RUN_GLOB" \
  --out-md "$SUMMARY_DIR/${OUT_PREFIX}_init_comparison.md" \
  --out-csv "$SUMMARY_DIR/${OUT_PREFIX}_init_comparison.csv"

if [[ "$RUN_POSTGEOM_WINDOWS" == "1" ]]; then
  echo "=== 3b/3 analyze_post_geom_predictive (event-aligned windows) ==="
  "$PY" scripts/analyze_post_geom_predictive.py \
    --runs-dir experiments/runs \
    --parent-glob "$RUN_GLOB" \
    --auto-group \
    --geom-d 0.05,0.10 \
    --abs-nll-ge=1.45,1.55,1.70 \
    --abs-f-margin-le=-0.20,-0.30,-0.38 \
    --window-geom-d "$POSTGEOM_GEOM_D" \
    --window-nll-thr "$POSTGEOM_NLL_THR" \
    --window-len-saves "$POSTGEOM_WINDOW_LEN_SAVES" \
    --out-csv "$SUMMARY_DIR/${OUT_PREFIX}_postgeom.csv" \
    --out-md "$SUMMARY_DIR/${OUT_PREFIX}_postgeom_windows.md"
fi

echo "Done. Open:"
echo "  $SUMMARY_DIR/${OUT_PREFIX}_chain_convergence_report.md"
echo "  $SUMMARY_DIR/${OUT_PREFIX}_init_comparison.md"
echo "  $SUMMARY_DIR/${OUT_PREFIX}_tau_escape.csv"
if [[ "$THRESHOLD_GRID" == "preset" ]]; then
  echo "  $SUMMARY_DIR/${OUT_PREFIX}_tau_threshold_grid.csv"
fi
if [[ "$RUN_POSTGEOM_WINDOWS" == "1" ]]; then
  echo "  $SUMMARY_DIR/${OUT_PREFIX}_postgeom.csv"
  echo "  $SUMMARY_DIR/${OUT_PREFIX}_postgeom_windows.md"
fi
