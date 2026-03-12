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

#
# Run unit tests, regression suite, and optional smoke run with clear pass/fail
# logging and diagnostics. Use from project root: ./scripts/run_tests.sh
#
# Options (pass in-line):
#   --extra               Run extra diagnostics (diagnose_sanity_checks, test_partition_invariance)
#   --no-extra            Skip extra diagnostics (default)
#   --quick                Regression suite with --quick, faster (default)
#   --no-quick, --full     Regression suite without --quick, full runs
#   --skip-smoke           Skip smoke run
#   --regression-sections 1,2,3   Run only these sections (default: all)
#   --log-dir DIR         Log directory (default: logs/run_tests)
#   -h, --help            Show this help and exit
#
# Examples:
#   ./scripts/run_tests.sh
#   ./scripts/run_tests.sh --extra --no-quick
#   ./scripts/run_tests.sh --extra --skip-smoke --regression-sections 3,4,7
#   sbatch scripts/run_tests.sh --extra --no-quick
#
# We do not use set -e so that all phases run and we report a full summary.

# Defaults (overridden by in-line args below)
QUICK=1
SKIP_SMOKE=0
RUN_EXTRA=0
REGRESSION_SECTIONS=all
LOG_DIR=logs/run_tests

# Parse in-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --extra)
            RUN_EXTRA=1
            shift
            ;;
        --no-extra)
            RUN_EXTRA=0
            shift
            ;;
        --quick)
            QUICK=1
            shift
            ;;
        --no-quick|--full)
            QUICK=0
            shift
            ;;
        --skip-smoke)
            SKIP_SMOKE=1
            shift
            ;;
        --regression-sections)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "run_tests.sh: --regression-sections requires a value (e.g. 1,2,3 or all)" >&2
                exit 1
            fi
            REGRESSION_SECTIONS="$2"
            shift 2
            ;;
        --log-dir)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "run_tests.sh: --log-dir requires a path" >&2
                exit 1
            fi
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --extra              Run extra diagnostics (diagnose_sanity_checks, test_partition_invariance)"
            echo "  --no-extra           Skip extra diagnostics (default)"
            echo "  --quick              Regression with --quick (default)"
            echo "  --no-quick, --full   Regression without --quick"
            echo "  --skip-smoke         Skip smoke run"
            echo "  --regression-sections 1,2,3   Only these sections (default: all)"
            echo "  --log-dir DIR        Log directory (default: logs/run_tests)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "run_tests.sh: unknown option: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

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

# Log dir and timestamped log file
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_tests_${TIMESTAMP}.log"
echo "Log file: $LOG_FILE" | tee "$LOG_FILE"

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

run_phase() {
    local name="$1"
    local cmd="$2"
    local hint="${3:-}"
    log ""
    log "=============================================="
    log "Phase: $name"
    log "=============================================="
    log "Command: $cmd"
    if [ -n "$hint" ]; then
        log "Hint: $hint"
    fi
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    r=${PIPESTATUS[0]}
    if [ "$r" -eq 0 ]; then
        log "Result: PASS"
        return 0
    else
        log "Result: FAIL (exit code $r)"
        [ -n "$hint" ] && log "Diagnostic: $hint"
        return 1
    fi
}

# Track results
UNIT_PASS=0
REGRESS_PASS=0
SMOKE_PASS=0
EXTRA_PASS=0

# Activate venv first so Python/PyTorch checks and all phases use it
if [ -d ".venv" ]; then
    source .venv/bin/activate
    log "Using .venv"
elif [ -d "venv" ]; then
    source venv/bin/activate
    log "Using venv"
fi

log "=== Test run started at $(date) ==="
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

# Optional: install deps (uncomment if you want this in CI)
# pip3 install -q -r requirements.txt || true

# --- Phase 1: Unit tests ---
UNIT_CMD='python3 -m unittest discover tests -v 2>&1'
if run_phase "Unit tests (data, model, ULA, probes, chain)" "$UNIT_CMD" \
    "Re-run: python3 -m unittest discover tests -v"; then
    UNIT_PASS=1
fi

# --- Phase 2: Regression suite ---
if [ "$REGRESSION_SECTIONS" = "all" ]; then
    REGRESS_CMD="python3 scripts/run_regression_suite.py --all"
else
    REGRESS_CMD="python3 scripts/run_regression_suite.py"
    for s in $(echo "$REGRESSION_SECTIONS" | tr ',' ' '); do
        REGRESS_CMD="$REGRESS_CMD --section $s"
    done
fi
[ "$QUICK" = "1" ] && REGRESS_CMD="$REGRESS_CMD --quick"

REGRESS_HINT="Inspect $LOG_FILE for which section failed. Re-run one section: python3 scripts/run_regression_suite.py --section N. Section 1: check iter_metrics.jsonl for NaNs; Section 3: param count ratio d(64)/d(16) in [10,25]; Section 7: prior gradient = alpha*theta."
if run_phase "Regression suite (sections=${REGRESSION_SECTIONS}, quick=${QUICK})" "$REGRESS_CMD" "$REGRESS_HINT"; then
    REGRESS_PASS=1
fi

# --- Phase 3: Extra diagnostics (optional) ---
if [ "$RUN_EXTRA" = "1" ]; then
    EXTRA_CMD1="python3 scripts/diagnose_sanity_checks.py 2>&1"
    EXTRA_CMD2="python3 scripts/test_partition_invariance.py 2>&1"
    log ""
    log "=============================================="
    log "Phase: Extra diagnostics (diagnose_sanity_checks + test_partition_invariance)"
    log "=============================================="
    EXTRA_FAIL=0
    log "Command: $EXTRA_CMD1"
    eval "$EXTRA_CMD1" 2>&1 | tee -a "$LOG_FILE"
    [ "${PIPESTATUS[0]}" -ne 0 ] && EXTRA_FAIL=1
    log "Command: $EXTRA_CMD2"
    eval "$EXTRA_CMD2" 2>&1 | tee -a "$LOG_FILE"
    [ "${PIPESTATUS[0]}" -ne 0 ] && EXTRA_FAIL=1
    if [ "$EXTRA_FAIL" -eq 0 ]; then
        log "Result: PASS"
        EXTRA_PASS=1
    else
        log "Result: FAIL"
        log "Diagnostic: Check BN partition, gradient determinism, noise scaling, partition invariance in $LOG_FILE"
        log "Note: If only 'BN partition batchstat_frozen' failed, that is expected: use full-batch or bn_mode=eval in runs."
    fi
else
    log ""
    log "Phase: Extra diagnostics (skipped, use --extra to run)"
    EXTRA_PASS=1
fi

# --- Phase 4: Smoke run (optional) ---
if [ "$SKIP_SMOKE" = "1" ]; then
    log ""
    log "=============================================="
    log "Phase: Smoke run (--skip-smoke, skipped)"
    log "=============================================="
    SMOKE_PASS=1
else
    SMOKE_CMD="python3 scripts/smoke_run.py 2>&1"
    if run_phase "Smoke run (short ULA chain T=500)" "$SMOKE_CMD" \
        "Smoke writes to experiments/runs/smoke_w0.5_n128_h1e-5_chain0/. Check run_config.yaml and iter_metrics.jsonl for errors."; then
        SMOKE_PASS=1
    fi
fi

# --- Summary ---
log ""
log "=============================================="
log "Summary"
log "=============================================="
log "Unit tests:    $([ $UNIT_PASS -eq 1 ] && echo 'PASS' || echo 'FAIL')"
log "Regression:    $([ $REGRESS_PASS -eq 1 ] && echo 'PASS' || echo 'FAIL')"
log "Extra diag:    $([ $EXTRA_PASS -eq 1 ] && echo 'PASS / skipped' || echo 'FAIL')"
log "Smoke run:     $([ $SMOKE_PASS -eq 1 ] && echo 'PASS' || echo 'FAIL / skipped')"
log "=============================================="

FAILED=0
[ $UNIT_PASS -eq 0 ] && { log "Unit tests failed. See $LOG_FILE"; FAILED=1; }
[ $REGRESS_PASS -eq 0 ] && { log "Regression suite failed. See $LOG_FILE (sections and diagnostics)"; FAILED=1; }
[ $EXTRA_PASS -eq 0 ] && { log "Extra diagnostics failed. See $LOG_FILE"; FAILED=1; }
[ $SMOKE_PASS -eq 0 ] && [ "$SKIP_SMOKE" != "1" ] && { log "Smoke run failed. See $LOG_FILE"; FAILED=1; }

log "Full log: $LOG_FILE"
log "=== Test run finished at $(date) ==="

if [ $FAILED -eq 1 ]; then
    log "Overall: FAIL (at least one phase failed)"
    exit 1
fi
log "Overall: PASS"
exit 0
