#!/usr/bin/env bash
# Fast sanity tests for the negative-curvature experiment (math + scipy bridge).
# Skips TestRunnerSmoke (full small_resnet_ln + eigsh) and TestPlotSmoke (matplotlib subprocess).
#
# Usage (from repo root):
#   ./scripts/run_neg_curvature_tests_quick.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

if [ -d ".venv" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
elif [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi

exec python3 -m unittest -v \
    tests.test_neg_curvature.TestHVP \
    tests.test_neg_curvature.TestGGNCE \
    tests.test_neg_curvature.TestNMEZeroForLinearModel \
    tests.test_neg_curvature.TestTopKSmallest \
    tests.test_neg_curvature.TestSLQ \
    tests.test_neg_curvature.TestCumulativeMetrics \
    tests.test_neg_curvature.TestWidthMapping \
    tests.test_neg_curvature.TestLinop \
    tests.test_neg_curvature.TestFlatGrad \
    tests.test_neg_curvature.TestCELossScaling \
    "$@"
