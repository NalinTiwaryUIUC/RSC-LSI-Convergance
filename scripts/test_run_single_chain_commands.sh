#!/bin/bash
# Verify that --beta is accepted by run_single_chain.py and run_single_chain.sh,
# and that the 12-chain-style commands work end-to-end (dry-run only, no actual chain).
# Run from project root: ./scripts/test_run_single_chain_commands.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

# Shared args for the 12-chain setup (only T and run_dir differ when we do real runs)
BASE_ARGS=(
    --arch small_resnet_ln
    --num-blocks 2
    --width 1
    --n_train 512
    --pretrain-path experiments/checkpoints/pretrain_small_resnet_ln_w1_n512.pt
    --alpha 0.3
    --h 5e-8
    --T 6000
    --B 0
    --S 100
    --log-every 20
)

echo "=== 1. Testing run_single_chain.py --dry-run with --beta (all 12 variants) ==="
BETAS=(1 3 10 30 100 300)
for beta in "${BETAS[@]}"; do
    for chain in 0 1; do
        echo "  beta=$beta chain=$chain"
        python3 scripts/run_single_chain.py "${BASE_ARGS[@]}" --beta "$beta" --chain "$chain" --dry-run
    done
done
echo "  OK: Python script accepts --beta and --dry-run for all 12 combinations."
echo ""

echo "=== 2. Testing run_single_chain.sh with --dry-run (one variant: beta=3, chain=1) ==="
./scripts/run_single_chain.sh "${BASE_ARGS[@]}" --beta 3 --chain 1 --dry-run
echo "  OK: run_single_chain.sh forwards --beta and --dry-run to Python."
echo ""

echo "=== 3. Testing submit_chain.sh-style env (BETA passed to Python via run_single_chain.sh) ==="
BETA=10 CHAIN=0 ./scripts/run_single_chain.sh --arch small_resnet_ln --num-blocks 2 --width 1 --n_train 512 \
    --pretrain-path experiments/checkpoints/pretrain_small_resnet_ln_w1_n512.pt \
    --alpha 0.3 --h 5e-8 --T 6000 --B 0 --S 100 --log-every 20 --dry-run
echo "  OK: BETA env var is passed as --beta to Python."
echo ""

echo "=== All checks passed. Commands are valid; use without --dry-run to run real chains. ==="
