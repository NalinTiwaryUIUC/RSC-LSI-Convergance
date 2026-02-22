"""
Pretrain a model with fixed random seed. Saves checkpoint for use by run_single_chain and diagnose_ula.
Use the same checkpoint across all chains for a given (width, n_train) to standardize initialization.

Objective: loss = ce_mean + 0.5 * (alpha / n_train) * ||θ||² — same minimizer as CE_sum + (α/2)||θ||².
BN: pretrain in train(); after pretrain, BN calibration pass (forward in train mode, fixed microbatch_size);
    then eval() for sampling.

Usage:
  python scripts/pretrain.py --width 0.1 --n_train 1024 --alpha 0.1 --pretrain-steps 2000
  python scripts/pretrain.py --width 0.1 --n_train 1024 -o experiments/checkpoints/pretrain_w0.1_n1024.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import get_train_loader
from models import create_model

# Fixed seed for reproducibility across runs (use same subset indices across widths)
PRETRAIN_SEED = 42

# Fixed microbatch size for BN calibration (constant across widths)
BN_CALIBRATION_MICROBATCH = 256


def set_pretrain_seed(seed: int = PRETRAIN_SEED) -> None:
    """Set all random seeds for deterministic pretraining."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    p = argparse.ArgumentParser(description="Pretrain model with fixed seed (MAP objective + BN calibration)")
    p.add_argument("--width", type=float, default=1.0, help="Width multiplier")
    p.add_argument("--n_train", type=int, default=1024, help="Training subset size")
    p.add_argument("--alpha", type=float, default=0.1, help="L2 prior strength (same as chain; scales loss L2 as alpha/n_train)")
    p.add_argument("--pretrain-steps", type=int, default=2000, help="SGD steps")
    p.add_argument("--pretrain-lr", type=float, default=0.02, help="Learning rate (no weight_decay; L2 in loss)")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output path; default: experiments/checkpoints/pretrain_w{WIDTH}_n{n_train}.pt")
    p.add_argument("--bn-calibration-microbatch", type=int, default=BN_CALIBRATION_MICROBATCH,
                   help="Microbatch size for BN calibration forward pass (fixed across widths)")
    p.add_argument("--data_dir", type=str, default="experiments/data")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--dataset-seed", type=int, default=42, help="For train_subset_indices.json only")
    p.add_argument("--pretrain-seed", type=int, default=PRETRAIN_SEED, help="Init + optimizer randomness")
    p.add_argument("--verify", action="store_true", help="Run 1: reload from disk and verify ce/acc on same batch")
    args = p.parse_args()

    set_pretrain_seed(args.pretrain_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"

    train_loader = get_train_loader(
        args.n_train,
        batch_size=args.n_train,
        dataset_seed=args.dataset_seed,
        data_dir=args.data_dir,
        root=args.root,
        pin_memory=use_gpu,
        eval_transform=True,
    )
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)

    model = create_model(width_multiplier=args.width).to(device)
    # No weight_decay; L2 penalty explicit in loss to match sampling target
    optimizer = torch.optim.SGD(model.parameters(), lr=args.pretrain_lr, momentum=0.9, weight_decay=0.0)

    # Objective: ce_mean + 0.5 * (alpha / n_train) * ||θ||² — same minimizer as CE_sum + (α/2)||θ||²
    model.train()
    for _ in range(args.pretrain_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        ce_mean = F.cross_entropy(logits, y_train, reduction="mean")
        reg = (0.5 * args.alpha / args.n_train) * sum((p * p).sum() for p in model.parameters())
        loss = ce_mean + reg
        loss.backward()
        optimizer.step()

    # BN calibration: forward passes in train mode (no grad) with fixed microbatch_size to populate running stats
    model.train()
    microbatch = args.bn_calibration_microbatch
    with torch.no_grad():
        for start in range(0, args.n_train, microbatch):
            end = min(start + microbatch, args.n_train)
            _ = model(x_train[start:end])
    model.eval()

    # Final metrics on same batch (eval mode for reproducibility)
    with torch.no_grad():
        logits = model(x_train)
        ce_mean = F.cross_entropy(logits, y_train, reduction="mean").item()
        pred = logits.argmax(dim=1)
        acc = (pred == y_train).float().mean().item() * 100
    print(f"Pretrain done: mean CE = {ce_mean:.4f}, accuracy = {acc:.2f}% (on train batch, eval mode)")

    out_path = args.output
    if out_path is None:
        w_str = int(args.width) if args.width == int(args.width) else args.width
        out_dir = Path("experiments/checkpoints")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pretrain_w{w_str}_n{args.n_train}.pt"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # state_dict includes BN running_mean/var after calibration (part of eval target)
    torch.save({
        "state_dict": model.state_dict(),
        "width": args.width,
        "n_train": args.n_train,
        "alpha": args.alpha,
    }, out_path)
    print("Wrote", out_path)

    # Run 1: Reload verify — re-instantiate, load from disk, eval on same batch
    if args.verify:
        batch_path = out_path.with_suffix(".batch.pt")
        torch.save({"x": x_train.cpu(), "y": y_train.cpu()}, batch_path)
        print("Saved batch to", batch_path)
        fresh = create_model(width_multiplier=args.width).to(device)
        loaded = torch.load(out_path, map_location=device, weights_only=True)
        fresh.load_state_dict(loaded["state_dict"], strict=True)
        fresh.eval()
        with torch.no_grad():
            logits = fresh(x_train)
            ce_reload = F.cross_entropy(logits, y_train, reduction="mean").item()
            acc_reload = (logits.argmax(dim=1) == y_train).float().mean().item() * 100
        print(f"Run 1 (reload verify): ce_mean = {ce_reload:.6f}, acc = {acc_reload:.2f}%")


if __name__ == "__main__":
    main()
