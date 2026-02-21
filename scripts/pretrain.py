"""
Pretrain a model with fixed random seed. Saves checkpoint for use by run_single_chain and diagnose_ula.
Use the same checkpoint across all chains for a given (width, n_train) to standardize initialization.

Usage:
  python scripts/pretrain.py --width 1 --n_train 1024 --pretrain-steps 2000
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

# Fixed seed for reproducibility across runs
PRETRAIN_SEED = 42


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
    p = argparse.ArgumentParser(description="Pretrain model with fixed seed")
    p.add_argument("--width", type=float, default=1.0, help="Width multiplier")
    p.add_argument("--n_train", type=int, default=1024, help="Training subset size")
    p.add_argument("--pretrain-steps", type=int, default=2000, help="SGD steps")
    p.add_argument("--pretrain-lr", type=float, default=0.02, help="Learning rate")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output path; default: experiments/checkpoints/pretrain_w{WIDTH}_n{n_train}.pt")
    p.add_argument("--data_dir", type=str, default="experiments/data")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--seed", type=int, default=PRETRAIN_SEED, help="Fixed seed for reproducibility")
    args = p.parse_args()

    set_pretrain_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"

    train_loader = get_train_loader(
        args.n_train,
        batch_size=args.n_train,
        dataset_seed=args.seed,
        data_dir=args.data_dir,
        root=args.root,
        pin_memory=use_gpu,
    )
    x_train, y_train = next(iter(train_loader))
    x_train = x_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)

    model = create_model(width_multiplier=args.width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.pretrain_lr, momentum=0.9)

    model.train()
    for _ in range(args.pretrain_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train, reduction="mean")
        loss.backward()
        optimizer.step()

    out_path = args.output
    if out_path is None:
        w_str = int(args.width) if args.width == int(args.width) else args.width
        out_dir = Path("experiments/checkpoints")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pretrain_w{w_str}_n{args.n_train}.pt"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "width": args.width, "n_train": args.n_train}, out_path)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
