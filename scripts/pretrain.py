"""
Pretrain a model with fixed random seed. Saves checkpoint for use by run_single_chain and diagnose_ula.
Use the same checkpoint across all chains for a given (width, n_train) to standardize initialization.

Objective: mean CE + (α/(2n))||θ||² on the n-point subset (n = n_train), via optimizer weight_decay:
  default λ = α/n so PyTorch's (λ/2)||θ||² equals (α/(2n))||θ||².
  torch.optim.SGD(..., momentum=0.9, nesterov=False, weight_decay=λ); use --pretrain-weight-decay -1 for λ = α/n.

Legacy explicit loss term (α/2n)||θ||² matching *sum* CE is removed; use the same ce_reduction at sampling time.

Data: full batch on the fixed subset, eval transforms (no train augmentation). No gradient clipping by default.

Usage:
  python scripts/pretrain.py --width 1 --n_train 512 --alpha 0.3 --pretrain-steps 2000 --snapshot-every 25
  python scripts/pretrain.py --width 0.1 -o experiments/checkpoints/out.pt
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


def _bn_calibrate(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    n_train: int,
    microbatch: int,
) -> None:
    model.train()
    with torch.no_grad():
        for start in range(0, n_train, microbatch):
            end = min(start + microbatch, n_train)
            _ = model(x_train[start:end])
    model.eval()


def _metrics_eval(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(x_train)
        ce_mean = F.cross_entropy(logits, y_train, reduction="mean").item()
        pred = logits.argmax(dim=1)
        acc = (pred == y_train).float().mean().item() * 100
    return ce_mean, acc


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    width: float,
    n_train: int,
    alpha: float,
    arch: str,
    num_blocks: int,
    step: int | None = None,
    *,
    weight_decay_effective: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "state_dict": model.state_dict(),
        "width": width,
        "n_train": n_train,
        "alpha": alpha,
        "arch": arch,
        "num_blocks": num_blocks,
    }
    if step is not None:
        payload["pretrain_step"] = step
    if weight_decay_effective is not None:
        payload["pretrain_weight_decay_effective"] = weight_decay_effective
    torch.save(payload, path)


def _default_snapshot_paths(
    base_dir: Path,
    w_str: str | float,
    n_train: int,
    num_blocks: int,
    step: int,
) -> Path:
    return base_dir / f"pretrain_w{w_str}_n{n_train}_nb{num_blocks}_step{step}.pt"


def main() -> None:
    p = argparse.ArgumentParser(description="Pretrain model with fixed seed (MAP objective + BN calibration)")
    p.add_argument("--width", type=float, default=1.0, help="Width multiplier")
    p.add_argument("--n_train", type=int, default=1024, help="Training subset size")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Prior strength α; default WD uses λ = α/n_train when --pretrain-weight-decay < 0.",
    )
    p.add_argument("--pretrain-steps", type=int, default=2000, help="SGD steps (stop early if train acc saturates; keep snapshots)")
    p.add_argument("--pretrain-lr", type=float, default=0.01, help="SGD learning rate (try 0.005 / 0.01 / 0.02 mini-sweep)")
    p.add_argument(
        "--pretrain-weight-decay",
        type=float,
        default=-1.0,
        help="SGD weight_decay λ for (λ/2)||θ||². -1 = λ = α/n_train. 0 = no L2.",
    )
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output path; default: experiments/checkpoints/pretrain_w{WIDTH}_n{n_train}_nb{num_blocks}.pt")
    p.add_argument(
        "--snapshot-steps",
        type=str,
        default="",
        help="Comma-separated step indices (1..pretrain-steps) at which to save intermediate checkpoints "
        "after BN calibration (files: ..._step{N}.pt).",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=25,
        help="Save a snapshot every K steps (default 25; set 0 to disable unless --snapshot-steps set).",
    )
    p.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Directory for intermediate *_step*.pt snapshots (default: experiments/checkpoints).",
    )
    p.add_argument("--bn-calibration-microbatch", type=int, default=BN_CALIBRATION_MICROBATCH,
                   help="Microbatch size for BN calibration forward pass (fixed across widths)")
    p.add_argument("--data_dir", type=str, default="experiments/data")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--dataset-seed", type=int, default=42, help="For train_subset_indices.json only")
    p.add_argument("--pretrain-seed", type=int, default=PRETRAIN_SEED, help="Init + optimizer randomness")
    p.add_argument("--verify", action="store_true", help="Run 1: reload from disk and verify ce/acc on same batch")
    p.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "small_resnet_ln"],
                   help="Model architecture to pretrain.")
    p.add_argument("--num-blocks", type=int, default=2,
                   help="Number of residual blocks for small_resnet_ln (ignored for resnet18).")
    args = p.parse_args()

    n_train = max(int(args.n_train), 1)
    wd = float(args.pretrain_weight_decay)
    if wd < 0:
        wd = float(args.alpha) / float(n_train)
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

    model = create_model(
        width_multiplier=args.width,
        num_classes=10,
        arch=args.arch,
        num_blocks=args.num_blocks,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.pretrain_lr,
        momentum=0.9,
        weight_decay=wd,
        nesterov=False,
    )

    w_str = int(args.width) if args.width == int(args.width) else args.width
    out_dir = Path("experiments/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = Path(args.snapshot_dir) if args.snapshot_dir else out_dir
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Parse snapshot steps
    snapshot_set: set[int] = set()
    if args.snapshot_steps.strip():
        for part in args.snapshot_steps.split(","):
            part = part.strip()
            if not part:
                continue
            snapshot_set.add(int(part))
    if args.snapshot_every and args.snapshot_every > 0:
        k = args.snapshot_every
        for s in range(k, args.pretrain_steps + 1, k):
            snapshot_set.add(s)
    snapshot_set = {s for s in snapshot_set if 1 <= s <= args.pretrain_steps}
    snapshot_list = sorted(snapshot_set)

    microbatch = args.bn_calibration_microbatch

    # U_train = mean CE + (wd/2)||θ||² via optimizer weight_decay (default wd = α/n_train)
    model.train()
    for step in range(1, args.pretrain_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train, reduction="mean")
        loss.backward()
        optimizer.step()

        if step in snapshot_set:
            _bn_calibrate(model, x_train, args.n_train, microbatch)
            ce_m, acc = _metrics_eval(model, x_train, y_train)
            snap_path = _default_snapshot_paths(snap_dir, w_str, args.n_train, args.num_blocks, step)
            _save_checkpoint(
                snap_path,
                model,
                args.width,
                args.n_train,
                args.alpha,
                args.arch,
                args.num_blocks,
                step=step,
                weight_decay_effective=wd,
            )
            print(
                f"Snapshot step {step}: mean CE = {ce_m:.4f}, accuracy = {acc:.2f}% (eval) -> {snap_path}"
            )
            model.train()

    # Final BN calibration + full eval (same as before)
    _bn_calibrate(model, x_train, args.n_train, microbatch)
    ce_mean, acc = _metrics_eval(model, x_train, y_train)
    print(f"Pretrain done: mean CE = {ce_mean:.4f}, accuracy = {acc:.2f}% (on train batch, eval mode)")

    out_path = args.output
    if out_path is None:
        out_path = out_dir / f"pretrain_w{w_str}_n{args.n_train}_nb{args.num_blocks}.pt"

    out_path = Path(out_path)
    _save_checkpoint(
        out_path,
        model,
        args.width,
        args.n_train,
        args.alpha,
        args.arch,
        args.num_blocks,
        step=None,
        weight_decay_effective=wd,
    )
    print("Wrote", out_path)
    print(
        f"Pretrain MAP setup: mean CE + (λ/2)||θ||² with λ={wd} (α/n_train when default WD; SGD lr={args.pretrain_lr}, momentum=0.9, nesterov=False)"
    )
    if snapshot_list:
        print(f"Intermediate snapshots requested at steps: {snapshot_list}")

    # Run 1: Reload verify — re-instantiate, load from disk, eval on same batch
    if args.verify:
        batch_path = out_path.with_suffix(".batch.pt")
        torch.save({"x": x_train.cpu(), "y": y_train.cpu()}, batch_path)
        print("Saved batch to", batch_path)
        fresh = create_model(
            width_multiplier=args.width,
            num_classes=10,
            arch=args.arch,
            num_blocks=args.num_blocks,
        ).to(device)
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
