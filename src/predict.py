#!/usr/bin/env python3
"""Evaluate one saved checkpoint on the CIFAR-10 test split.

This is the quickest way to confirm that a shipped ``checkpoint_last.pt``
reproduces the accuracy recorded in its ``metrics.json`` without retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from models import build_model, parameter_count
from train import build_test_transform, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/runs/E_wrn28_10_ra_cutmix/checkpoint_last.pt"),
        help="checkpoint written by src/train.py",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/CIFAR10"))
    parser.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="ema weights (default) or the last-epoch weights",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--no-tta", action="store_true", help="disable horizontal-flip TTA"
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="reference .npy probabilities (e.g. test_probs.npy) to check against",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = SimpleNamespace(**checkpoint["args"])

    model = build_model(train_args)
    model.load_state_dict(checkpoint[args.weights])
    device = torch.device(args.device)
    model.to(device, memory_format=torch.channels_last).eval()
    print(f"checkpoint: {args.checkpoint}")
    print(
        f"model: WRN-{train_args.depth}-{train_args.width}  SE={train_args.se}  "
        f"params={parameter_count(model):,}  epoch={checkpoint['epoch'] + 1}  "
        f"weights={args.weights}"
    )

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"{test_dir} not found")
    test_set = datasets.ImageFolder(test_dir, transform=build_test_transform())
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    amp_dtype = None
    if device.type == "cuda" and train_args.amp != "none":
        amp_dtype = torch.bfloat16 if train_args.amp == "bf16" else torch.float16

    attn = "" if args.no_tta else ", flip TTA"
    accuracy, probs, _ = evaluate(
        model, loader, device, amp_dtype, tta=not args.no_tta
    )
    print(f"test images: {len(test_set)}  classes: {test_set.classes}")
    print(f"test accuracy ({args.weights}{attn}): {accuracy * 100:.2f}%")

    if args.compare is not None:
        reference = np.load(args.compare)
        if reference.shape != probs.shape:
            raise RuntimeError(f"shape mismatch: {reference.shape} vs {probs.shape}")
        max_diff = float(np.abs(reference - probs).max())
        agreement = float((reference.argmax(axis=1) == probs.argmax(axis=1)).mean())
        print(
            f"compare {args.compare}: max|diff|={max_diff:.2e}  "
            f"argmax agreement={agreement * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
