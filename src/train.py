#!/usr/bin/env python3
"""Train a Wide ResNet on CIFAR-10 and evaluate it on the official test set.

The script deliberately keeps the training recipe self-contained: no pretrained
weights or external datasets are used.  Each invocation trains one model; the
companion ``ensemble.py`` combines independently trained models.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment

from models import build_model, parameter_count

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


class Cutout:
    """Zero one square region of a normalised CHW image with probability p."""

    def __init__(self, size: int, p: float = 0.5) -> None:
        self.size = size
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.size <= 0 or random.random() > self.p:
            return image
        _, height, width = image.shape
        cy = random.randint(0, height - 1)
        cx = random.randint(0, width - 1)
        half = self.size // 2
        y1, y2 = max(cy - half, 0), min(cy + half, height)
        x1, x2 = max(cx - half, 0), min(cx + half, width)
        image[:, y1:y2, x1:x2] = 0.0
        return image


def build_train_transform(args: argparse.Namespace) -> transforms.Compose:
    ops: list = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ]
    if args.randaugment:
        ops.append(RandAugment(num_ops=2, magnitude=9))
    ops.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if args.cutout > 0:
        ops.append(Cutout(args.cutout, p=args.cutout_prob))
    return transforms.Compose(ops)


def build_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


def build_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, list[str], datasets.ImageFolder, datasets.ImageFolder, np.ndarray]:
    data_root = Path(args.data_dir)
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {train_dir} and {test_dir} to contain class folders"
        )

    train_set = datasets.ImageFolder(train_dir, transform=build_train_transform(args))
    test_set = datasets.ImageFolder(test_dir, transform=build_test_transform())
    if train_set.class_to_idx != test_set.class_to_idx:
        raise RuntimeError("Train and test class mappings differ")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=max(args.workers // 2, 2),
        pin_memory=True,
        persistent_workers=False,
    )
    classes = list(train_set.classes)
    counts = np.bincount([label for _, label in train_set.samples], minlength=NUM_CLASSES)
    return train_loader, test_loader, classes, train_set, test_set, counts


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ModelEMA:
    """Exponential moving average of parameters and floating-point buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.decay
        ema_params = [p.detach() for p in self.module.parameters()]
        model_params = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, model_params, alpha=1.0 - decay)

        ema_buffers = list(self.module.buffers())
        model_buffers = list(model.buffers())
        if len(ema_buffers) != len(model_buffers):
            raise RuntimeError("EMA/model buffer mismatch")
        for ema_buffer, buffer in zip(ema_buffers, model_buffers):
            if buffer.dtype.is_floating_point:
                ema_buffer.mul_(decay).add_(buffer.detach(), alpha=1.0 - decay)
            else:
                ema_buffer.copy_(buffer)


def soft_targets(
    labels: torch.Tensor, num_classes: int, smoothing: float
) -> torch.Tensor:
    target = torch.full(
        (labels.size(0), num_classes),
        smoothing / num_classes,
        device=labels.device,
        dtype=torch.float32,
    )
    target.scatter_(
        1, labels.unsqueeze(1), 1.0 - smoothing + smoothing / num_classes
    )
    return target


def mix_batch(
    images: torch.Tensor,
    target: torch.Tensor,
    method: str,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp or CutMix.  Both images and soft targets are interpolated."""
    batch_size, _, height, width = images.shape
    perm = torch.randperm(batch_size, device=images.device)
    lam = float(np.random.beta(alpha, alpha))

    if method == "mixup":
        images = lam * images + (1.0 - lam) * images[perm]
    elif method == "cutmix":
        ratio = math.sqrt(max(1.0 - lam, 0.0))
        cut_w, cut_h = int(width * ratio), int(height * ratio)
        cx, cy = random.randint(0, width - 1), random.randint(0, height - 1)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    else:
        raise ValueError(f"Unknown mix method: {method}")

    target = lam * target + (1.0 - lam) * target[perm]
    return images, target


def learning_rate_at(epoch: int, args: argparse.Namespace) -> float:
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: ModelEMA,
    scaler: GradScaler,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    use_mix = args.mix != "none"
    mix_prob = args.mix_prob if use_mix else 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        target = soft_targets(labels, NUM_CLASSES, args.label_smoothing)

        if use_mix and random.random() < mix_prob:
            images, target = mix_batch(images, target, args.mix, args.mix_alpha)
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = torch.sum(
                -target * F.log_softmax(logits.float(), dim=1), dim=1
            ).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    tta: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            probs = F.softmax(model(images).float(), dim=1)
            if tta:
                # Horizontal flip is label preserving and slightly reduces variance.
                probs = probs + F.softmax(model(torch.flip(images, dims=[3])).float(), dim=1)
                probs = probs / 2.0
        all_probs.append(probs.cpu())
        all_labels.append(labels)

    probs = torch.cat(all_probs).numpy().astype(np.float32)
    labels = torch.cat(all_labels).numpy()
    accuracy = float((probs.argmax(axis=1) == labels).mean())
    return accuracy, probs, labels


def per_class_accuracy(
    probs: np.ndarray, labels: np.ndarray, classes: list[str]
) -> dict[str, float]:
    predictions = probs.argmax(axis=1)
    return {
        name: float((predictions[labels == index] == index).mean())
        for index, name in enumerate(classes)
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/CIFAR10"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")

    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--se", action="store_true", help="use SE blocks")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mix", choices=["none", "mixup", "cutmix"], default="cutmix")
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--mix-alpha", type=float, default=1.0)
    parser.add_argument("--cutout", type=int, default=8)
    parser.add_argument("--cutout-prob", type=float, default=0.5)
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9995)

    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= args.warmup_epochs:
        raise ValueError("epochs must exceed warmup_epochs")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    log_path = args.output_dir / "train.log"
    log_file = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    train_loader, test_loader, classes, train_set, test_set, counts = build_loaders(args)
    log(f"train images: {len(train_set)}  test images: {len(test_set)}")
    log(f"train class counts: {counts.tolist()}")
    log(f"classes: {classes}")

    model = build_model(args).to(device, memory_format=torch.channels_last)
    log(f"model: WRN-{args.depth}-{args.width}, SE={args.se}, "
        f"dropout={args.dropout}, params={parameter_count(model):,}")

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.SGD(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True,
    )
    ema = ModelEMA(model, args.ema_decay)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    scaler = GradScaler("cuda", enabled=args.amp == "fp16")

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.module.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        log(f"resumed from {args.resume} at epoch {start_epoch}")

    history_path = args.output_dir / "history.csv"
    history_exists = history_path.exists()
    history_file = history_path.open("a" if start_epoch > 0 and history_exists else "w", newline="")
    writer = csv.writer(history_file)
    if not (start_epoch > 0 and history_exists):
        writer.writerow(["epoch", "lr", "loss", "train_acc", "seconds"])

    log(
        f"start training: epochs={args.epochs}, batch={args.batch_size}, "
        f"lr={args.lr}, wd={args.weight_decay}, mix={args.mix}, "
        f"label_smoothing={args.label_smoothing}, amp={args.amp}"
    )
    train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        lr = learning_rate_at(epoch, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        epoch_start = time.time()
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, ema, scaler, args, device, amp_dtype
        )
        elapsed = time.time() - epoch_start
        writer.writerow([epoch + 1, f"{lr:.6g}", f"{loss:.5f}", f"{train_acc:.5f}", f"{elapsed:.1f}"])
        history_file.flush()
        log(
            f"epoch {epoch + 1:03d}/{args.epochs}  lr={lr:.5f}  "
            f"loss={loss:.4f}  train_acc={train_acc:.4f}  time={elapsed:.1f}s"
        )

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(
                args.output_dir / "checkpoint_last.pt",
                model, ema, optimizer, epoch, args,
            )

    train_seconds = time.time() - train_start
    history_file.close()

    log("evaluating raw and EMA weights on the 10,000-image test set (flip TTA)")
    raw_acc, raw_probs, test_labels = evaluate(
        model, test_loader, device, amp_dtype, tta=True
    )
    ema_acc, ema_probs, _ = evaluate(
        ema.module, test_loader, device, amp_dtype, tta=True
    )
    log(f"test accuracy raw: {raw_acc * 100:.2f}%")
    log(f"test accuracy EMA: {ema_acc * 100:.2f}%")

    np.save(args.output_dir / "test_probs.npy", ema_probs)
    np.save(args.output_dir / "test_probs_raw.npy", raw_probs)
    np.save(args.output_dir / "test_labels.npy", test_labels)
    per_class = per_class_accuracy(ema_probs, test_labels, classes)

    metrics = {
        "model": f"WRN-{args.depth}-{args.width}",
        "se": args.se,
        "params": parameter_count(model),
        "train_images": len(train_set),
        "test_images": len(test_set),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "mix": args.mix,
        "mix_prob": args.mix_prob,
        "label_smoothing": args.label_smoothing,
        "cutout": args.cutout,
        "randaugment": args.randaugment,
        "ema_decay": args.ema_decay,
        "tta": "horizontal_flip",
        "test_accuracy_raw": raw_acc,
        "test_accuracy_ema": ema_acc,
        "test_error_ema": 1.0 - ema_acc,
        "per_class_accuracy_ema": per_class,
        "train_seconds": train_seconds,
        "test_seconds": time.time() - train_start - train_seconds,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"per-class accuracy (EMA): {json.dumps(per_class, ensure_ascii=False)}")
    log(f"artifacts written to {args.output_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
