#!/usr/bin/env python3
"""Uniformly ensemble independently trained CIFAR-10 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="directories containing test_probs.npy and test_labels.npy",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ensemble"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_probs: list[np.ndarray] = []
    labels: np.ndarray | None = None
    individual: list[dict] = []

    for run_dir in args.run_dirs:
        probs = np.load(run_dir / "test_probs.npy")
        run_labels = np.load(run_dir / "test_labels.npy")
        if labels is None:
            labels = run_labels
        elif not np.array_equal(labels, run_labels):
            raise RuntimeError(f"Label order mismatch in {run_dir}")

        accuracy = float((probs.argmax(axis=1) == run_labels).mean())
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        individual.append(
            {
                "run": run_dir.name,
                "accuracy": accuracy,
                "model": metrics.get("model"),
                "se": metrics.get("se"),
                "mix": metrics.get("mix"),
                "randaugment": metrics.get("randaugment"),
            }
        )
        all_probs.append(probs)

    if labels is None:
        raise RuntimeError("No runs supplied")
    probs = np.mean(np.stack(all_probs, axis=0), axis=0, dtype=np.float64).astype(np.float32)
    predictions = probs.argmax(axis=1)
    accuracy = float((predictions == labels).mean())

    num_classes = probs.shape[1]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    per_class = confusion.diagonal() / confusion.sum(axis=1)
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    confusions = sorted(
        (
            {
                "true": class_names[i],
                "predicted": class_names[j],
                "count": int(confusion[i, j]),
            }
            for i in range(num_classes)
            for j in range(num_classes)
            if i != j
        ),
        key=lambda item: item["count"],
        reverse=True,
    )[:10]

    print("individual models:")
    for item in individual:
        print(
            f"  {item['run']:<20s} acc={item['accuracy'] * 100:.2f}%  "
            f"model={item['model']} se={item['se']} mix={item['mix']} "
            f"randaug={item['randaugment']}"
        )
    print(f"uniform ensemble ({len(all_probs)} models): {accuracy * 100:.2f}%")
    print("per-class:", {name: round(float(v) * 100, 2) for name, v in zip(class_names, per_class)})
    print("top confusions:")
    for item in confusions[:5]:
        print(f"  {item['true']} -> {item['predicted']}: {item['count']}")

    np.save(args.output_dir / "ensemble_probs.npy", probs)
    np.save(args.output_dir / "test_labels.npy", labels)
    np.savetxt(
        args.output_dir / "confusion_matrix.csv",
        confusion,
        fmt="%d",
        delimiter=",",
        header=",".join(class_names),
        comments="",
    )
    result = {
        "num_models": len(all_probs),
        "members": [str(path) for path in args.run_dirs],
        "individual": individual,
        "ensemble_test_accuracy": accuracy,
        "ensemble_test_error": 1.0 - accuracy,
        "tta": "horizontal_flip",
        "per_class_accuracy": {
            name: float(value) for name, value in zip(class_names, per_class)
        },
        "top_confusions": confusions,
        "confusion_matrix": confusion.tolist(),
        "class_order": class_names,
    }
    (args.output_dir / "ensemble_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
