from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(path: str) -> Dict[str, List[float]]:
	with open(path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	if not rows:
		raise ValueError(f"Metrics file is empty: {path}")

	required = [
		"epoch",
		"train_loss",
		"val_loss",
		"train_acc1",
		"val_acc1",
		"train_acc5",
		"val_acc5",
		"lr",
	]
	for col in required:
		if col not in rows[0]:
			raise ValueError(f"Missing required column '{col}' in metrics CSV.")

	metrics: Dict[str, List[float]] = {k: [] for k in required}
	for row in rows:
		for key in required:
			metrics[key].append(float(row[key]))

	return metrics


def _plot_metrics(metrics: Dict[str, List[float]], out_path: str, title: str) -> None:
	epochs = metrics["epoch"]
	val_acc1 = metrics["val_acc1"]
	best_idx = max(range(len(val_acc1)), key=val_acc1.__getitem__)
	best_epoch = int(epochs[best_idx])
	best_val = val_acc1[best_idx]

	fig, axes = plt.subplots(2, 2, figsize=(12, 8))
	fig.suptitle(title, fontsize=14, y=1.02)

	ax = axes[0, 0]
	ax.plot(epochs, metrics["train_loss"], marker="o", label="train_loss")
	ax.plot(epochs, metrics["val_loss"], marker="o", label="val_loss")
	ax.set_title("Loss")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Cross-Entropy")
	ax.grid(alpha=0.25)
	ax.legend()

	ax = axes[0, 1]
	ax.plot(epochs, metrics["train_acc1"], marker="o", label="train_acc1")
	ax.plot(epochs, metrics["val_acc1"], marker="o", label="val_acc1")
	ax.scatter([best_epoch], [best_val], color="red", zorder=3, label="best_val_acc1")
	ax.set_title("Top-1 Accuracy")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Accuracy")
	ax.grid(alpha=0.25)
	ax.legend()

	ax = axes[1, 0]
	ax.plot(epochs, metrics["train_acc5"], marker="o", label="train_acc5")
	ax.plot(epochs, metrics["val_acc5"], marker="o", label="val_acc5")
	ax.set_title("Top-5 Accuracy")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Accuracy")
	ax.grid(alpha=0.25)
	ax.legend()

	ax = axes[1, 1]
	ax.plot(epochs, metrics["lr"], marker="o", label="lr")
	ax.set_title("Learning Rate")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("LR")
	ax.grid(alpha=0.25)
	ax.legend()

	plt.tight_layout()
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot training curves from metrics CSV")
	parser.add_argument("--metrics", default="artifacts/metrics.csv")
	parser.add_argument("--out", default="assets/training_curves.png")
	parser.add_argument("--title", default="Training Curves")
	args = parser.parse_args()

	metrics = _load_metrics(args.metrics)
	_plot_metrics(metrics, out_path=args.out, title=args.title)
	print(f"Training curves saved to: {args.out}")


if __name__ == "__main__":
	main()
