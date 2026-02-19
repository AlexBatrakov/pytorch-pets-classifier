from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from .config import load_config
from .data import build_loaders, build_test_loader
from .model import build_model
from .utils import accuracy_topk, get_device, set_seed


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate pets classifier")
	parser.add_argument("--ckpt", default="checkpoints/best.pt")
	parser.add_argument("--config", default=None)
	parser.add_argument("--split", default="val", choices=["val", "test"])
	parser.add_argument("--cm-out", default=None, help="Save confusion matrix image to path")
	parser.add_argument(
		"--cm-normalize",
		action="store_true",
		help="Normalize confusion matrix rows to percentages",
	)
	args = parser.parse_args()

	ckpt = torch.load(args.ckpt, map_location="cpu")
	if args.config:
		cfg = load_config(args.config)
	else:
		cfg = ckpt.get("config", load_config("configs/default.yaml"))

	set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
	device = get_device()

	if args.split == "test":
		val_loader, class_names = build_test_loader(cfg)
	else:
		_, val_loader, class_names = build_loaders(cfg)
	model = build_model(num_classes=len(class_names), pretrained=False, freeze_backbone=False)
	model.load_state_dict(ckpt["model_state_dict"])
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	model.eval()
	total_loss = 0.0
	total = 0
	correct1 = 0.0
	correct5 = 0.0

	all_preds = []
	all_targets = []
	with torch.no_grad():
		for images, targets in val_loader:
			images = images.to(device)
			targets = targets.to(device)

			logits = model(images)
			loss = criterion(logits, targets)
			top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))

			total_loss += loss.item() * images.size(0)
			total += images.size(0)
			correct1 += top1 * images.size(0)
			correct5 += top5 * images.size(0)

			all_preds.append(logits.argmax(dim=1).cpu())
			all_targets.append(targets.cpu())

	val_loss = total_loss / total
	val_acc1 = correct1 / total
	val_acc5 = correct5 / total

	split_label = "Test" if args.split == "test" else "Val"
	print(f"{split_label} loss {val_loss:.4f} | acc@1 {val_acc1:.3f} | acc@5 {val_acc5:.3f}")

	try:
		from sklearn.metrics import confusion_matrix

		preds = torch.cat(all_preds).numpy()
		targets = torch.cat(all_targets).numpy()
		cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
		print(f"Confusion matrix shape: {cm.shape}")

		if args.cm_out:
			import matplotlib

			matplotlib.use("Agg")
			import matplotlib.pyplot as plt

			cm_plot = cm.astype("float")
			if args.cm_normalize:
				row_sums = cm_plot.sum(axis=1, keepdims=True)
				row_sums[row_sums == 0] = 1.0
				cm_plot = cm_plot / row_sums

			fig_w = 12
			fig_h = 10
			plt.figure(figsize=(fig_w, fig_h))
			plt.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
			plt.title(f"Confusion Matrix ({split_label})")
			plt.colorbar(fraction=0.046, pad=0.04)
			tick_marks = list(range(len(class_names)))
			plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
			plt.yticks(tick_marks, class_names, fontsize=6)
			plt.ylabel("True label")
			plt.xlabel("Predicted label")
			plt.tight_layout()

			os.makedirs(os.path.dirname(args.cm_out) or ".", exist_ok=True)
			plt.savefig(args.cm_out, dpi=200)
			plt.close()
	except Exception:
		pass


if __name__ == "__main__":
	main()
