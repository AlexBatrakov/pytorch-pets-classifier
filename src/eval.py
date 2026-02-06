from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .config import load_config
from .data import build_loaders
from .model import build_model
from .utils import accuracy_topk, get_device, set_seed


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate pets classifier")
	parser.add_argument("--ckpt", default="checkpoints/best.pt")
	parser.add_argument("--config", default=None)
	args = parser.parse_args()

	ckpt = torch.load(args.ckpt, map_location="cpu")
	if args.config:
		cfg = load_config(args.config)
	else:
		cfg = ckpt.get("config", load_config("configs/default.yaml"))

	set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
	device = get_device()

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

	print(f"Val loss {val_loss:.4f} | acc@1 {val_acc1:.3f} | acc@5 {val_acc5:.3f}")

	try:
		from sklearn.metrics import confusion_matrix

		preds = torch.cat(all_preds).numpy()
		targets = torch.cat(all_targets).numpy()
		cm = confusion_matrix(targets, preds)
		print(f"Confusion matrix shape: {cm.shape}")
	except Exception:
		pass


if __name__ == "__main__":
	main()
