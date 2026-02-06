from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from .config import apply_overrides, load_config
from .data import build_loaders
from .model import build_model
from .utils import AverageMeter, accuracy_topk, get_device, set_seed


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
	for name, param in model.named_parameters():
		if name.startswith("fc."):
			param.requires_grad = True
		else:
			param.requires_grad = trainable


def _build_optimizer_and_scheduler(
	model: nn.Module, train_cfg: Dict
) -> Tuple[AdamW, object]:
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = AdamW(
		params,
		lr=train_cfg.get("lr", 3e-4),
		weight_decay=train_cfg.get("weight_decay", 0.01),
	)

	sched_cfg = train_cfg.get("scheduler", {})
	name = (sched_cfg.get("name", "none") or "none").lower()
	scheduler = None
	if name == "step":
		scheduler = StepLR(
			optimizer,
			step_size=sched_cfg.get("step_size", 5),
			gamma=sched_cfg.get("gamma", 0.1),
		)
	elif name == "cosine":
		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=sched_cfg.get("t_max", 10),
		)

	return optimizer, scheduler


def _train_one_epoch(model, loader, criterion, optimizer, device):
	model.train()
	losses = AverageMeter()
	acc1 = AverageMeter()
	acc5 = AverageMeter()

	pbar = tqdm(loader, desc="train", leave=False)
	for images, targets in pbar:
		images = images.to(device)
		targets = targets.to(device)

		optimizer.zero_grad(set_to_none=True)
		logits = model(images)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()

		top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))
		losses.update(loss.item(), images.size(0))
		acc1.update(top1, images.size(0))
		acc5.update(top5, images.size(0))
		pbar.set_postfix(loss=losses.avg, acc1=acc1.avg)

	return losses.avg, acc1.avg, acc5.avg


def _eval_one_epoch(model, loader, criterion, device):
	model.eval()
	losses = AverageMeter()
	acc1 = AverageMeter()
	acc5 = AverageMeter()

	with torch.no_grad():
		pbar = tqdm(loader, desc="val", leave=False)
		for images, targets in pbar:
			images = images.to(device)
			targets = targets.to(device)

			logits = model(images)
			loss = criterion(logits, targets)
			top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))

			losses.update(loss.item(), images.size(0))
			acc1.update(top1, images.size(0))
			acc5.update(top5, images.size(0))
			pbar.set_postfix(loss=losses.avg, acc1=acc1.avg)

	return losses.avg, acc1.avg, acc5.avg


def main() -> None:
	parser = argparse.ArgumentParser(description="Train pets classifier")
	parser.add_argument("--config", default="configs/default.yaml")
	parser.add_argument("--epochs", type=int)
	parser.add_argument("--batch-size", type=int)
	parser.add_argument("--lr", type=float)
	parser.add_argument("--freeze-backbone", action="store_true")
	parser.add_argument("--freeze-epochs", type=int)
	parser.add_argument("--num-workers", type=int)
	parser.add_argument("--seed", type=int)
	args = parser.parse_args()

	cfg = load_config(args.config)
	overrides = {
		"train.epochs": args.epochs,
		"data.batch_size": args.batch_size,
		"train.lr": args.lr,
		"train.freeze_backbone": args.freeze_backbone if args.freeze_backbone else None,
		"train.freeze_epochs": args.freeze_epochs,
		"data.num_workers": args.num_workers,
		"seed": args.seed,
	}
	cfg = apply_overrides(cfg, overrides)

	set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
	device = get_device()

	train_loader, val_loader, class_names = build_loaders(cfg)

	train_cfg = cfg["train"]
	freeze_epochs = int(train_cfg.get("freeze_epochs", 0) or 0)
	initial_freeze = bool(train_cfg.get("freeze_backbone", False) or freeze_epochs > 0)

	model = build_model(num_classes=len(class_names), pretrained=True, freeze_backbone=initial_freeze)
	model.to(device)

	if initial_freeze:
		_set_backbone_trainable(model, False)

	criterion = nn.CrossEntropyLoss()
	optimizer, scheduler = _build_optimizer_and_scheduler(model, train_cfg)

	best_val_acc = 0.0
	best_path = os.path.join(cfg["paths"]["checkpoints_dir"], "best.pt")
	last_path = os.path.join(cfg["paths"]["checkpoints_dir"], "last.pt")
	os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)

	epochs = int(train_cfg.get("epochs", 1))
	for epoch in range(epochs):
		if freeze_epochs > 0 and epoch == freeze_epochs:
			_set_backbone_trainable(model, True)
			optimizer, scheduler = _build_optimizer_and_scheduler(model, train_cfg)

		train_loss, train_acc1, train_acc5 = _train_one_epoch(
			model, train_loader, criterion, optimizer, device
		)
		val_loss, val_acc1, val_acc5 = _eval_one_epoch(
			model, val_loader, criterion, device
		)

		if scheduler is not None:
			scheduler.step()

		is_best = val_acc1 > best_val_acc
		if is_best:
			best_val_acc = val_acc1
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"class_names": class_names,
					"config": cfg,
					"epoch": epoch + 1,
					"best_val_acc": best_val_acc,
				},
				best_path,
			)

		torch.save(
			{
				"model_state_dict": model.state_dict(),
				"class_names": class_names,
				"config": cfg,
				"epoch": epoch + 1,
				"best_val_acc": best_val_acc,
			},
			last_path,
		)

		print(
			f"Epoch {epoch + 1}/{epochs} | "
			f"train loss {train_loss:.4f} acc@1 {train_acc1:.3f} acc@5 {train_acc5:.3f} | "
			f"val loss {val_loss:.4f} acc@1 {val_acc1:.3f} acc@5 {val_acc5:.3f}"
		)

	print(f"Best checkpoint saved to: {best_path}")
	print(f"Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
	main()
