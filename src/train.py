from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

from .config import apply_overrides, load_config
from .data import build_loaders
from .model import build_model
from .utils import AverageMeter, accuracy_topk, get_device, set_seed


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _get_git_commit_hash() -> str:
	try:
		out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
		return out or "unknown"
	except Exception:
		return "unknown"


def _count_parameters(model: nn.Module) -> Tuple[int, int]:
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return total, trainable


def _build_checkpoint_payload(
	model: nn.Module,
	class_names: list[str],
	cfg: Dict[str, Any],
	epoch: int,
	best_val_acc: float,
	epoch_metrics: Dict[str, float],
	run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
	return {
		"model_state_dict": model.state_dict(),
		"class_names": class_names,
		"config": cfg,
		"epoch": epoch,
		"best_val_acc": best_val_acc,
		"epoch_metrics": epoch_metrics,
		"run_metadata": run_metadata,
		"saved_at_utc": _utc_now_iso(),
	}


def _update_best_val_acc(best_val_acc: float, current_val_acc: float) -> Tuple[float, bool]:
	is_best = current_val_acc > best_val_acc
	if is_best:
		return current_val_acc, True
	return best_val_acc, False


def _get_metric_value(metrics: Dict[str, float], metric_name: str) -> float:
	if metric_name not in metrics:
		raise ValueError(f"Unknown metric '{metric_name}'. Available: {sorted(metrics.keys())}")
	return float(metrics[metric_name])


def _is_metric_improved(
	current: float,
	best: float,
	mode: str,
	min_delta: float,
) -> bool:
	mode_norm = mode.lower()
	if mode_norm == "max":
		return current > best + min_delta
	if mode_norm == "min":
		return current < best - min_delta
	raise ValueError(f"Unsupported early stopping mode '{mode}'. Use 'max' or 'min'.")


def _init_metrics_csv(path: str) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"epoch",
				"train_loss",
				"train_acc1",
				"train_acc5",
				"val_loss",
				"val_acc1",
				"val_acc5",
				"lr",
			]
		)


def _append_metrics_csv(
	path: str,
	epoch: int,
	train_loss: float,
	train_acc1: float,
	train_acc5: float,
	val_loss: float,
	val_acc1: float,
	val_acc5: float,
	lr: float,
) -> None:
	with open(path, "a", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				epoch,
				f"{train_loss:.6f}",
				f"{train_acc1:.6f}",
				f"{train_acc5:.6f}",
				f"{val_loss:.6f}",
				f"{val_acc1:.6f}",
				f"{val_acc5:.6f}",
				f"{lr:.8f}",
			]
		)


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
	elif name == "plateau":
		scheduler = ReduceLROnPlateau(
			optimizer,
			mode=str(sched_cfg.get("mode", "min")),
			factor=float(sched_cfg.get("factor", 0.5)),
			patience=int(sched_cfg.get("patience", 2)),
			min_lr=float(sched_cfg.get("min_lr", 1e-6)),
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
	early_cfg = train_cfg.get("early_stopping", {}) or {}
	es_enabled = bool(early_cfg.get("enabled", False))
	es_monitor = str(early_cfg.get("monitor", "val_acc1"))
	es_mode = str(early_cfg.get("mode", "max"))
	es_patience = int(early_cfg.get("patience", 3))
	es_min_delta = float(early_cfg.get("min_delta", 0.0))

	best_es_metric = float("-inf") if es_mode.lower() == "max" else float("inf")
	epochs_without_improve = 0

	model = build_model(num_classes=len(class_names), pretrained=True, freeze_backbone=initial_freeze)
	model.to(device)

	if initial_freeze:
		_set_backbone_trainable(model, False)

	criterion = nn.CrossEntropyLoss()
	optimizer, scheduler = _build_optimizer_and_scheduler(model, train_cfg)

	best_val_acc = 0.0
	best_path = os.path.join(cfg["paths"]["checkpoints_dir"], "best.pt")
	last_path = os.path.join(cfg["paths"]["checkpoints_dir"], "last.pt")
	metrics_path = os.path.join(cfg["paths"].get("artifacts_dir", "./artifacts"), "metrics.csv")
	os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
	_init_metrics_csv(metrics_path)

	total_params, trainable_params = _count_parameters(model)
	run_metadata: Dict[str, Any] = {
		"created_at_utc": _utc_now_iso(),
		"git_commit": _get_git_commit_hash(),
		"device": str(device),
		"torch_version": str(torch.__version__),
		"python_version": platform.python_version(),
		"platform": platform.platform(),
		"total_params": total_params,
		"trainable_params": trainable_params,
		"command": " ".join(sys.argv),
		"metrics_csv": metrics_path,
		"config_path": args.config,
		"early_stopping": {
			"enabled": es_enabled,
			"monitor": es_monitor,
			"mode": es_mode,
			"patience": es_patience,
			"min_delta": es_min_delta,
		},
	}

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

		# Log LR actually used during this epoch. Scheduler updates LR for the next epoch.
		current_lr = optimizer.param_groups[0]["lr"]
		epoch_metrics = {
			"train_loss": train_loss,
			"train_acc1": train_acc1,
			"train_acc5": train_acc5,
			"val_loss": val_loss,
			"val_acc1": val_acc1,
			"val_acc5": val_acc5,
			"lr": current_lr,
		}

		_append_metrics_csv(
			path=metrics_path,
			epoch=epoch + 1,
			train_loss=train_loss,
			train_acc1=train_acc1,
			train_acc5=train_acc5,
			val_loss=val_loss,
			val_acc1=val_acc1,
			val_acc5=val_acc5,
			lr=current_lr,
		)

		if scheduler is not None:
			if isinstance(scheduler, ReduceLROnPlateau):
				sched_monitor = str(train_cfg.get("scheduler", {}).get("monitor", "val_loss"))
				scheduler.step(_get_metric_value(epoch_metrics, sched_monitor))
			else:
				scheduler.step()

		best_val_acc, is_best = _update_best_val_acc(best_val_acc, val_acc1)
		if is_best:
			best_payload = _build_checkpoint_payload(
				model=model,
				class_names=class_names,
				cfg=cfg,
				epoch=epoch + 1,
				best_val_acc=best_val_acc,
				epoch_metrics=epoch_metrics,
				run_metadata=run_metadata,
			)
			torch.save(best_payload, best_path)

		last_payload = _build_checkpoint_payload(
			model=model,
			class_names=class_names,
			cfg=cfg,
			epoch=epoch + 1,
			best_val_acc=best_val_acc,
			epoch_metrics=epoch_metrics,
			run_metadata=run_metadata,
		)
		torch.save(last_payload, last_path)

		print(
			f"Epoch {epoch + 1}/{epochs} | "
			f"train loss {train_loss:.4f} acc@1 {train_acc1:.3f} acc@5 {train_acc5:.3f} | "
			f"val loss {val_loss:.4f} acc@1 {val_acc1:.3f} acc@5 {val_acc5:.3f}"
		)

		if es_enabled:
			current_es_metric = _get_metric_value(epoch_metrics, es_monitor)
			if _is_metric_improved(current_es_metric, best_es_metric, es_mode, es_min_delta):
				best_es_metric = current_es_metric
				epochs_without_improve = 0
			else:
				epochs_without_improve += 1
				if epochs_without_improve >= es_patience:
					print(
						"Early stopping triggered: "
						f"no improvement in {es_monitor} for {es_patience} epochs."
					)
					break

	print(f"Best checkpoint saved to: {best_path}")
	print(f"Best val acc: {best_val_acc:.4f}")
	print(f"Epoch metrics saved to: {metrics_path}")


if __name__ == "__main__":
	main()
