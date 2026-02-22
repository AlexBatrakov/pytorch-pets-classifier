from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F

from .config import load_config
from .data import build_loaders, build_test_loader
from .model import build_model
from .utils import get_device, load_checkpoint, set_seed


def _resolve_eval_loader(cfg: dict[str, Any], split: str):
	if split == "test":
		loader, class_names = build_test_loader(cfg)
		return loader, class_names
	_, loader, class_names = build_loaders(cfg)
	return loader, class_names


def _resolve_loader_image_paths(loader) -> list[str]:
	"""
	Resolve image paths in the exact iteration order of the evaluation loader.

	For `val`, the loader dataset is a `Subset` of the trainval dataset.
	For `test`, the loader dataset is the dataset itself.
	"""
	ds = loader.dataset

	# torch.utils.data.Subset stores the original dataset and selected indices.
	if hasattr(ds, "dataset") and hasattr(ds, "indices"):
		base_ds = ds.dataset
		if hasattr(base_ds, "_images"):
			return [str(base_ds._images[i]) for i in ds.indices]

	if hasattr(ds, "_images"):
		return [str(p) for p in ds._images]

	raise AttributeError(
		"Could not resolve image paths from loader.dataset. "
		"Expected torchvision OxfordIIITPet dataset with `_images` attribute."
	)


def _write_predictions_csv(path: str, rows: list[dict[str, Any]]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	fieldnames = [
		"row_index",
		"image_path",
		"true_idx",
		"true_label",
		"pred_idx",
		"pred_label",
		"is_correct",
		"pred_confidence",
		"true_class_confidence",
		"true_rank",
		"is_top5_correct",
		"top5_indices",
		"top5_labels",
		"top5_confidences",
	]
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _build_per_class_rows(
	class_names: list[str],
	supports: list[int],
	correct1_counts: list[int],
	correct5_counts: list[int],
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for class_idx, class_name in enumerate(class_names):
		support = int(supports[class_idx])
		correct1 = int(correct1_counts[class_idx])
		correct5 = int(correct5_counts[class_idx])
		acc1 = (correct1 / support) if support else 0.0
		acc5 = (correct5 / support) if support else 0.0
		rows.append(
			{
				"class_idx": class_idx,
				"class_name": class_name,
				"support": support,
				"correct1_count": correct1,
				"correct5_count": correct5,
				"error_count": support - correct1,
				"acc1": f"{acc1:.6f}",
				"acc5": f"{acc5:.6f}",
			}
		)

	# Hardest classes first (lowest acc1), then by support descending.
	rows.sort(key=lambda r: (float(r["acc1"]), -int(r["support"]), str(r["class_name"])))
	return rows


def _build_top_confusion_rows(
	class_names: list[str],
	confusion_counts: Counter[tuple[int, int]],
	supports: list[int],
	limit: int = 20,
) -> list[dict[str, Any]]:
	items = []
	for (true_idx, pred_idx), count in confusion_counts.items():
		support_true = int(supports[true_idx]) if true_idx < len(supports) else 0
		rate = (count / support_true) if support_true else 0.0
		items.append((true_idx, pred_idx, int(count), support_true, rate))

	items.sort(key=lambda x: (-x[2], -x[4], x[0], x[1]))
	rows: list[dict[str, Any]] = []
	for rank, (true_idx, pred_idx, count, support_true, rate) in enumerate(items[: max(0, limit)], start=1):
		rows.append(
			{
				"rank": rank,
				"true_idx": true_idx,
				"true_label": class_names[true_idx],
				"pred_idx": pred_idx,
				"pred_label": class_names[pred_idx],
				"count": count,
				"support_true": support_true,
				"row_normalized_rate": f"{rate:.6f}",
			}
		)
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description="Export per-sample predictions for error analysis")
	parser.add_argument("--ckpt", default="checkpoints/best.pt")
	parser.add_argument("--config", default=None)
	parser.add_argument("--split", default="test", choices=["val", "test"])
	parser.add_argument(
		"--out-dir",
		default="artifacts/error_analysis",
		help="Directory to save predictions_<split>.csv and summary JSON",
	)
	parser.add_argument(
		"--topk",
		type=int,
		default=5,
		help="Top-K predictions to export for each sample (default: 5)",
	)
	parser.add_argument(
		"--top-confusions-limit",
		type=int,
		default=20,
		help="How many top confusion pairs to save (default: 20)",
	)
	args = parser.parse_args()

	ckpt = load_checkpoint(args.ckpt, map_location="cpu")
	if args.config:
		cfg = load_config(args.config)
	else:
		cfg = ckpt.get("config", load_config("configs/default.yaml"))

	set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
	device = get_device()

	loader, dataset_class_names = _resolve_eval_loader(cfg, args.split)
	class_names = list(ckpt.get("class_names", dataset_class_names))
	if len(class_names) != len(dataset_class_names):
		raise ValueError(
			f"Checkpoint classes ({len(class_names)}) do not match dataset classes ({len(dataset_class_names)})."
		)

	model = build_model(num_classes=len(class_names), pretrained=False, freeze_backbone=False)
	model.load_state_dict(ckpt["model_state_dict"])
	model.to(device)
	model.eval()

	image_paths = _resolve_loader_image_paths(loader)
	if len(image_paths) != len(loader.dataset):
		raise RuntimeError(
			f"Resolved {len(image_paths)} image paths, but loader dataset has {len(loader.dataset)} samples."
		)

	topk = max(1, min(int(args.topk), len(class_names)))
	rows: list[dict[str, Any]] = []
	row_offset = 0
	correct = 0
	total = 0
	top5_correct = 0
	num_classes = len(class_names)
	per_class_supports = [0] * num_classes
	per_class_correct1 = [0] * num_classes
	per_class_correct5 = [0] * num_classes
	confusion_counts: Counter[tuple[int, int]] = Counter()

	with torch.no_grad():
		for images, targets in loader:
			images = images.to(device)
			targets = targets.to(device)

			logits = model(images)
			probs = F.softmax(logits, dim=1)

			k = min(topk, probs.size(1))
			top_probs, top_indices = probs.topk(k, dim=1, largest=True, sorted=True)
			pred_indices = top_indices[:, 0]
			pred_probs = top_probs[:, 0]

			batch_size = images.size(0)
			batch_paths = image_paths[row_offset : row_offset + batch_size]
			if len(batch_paths) != batch_size:
				raise RuntimeError("Failed to align image paths with prediction batch.")

			for i in range(batch_size):
				target_idx = int(targets[i].item())
				pred_idx = int(pred_indices[i].item())
				top_idx_list = [int(x) for x in top_indices[i].tolist()]
				top_prob_list = [float(x) for x in top_probs[i].tolist()]
				top_label_list = [class_names[idx] for idx in top_idx_list]

				# Rank of the true class among all classes (1-based) is useful for later analysis.
				target_prob = float(probs[i, target_idx].item())
				true_rank = 1 + int((probs[i] > probs[i, target_idx]).sum().item())

				is_correct = int(pred_idx == target_idx)
				is_top5_correct = int(target_idx in top_idx_list[: min(5, len(top_idx_list))])
				per_class_supports[target_idx] += 1
				per_class_correct1[target_idx] += is_correct
				per_class_correct5[target_idx] += is_top5_correct
				if not is_correct:
					confusion_counts[(target_idx, pred_idx)] += 1

				rows.append(
					{
						"row_index": row_offset + i,
						"image_path": batch_paths[i],
						"true_idx": target_idx,
						"true_label": class_names[target_idx],
						"pred_idx": pred_idx,
						"pred_label": class_names[pred_idx],
						"is_correct": is_correct,
						"pred_confidence": f"{float(pred_probs[i].item()):.6f}",
						"true_class_confidence": f"{target_prob:.6f}",
						"true_rank": true_rank,
						"is_top5_correct": is_top5_correct,
						"top5_indices": json.dumps(top_idx_list, ensure_ascii=True),
						"top5_labels": json.dumps(top_label_list, ensure_ascii=True),
						"top5_confidences": json.dumps([round(p, 6) for p in top_prob_list], ensure_ascii=True),
					}
				)

				correct += is_correct
				top5_correct += is_top5_correct
				total += 1

			row_offset += batch_size

	out_csv = os.path.join(args.out_dir, f"predictions_{args.split}.csv")
	_write_predictions_csv(out_csv, rows)

	per_class_rows = _build_per_class_rows(
		class_names=class_names,
		supports=per_class_supports,
		correct1_counts=per_class_correct1,
		correct5_counts=per_class_correct5,
	)
	per_class_csv = os.path.join(args.out_dir, f"per_class_metrics_{args.split}.csv")
	_write_csv(
		per_class_csv,
		fieldnames=[
			"class_idx",
			"class_name",
			"support",
			"correct1_count",
			"correct5_count",
			"error_count",
			"acc1",
			"acc5",
		],
		rows=per_class_rows,
	)

	top_conf_rows = _build_top_confusion_rows(
		class_names=class_names,
		confusion_counts=confusion_counts,
		supports=per_class_supports,
		limit=args.top_confusions_limit,
	)
	top_confusions_csv = os.path.join(args.out_dir, f"top_confusions_{args.split}.csv")
	_write_csv(
		top_confusions_csv,
		fieldnames=[
			"rank",
			"true_idx",
			"true_label",
			"pred_idx",
			"pred_label",
			"count",
			"support_true",
			"row_normalized_rate",
		],
		rows=top_conf_rows,
	)

	summary = {
		"split": args.split,
		"num_samples": total,
		"acc1": float(correct / total) if total else 0.0,
		"acc5": float(top5_correct / total) if total else 0.0,
		"ckpt": args.ckpt,
		"topk_exported": topk,
		"csv_path": out_csv,
		"per_class_metrics_csv": per_class_csv,
		"top_confusions_csv": top_confusions_csv,
		"top_confusions_limit": int(args.top_confusions_limit),
	}
	out_summary = os.path.join(args.out_dir, f"summary_{args.split}.json")
	os.makedirs(args.out_dir, exist_ok=True)
	with open(out_summary, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	split_label = "Test" if args.split == "test" else "Val"
	print(f"{split_label} predictions exported: {out_csv}")
	print(f"Per-class metrics saved: {per_class_csv}")
	print(f"Top confusions saved: {top_confusions_csv}")
	print(
		f"Samples: {total} | acc@1 {summary['acc1']:.3f} | acc@5 {summary['acc5']:.3f} "
		f"| summary: {out_summary}"
	)


if __name__ == "__main__":
	main()
