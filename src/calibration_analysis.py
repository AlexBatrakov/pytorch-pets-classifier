from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .calibration import (
	apply_temperature,
	build_reliability_bins,
	calibration_metrics,
	confidence_threshold_sweep,
	fit_temperature,
	top1_confidence_and_correctness,
)
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


def _collect_logits_and_targets(
	model: torch.nn.Module,
	loader,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
	all_logits: list[torch.Tensor] = []
	all_targets: list[torch.Tensor] = []
	model.eval()

	with torch.no_grad():
		for images, targets in loader:
			images = images.to(device)
			logits = model(images)
			all_logits.append(logits.detach().cpu())
			all_targets.append(targets.detach().cpu())

	if not all_logits:
		raise RuntimeError("No logits were collected from the evaluation loader.")
	return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def _plot_reliability_comparison(
	uncal_bins: list[dict[str, float | int]],
	cal_bins: list[dict[str, float | int]],
	out_path: str,
) -> None:
	if not uncal_bins or not cal_bins:
		raise ValueError("Reliability bins must not be empty.")

	fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
	series = [
		(axes[0], uncal_bins, "Uncalibrated"),
		(axes[1], cal_bins, "Calibrated"),
	]
	for ax, rows, title in series:
		centers = [(float(r["bin_lower"]) + float(r["bin_upper"])) / 2.0 for r in rows]
		width = max(0.02, 0.9 / max(1, len(rows)))
		accuracies = [float(r["accuracy"]) for r in rows]
		confidences = [float(r["mean_confidence"]) for r in rows]
		counts = [int(r["count"]) for r in rows]

		ax.bar(centers, accuracies, width=width, color="#0072b2", alpha=0.75, label="Bin accuracy")
		ax.plot(centers, confidences, marker="o", color="#d55e00", linewidth=1.6, label="Mean confidence")
		ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#555555", linewidth=1.0, label="Perfect calibration")
		ax.set_title(title)
		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.0)
		ax.set_xlabel("Confidence")
		ax.grid(alpha=0.2)

		non_empty = sum(1 for c in counts if c > 0)
		ax.text(
			0.02,
			0.04,
			f"non-empty bins: {non_empty}/{len(rows)}",
			transform=ax.transAxes,
			fontsize=8,
			color="#444444",
		)

	axes[0].set_ylabel("Accuracy")
	handles, labels = axes[1].get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
	fig.suptitle("Test Reliability Diagram: Before vs After Temperature Scaling", y=1.08, fontsize=13)
	fig.tight_layout()

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	fig.savefig(out_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def _plot_coverage_accuracy(
	threshold_rows: list[dict[str, float | int]],
	out_path: str,
) -> None:
	if not threshold_rows:
		raise ValueError("Threshold rows must not be empty.")

	thresholds = [float(r["threshold"]) for r in threshold_rows]
	coverages = [float(r["coverage"]) for r in threshold_rows]
	accuracies = [float(r["retained_accuracy"]) for r in threshold_rows]

	plt.figure(figsize=(10, 6))
	plt.plot(thresholds, coverages, label="Coverage", color="#009e73", linewidth=2.0)
	plt.plot(thresholds, accuracies, label="Retained accuracy", color="#cc79a7", linewidth=2.0)
	plt.xlim(0.0, 1.0)
	plt.ylim(0.0, 1.0)
	plt.xlabel("Confidence threshold (calibrated top-1)")
	plt.ylabel("Value")
	plt.title("Test Split: Coverage vs Retained Accuracy")
	plt.grid(alpha=0.2)
	plt.legend()
	plt.tight_layout()

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=180)
	plt.close()


def _copy_public_assets(
	reliability_src: str,
	coverage_src: str,
	public_assets_dir: str,
	public_prefix: str,
) -> tuple[str, str]:
	os.makedirs(public_assets_dir, exist_ok=True)
	reliability_dst = os.path.join(public_assets_dir, f"{public_prefix}_reliability_test.png")
	coverage_dst = os.path.join(public_assets_dir, f"{public_prefix}_coverage_accuracy_test.png")
	shutil.copy2(reliability_src, reliability_dst)
	shutil.copy2(coverage_src, coverage_dst)
	return reliability_dst, coverage_dst


def _round_metric_dict(metrics: dict[str, float], digits: int = 6) -> dict[str, float]:
	return {key: round(float(value), digits) for key, value in metrics.items()}


def _round_threshold_rows(
	rows: list[dict[str, float | int]],
	digits: int = 6,
) -> list[dict[str, float | int]]:
	result: list[dict[str, float | int]] = []
	for row in rows:
		result.append(
			{
				"threshold": round(float(row["threshold"]), digits),
				"retained_count": int(row["retained_count"]),
				"coverage": round(float(row["coverage"]), digits),
				"retained_accuracy": round(float(row["retained_accuracy"]), digits),
			}
		)
	return result


def main() -> None:
	parser = argparse.ArgumentParser(description="Run post-hoc calibration analysis for a checkpoint")
	parser.add_argument("--ckpt", default="checkpoints/best.pt")
	parser.add_argument("--config", default=None)
	parser.add_argument("--run-dir", required=True)
	parser.add_argument("--num-bins", type=int, default=15)
	parser.add_argument("--max-iter", type=int, default=100)
	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--public-assets-dir", default="docs/experiments/assets")
	parser.add_argument("--public-prefix", default="exp17_calibration")
	parser.add_argument("--skip-public-assets", action="store_true")
	args = parser.parse_args()

	ckpt = load_checkpoint(args.ckpt, map_location="cpu")
	if args.config:
		cfg = load_config(args.config)
		config_label = args.config
	else:
		cfg = ckpt.get("config", load_config("configs/default.yaml"))
		config_label = "from_checkpoint_config"
	# Calibration analysis is local post-hoc evaluation; avoid network fetch attempts.
	cfg.setdefault("data", {})["download"] = False

	set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
	device = get_device()

	test_loader, dataset_class_names = _resolve_eval_loader(cfg, "test")
	class_names = list(ckpt.get("class_names", dataset_class_names))
	if len(class_names) != len(dataset_class_names):
		raise ValueError(
			f"Checkpoint classes ({len(class_names)}) do not match dataset classes ({len(dataset_class_names)})."
		)

	model = build_model(num_classes=len(class_names), pretrained=False, freeze_backbone=False)
	model.load_state_dict(ckpt["model_state_dict"])
	model.to(device)
	model.eval()

	val_loader, _ = _resolve_eval_loader(cfg, "val")
	val_logits, val_targets = _collect_logits_and_targets(model=model, loader=val_loader, device=device)
	test_logits, test_targets = _collect_logits_and_targets(model=model, loader=test_loader, device=device)

	temperature = fit_temperature(
		val_logits=val_logits,
		val_targets=val_targets,
		max_iter=args.max_iter,
		lr=args.lr,
	)

	val_cal_logits = apply_temperature(val_logits, temperature)
	test_cal_logits = apply_temperature(test_logits, temperature)

	val_uncal_metrics = calibration_metrics(val_logits, val_targets, num_bins=args.num_bins)
	val_cal_metrics = calibration_metrics(val_cal_logits, val_targets, num_bins=args.num_bins)
	test_uncal_metrics = calibration_metrics(test_logits, test_targets, num_bins=args.num_bins)
	test_cal_metrics = calibration_metrics(test_cal_logits, test_targets, num_bins=args.num_bins)

	test_uncal_conf, test_uncal_corr = top1_confidence_and_correctness(test_logits, test_targets)
	test_cal_conf, test_cal_corr = top1_confidence_and_correctness(test_cal_logits, test_targets)
	test_uncal_bins = build_reliability_bins(test_uncal_conf, test_uncal_corr, num_bins=args.num_bins)
	test_cal_bins = build_reliability_bins(test_cal_conf, test_cal_corr, num_bins=args.num_bins)
	threshold_rows = confidence_threshold_sweep(test_cal_conf, test_cal_corr)

	run_artifacts_dir = os.path.join(args.run_dir, "artifacts", "calibration")
	run_assets_dir = os.path.join(args.run_dir, "assets", "calibration")
	os.makedirs(run_artifacts_dir, exist_ok=True)
	os.makedirs(run_assets_dir, exist_ok=True)

	reliability_path = os.path.join(run_assets_dir, "reliability_test.png")
	coverage_path = os.path.join(run_assets_dir, "coverage_accuracy_test.png")
	summary_path = os.path.join(run_artifacts_dir, "calibration_summary.json")

	_plot_reliability_comparison(uncal_bins=test_uncal_bins, cal_bins=test_cal_bins, out_path=reliability_path)
	_plot_coverage_accuracy(threshold_rows=threshold_rows, out_path=coverage_path)

	public_paths: dict[str, str] = {}
	if not args.skip_public_assets:
		public_rel, public_cov = _copy_public_assets(
			reliability_src=reliability_path,
			coverage_src=coverage_path,
			public_assets_dir=args.public_assets_dir,
			public_prefix=args.public_prefix,
		)
		public_paths = {
			"reliability_test_png": public_rel,
			"coverage_accuracy_test_png": public_cov,
		}

	summary = {
		"packet_id": "P-002",
		"ckpt": args.ckpt,
		"config": config_label,
		"run_dir": args.run_dir,
		"fit_split": "val",
		"report_splits": ["val", "test"],
		"temperature": round(float(temperature), 6),
		"num_bins": int(args.num_bins),
		"notes": [
			"Temperature scaling is fit on validation logits only; test split is report-only.",
			"Positive scalar temperature scaling preserves class ranking, so top-k ranking metrics should remain unchanged.",
		],
		"metrics": {
			"val": {
				"uncalibrated": _round_metric_dict(val_uncal_metrics),
				"calibrated": _round_metric_dict(val_cal_metrics),
			},
			"test": {
				"uncalibrated": _round_metric_dict(test_uncal_metrics),
				"calibrated": _round_metric_dict(test_cal_metrics),
			},
		},
		"threshold_sweep_test_calibrated": _round_threshold_rows(threshold_rows),
		"paths": {
			"summary_json": summary_path,
			"run_assets": {
				"reliability_test_png": reliability_path,
				"coverage_accuracy_test_png": coverage_path,
			},
			"public_assets": public_paths,
		},
	}
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print(f"Temperature (fit on val): {summary['temperature']:.6f}")
	print(
		"Val   uncal -> cal: "
		f"ECE {val_uncal_metrics['ece']:.4f} -> {val_cal_metrics['ece']:.4f}, "
		f"NLL {val_uncal_metrics['nll']:.4f} -> {val_cal_metrics['nll']:.4f}"
	)
	print(
		"Test  uncal -> cal: "
		f"ECE {test_uncal_metrics['ece']:.4f} -> {test_cal_metrics['ece']:.4f}, "
		f"NLL {test_uncal_metrics['nll']:.4f} -> {test_cal_metrics['nll']:.4f}"
	)
	print(f"Summary JSON: {summary_path}")
	print(f"Run-local plots: {reliability_path}, {coverage_path}")
	if public_paths:
		print(
			"Public plots: "
			f"{public_paths['reliability_test_png']}, {public_paths['coverage_accuracy_test_png']}"
		)


if __name__ == "__main__":
	main()
