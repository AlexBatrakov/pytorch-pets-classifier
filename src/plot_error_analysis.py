from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _load_csv_rows(path: str) -> list[dict[str, str]]:
	with open(path, newline="", encoding="utf-8") as f:
		return list(csv.DictReader(f))


def _infer_split_from_predictions_path(predictions_path: str) -> str:
	name = Path(predictions_path).name
	prefix = "predictions_"
	suffix = ".csv"
	if name.startswith(prefix) and name.endswith(suffix):
		return name[len(prefix) : -len(suffix)]
	return "unknown"


def _infer_related_csv_paths(
	predictions_path: str,
	per_class_csv: str | None,
	top_confusions_csv: str | None,
) -> tuple[str, str]:
	if per_class_csv and top_confusions_csv:
		return per_class_csv, top_confusions_csv

	split = _infer_split_from_predictions_path(predictions_path)
	base_dir = os.path.dirname(predictions_path) or "."

	if not per_class_csv:
		per_class_csv = os.path.join(base_dir, f"per_class_metrics_{split}.csv")
	if not top_confusions_csv:
		top_confusions_csv = os.path.join(base_dir, f"top_confusions_{split}.csv")
	return per_class_csv, top_confusions_csv


def _as_bool(value: str) -> bool:
	return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value: str, default: float = 0.0) -> float:
	try:
		return float(value)
	except Exception:
		return default


def _select_error_rows_for_gallery(rows: list[dict[str, str]], max_items: int) -> list[dict[str, str]]:
	errors = [r for r in rows if not _as_bool(r.get("is_correct", "0"))]
	errors.sort(key=lambda r: _safe_float(r.get("pred_confidence", "0")), reverse=True)
	return errors[: max(0, max_items)]


def _plot_per_class_acc(per_class_rows: list[dict[str, str]], out_path: str, worst_n: int = 15) -> None:
	if not per_class_rows:
		return

	rows = sorted(per_class_rows, key=lambda r: float(r["acc1"]))[: max(1, worst_n)]
	labels = [r["class_name"] for r in rows]
	values = [float(r["acc1"]) for r in rows]
	supports = [int(r["support"]) for r in rows]

	fig_h = max(6, 0.4 * len(rows) + 2)
	plt.figure(figsize=(10, fig_h))
	y = list(range(len(rows)))
	bars = plt.barh(y, values, color="#d55e00", alpha=0.85)
	plt.yticks(y, labels, fontsize=9)
	plt.xlim(0.0, 1.0)
	plt.xlabel("Accuracy (Top-1)")
	plt.title(f"Worst-{len(rows)} Classes by Top-1 Accuracy")
	plt.gca().invert_yaxis()
	plt.grid(axis="x", alpha=0.25)

	for bar, acc, support in zip(bars, values, supports):
		plt.text(
			min(acc + 0.01, 0.98),
			bar.get_y() + bar.get_height() / 2,
			f"{acc:.2f} (n={support})",
			va="center",
			fontsize=8,
		)

	plt.tight_layout()
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=180)
	plt.close()


def _plot_top_confusions(top_conf_rows: list[dict[str, str]], out_path: str, top_n: int = 10) -> None:
	if not top_conf_rows:
		return

	rows = top_conf_rows[: max(1, top_n)]
	labels = [f"{r['true_label']} -> {r['pred_label']}" for r in rows]
	counts = [int(r["count"]) for r in rows]
	rates = [float(r["row_normalized_rate"]) for r in rows]

	fig_h = max(6, 0.5 * len(rows) + 2)
	plt.figure(figsize=(12, fig_h))
	y = list(range(len(rows)))
	bars = plt.barh(y, counts, color="#0072b2", alpha=0.85)
	plt.yticks(y, labels, fontsize=8)
	plt.xlabel("Count of mistakes")
	plt.title(f"Top-{len(rows)} Confusion Pairs (Test Errors)")
	plt.gca().invert_yaxis()
	plt.grid(axis="x", alpha=0.25)

	for bar, count, rate in zip(bars, counts, rates):
		plt.text(
			count + 0.2,
			bar.get_y() + bar.get_height() / 2,
			f"{count} ({rate:.2%})",
			va="center",
			fontsize=8,
		)

	plt.tight_layout()
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=180)
	plt.close()


def _plot_confidence_hist(pred_rows: list[dict[str, str]], out_path: str, log_y: bool = False) -> None:
	if not pred_rows:
		return

	correct = []
	incorrect = []
	for row in pred_rows:
		conf = _safe_float(row.get("pred_confidence", "0"))
		if _as_bool(row.get("is_correct", "0")):
			correct.append(conf)
		else:
			incorrect.append(conf)

	plt.figure(figsize=(10, 6))
	num_bins = 20
	# Use shared fixed bins for both histograms to make the distributions comparable.
	bins = [i / num_bins for i in range(num_bins + 1)]
	if correct:
		plt.hist(
			correct,
			bins=bins,
			alpha=0.55,
			label=f"Correct (n={len(correct)})",
			color="#009e73",
			edgecolor="white",
		)
	if incorrect:
		plt.hist(
			incorrect,
			bins=bins,
			alpha=0.6,
			label=f"Incorrect (n={len(incorrect)})",
			color="#cc79a7",
			edgecolor="white",
		)
	plt.xlabel("Top-1 predicted confidence")
	plt.ylabel("Count")
	title = "Confidence Distribution: Correct vs Incorrect"
	if log_y:
		plt.yscale("log")
		title += " (log Y)"
	plt.title(title)
	plt.grid(alpha=0.2)
	plt.legend()
	plt.tight_layout()

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=180)
	plt.close()


def _plot_error_gallery(
	pred_rows: list[dict[str, str]],
	out_path: str,
	gallery_count: int = 16,
	cols: int = 4,
) -> None:
	rows = _select_error_rows_for_gallery(pred_rows, gallery_count)
	if not rows:
		return

	cols = max(1, cols)
	n = len(rows)
	nrows = math.ceil(n / cols)
	fig, axes = plt.subplots(nrows=nrows, ncols=cols, figsize=(4 * cols, 3.4 * nrows))
	axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

	for ax, row in zip(axes_list, rows):
		img_path = row.get("image_path", "")
		try:
			img = mpimg.imread(img_path)
			ax.imshow(img)
		except Exception:
			ax.text(0.5, 0.5, "Image load failed", ha="center", va="center", fontsize=10)
			ax.set_facecolor("#f0f0f0")

		true_label = row.get("true_label", "?")
		pred_label = row.get("pred_label", "?")
		conf = _safe_float(row.get("pred_confidence", "0"))
		ax.set_title(f"T: {true_label}\nP: {pred_label} ({conf:.2f})", fontsize=9)
		ax.axis("off")

	for ax in axes_list[n:]:
		ax.axis("off")

	fig.suptitle("Top Overconfident Mistakes (by predicted confidence)", fontsize=14)
	plt.tight_layout()
	fig.subplots_adjust(top=0.92)

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=180)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot error analysis charts from exported predictions CSV")
	parser.add_argument("--predictions", required=True, help="Path to predictions_<split>.csv")
	parser.add_argument("--per-class-csv", default=None, help="Optional path to per_class_metrics_<split>.csv")
	parser.add_argument("--top-confusions-csv", default=None, help="Optional path to top_confusions_<split>.csv")
	parser.add_argument("--out-dir", required=True, help="Output directory for plots")
	parser.add_argument("--worst-classes", type=int, default=15)
	parser.add_argument("--top-confusions", type=int, default=10)
	parser.add_argument("--gallery-count", type=int, default=16)
	parser.add_argument("--gallery-cols", type=int, default=4)
	args = parser.parse_args()

	per_class_csv, top_confusions_csv = _infer_related_csv_paths(
		args.predictions, args.per_class_csv, args.top_confusions_csv
	)

	pred_rows = _load_csv_rows(args.predictions)
	per_class_rows = _load_csv_rows(per_class_csv)
	top_conf_rows = _load_csv_rows(top_confusions_csv)

	os.makedirs(args.out_dir, exist_ok=True)
	out1 = os.path.join(args.out_dir, "per_class_acc.png")
	out2 = os.path.join(args.out_dir, "confusion_top_pairs.png")
	out3 = os.path.join(args.out_dir, "conf_error_hist.png")
	out3_log = os.path.join(args.out_dir, "conf_error_hist_log.png")
	out4 = os.path.join(args.out_dir, "error_gallery.png")

	_plot_per_class_acc(per_class_rows, out1, worst_n=args.worst_classes)
	_plot_top_confusions(top_conf_rows, out2, top_n=args.top_confusions)
	_plot_confidence_hist(pred_rows, out3, log_y=False)
	_plot_confidence_hist(pred_rows, out3_log, log_y=True)
	_plot_error_gallery(pred_rows, out4, gallery_count=args.gallery_count, cols=args.gallery_cols)

	print(f"Saved: {out1}")
	print(f"Saved: {out2}")
	print(f"Saved: {out3}")
	print(f"Saved: {out3_log}")
	print(f"Saved: {out4}")


if __name__ == "__main__":
	main()
