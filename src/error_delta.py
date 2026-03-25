from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIGH_CONFIDENCE_THRESHOLDS = (0.90, 0.95, 0.99)


def _load_csv_rows(path: str) -> list[dict[str, str]]:
	with open(path, newline="", encoding="utf-8") as f:
		return list(csv.DictReader(f))


def _load_json(path: str) -> dict[str, Any]:
	with open(path, encoding="utf-8") as f:
		return json.load(f)


def _safe_int(value: Any) -> int:
	return int(str(value).strip())


def _safe_float(value: Any) -> float:
	return float(str(value).strip())


def _fmt_float(value: float, digits: int = 6) -> str:
	return f"{value:.{digits}f}"


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _load_predictions(path: str) -> list[dict[str, Any]]:
	rows = _load_csv_rows(path)
	parsed: list[dict[str, Any]] = []
	for row in rows:
		parsed.append(
			{
				"row_index": _safe_int(row["row_index"]),
				"image_path": row["image_path"],
				"true_idx": _safe_int(row["true_idx"]),
				"true_label": row["true_label"],
				"pred_idx": _safe_int(row["pred_idx"]),
				"pred_label": row["pred_label"],
				"is_correct": _safe_int(row["is_correct"]),
				"pred_confidence": _safe_float(row["pred_confidence"]),
				"true_class_confidence": _safe_float(row["true_class_confidence"]),
				"true_rank": _safe_int(row["true_rank"]),
				"is_top5_correct": _safe_int(row["is_top5_correct"]),
			}
		)
	return parsed


def _infer_run_name(error_analysis_dir: str) -> str:
	path = Path(error_analysis_dir).resolve(strict=False)
	return path.parent.parent.name


def _infer_alias(run_name: str) -> str:
	return run_name.split("_", 1)[0] if "_" in run_name else run_name


def _transition_label(before: int, after: int) -> str:
	return f"{'C' if before else 'W'}->{'C' if after else 'W'}"


def _top5_transition_label(before: int, after: int) -> str:
	return f"{'T' if before else 'F'}->{'T' if after else 'F'}"


def _validate_unique_image_paths(rows: list[dict[str, Any]], label: str) -> None:
	paths = [row["image_path"] for row in rows]
	if len(paths) != len(set(paths)):
		raise ValueError(f"{label} predictions contain duplicate image_path values.")


def _join_prediction_rows(
	baseline_rows: list[dict[str, Any]],
	candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
	if len(baseline_rows) != len(candidate_rows):
		raise ValueError(
			f"Prediction row count mismatch: baseline={len(baseline_rows)} candidate={len(candidate_rows)}."
		)

	_validate_unique_image_paths(baseline_rows, "Baseline")
	_validate_unique_image_paths(candidate_rows, "Candidate")

	candidate_by_image = {row["image_path"]: row for row in candidate_rows}
	joined: list[dict[str, Any]] = []
	for baseline in baseline_rows:
		image_path = baseline["image_path"]
		candidate = candidate_by_image.get(image_path)
		if candidate is None:
			raise ValueError(f"Candidate predictions missing image_path: {image_path}")

		for key in ("row_index", "true_idx", "true_label"):
			if baseline[key] != candidate[key]:
				raise ValueError(
					f"Joined row mismatch for {image_path}: field={key} "
					f"baseline={baseline[key]!r} candidate={candidate[key]!r}"
				)

		joined.append({"image_path": image_path, "baseline": baseline, "candidate": candidate})
	return joined


def _build_sample_transition_rows(
	joined_rows: list[dict[str, Any]],
	baseline_alias: str,
	candidate_alias: str,
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for joined in joined_rows:
		baseline = joined["baseline"]
		candidate = joined["candidate"]
		rows.append(
			{
				"row_index": baseline["row_index"],
				"image_path": baseline["image_path"],
				"true_idx": baseline["true_idx"],
				"true_label": baseline["true_label"],
				f"{baseline_alias}_pred_idx": baseline["pred_idx"],
				f"{baseline_alias}_pred_label": baseline["pred_label"],
				f"{baseline_alias}_is_correct": baseline["is_correct"],
				f"{baseline_alias}_pred_confidence": _fmt_float(baseline["pred_confidence"]),
				f"{baseline_alias}_true_class_confidence": _fmt_float(baseline["true_class_confidence"]),
				f"{baseline_alias}_true_rank": baseline["true_rank"],
				f"{baseline_alias}_is_top5_correct": baseline["is_top5_correct"],
				f"{candidate_alias}_pred_idx": candidate["pred_idx"],
				f"{candidate_alias}_pred_label": candidate["pred_label"],
				f"{candidate_alias}_is_correct": candidate["is_correct"],
				f"{candidate_alias}_pred_confidence": _fmt_float(candidate["pred_confidence"]),
				f"{candidate_alias}_true_class_confidence": _fmt_float(candidate["true_class_confidence"]),
				f"{candidate_alias}_true_rank": candidate["true_rank"],
				f"{candidate_alias}_is_top5_correct": candidate["is_top5_correct"],
				"outcome_transition": _transition_label(baseline["is_correct"], candidate["is_correct"]),
				"top5_transition": _top5_transition_label(
					baseline["is_top5_correct"], candidate["is_top5_correct"]
				),
				"true_rank_delta": candidate["true_rank"] - baseline["true_rank"],
				"pred_confidence_delta": _fmt_float(candidate["pred_confidence"] - baseline["pred_confidence"]),
				"true_class_confidence_delta": _fmt_float(
					candidate["true_class_confidence"] - baseline["true_class_confidence"]
				),
			}
		)
	return rows


def _build_per_class_delta_rows(
	joined_rows: list[dict[str, Any]],
	baseline_alias: str,
	candidate_alias: str,
) -> list[dict[str, Any]]:
	aggregates: dict[str, dict[str, Any]] = {}
	for joined in joined_rows:
		baseline = joined["baseline"]
		candidate = joined["candidate"]
		class_name = baseline["true_label"]
		record = aggregates.setdefault(
			class_name,
			{
				"class_idx": baseline["true_idx"],
				"class_name": class_name,
				"support": 0,
				f"{baseline_alias}_correct1_count": 0,
				f"{candidate_alias}_correct1_count": 0,
				f"{baseline_alias}_correct5_count": 0,
				f"{candidate_alias}_correct5_count": 0,
				"fixed_count": 0,
				"regressed_count": 0,
				"persistent_error_count": 0,
			},
		)
		record["support"] += 1
		record[f"{baseline_alias}_correct1_count"] += baseline["is_correct"]
		record[f"{candidate_alias}_correct1_count"] += candidate["is_correct"]
		record[f"{baseline_alias}_correct5_count"] += baseline["is_top5_correct"]
		record[f"{candidate_alias}_correct5_count"] += candidate["is_top5_correct"]
		if not baseline["is_correct"] and candidate["is_correct"]:
			record["fixed_count"] += 1
		elif baseline["is_correct"] and not candidate["is_correct"]:
			record["regressed_count"] += 1
		elif not baseline["is_correct"] and not candidate["is_correct"]:
			record["persistent_error_count"] += 1

	rows: list[dict[str, Any]] = []
	for class_name, record in aggregates.items():
		support = int(record["support"])
		baseline_acc1 = record[f"{baseline_alias}_correct1_count"] / support
		candidate_acc1 = record[f"{candidate_alias}_correct1_count"] / support
		baseline_acc5 = record[f"{baseline_alias}_correct5_count"] / support
		candidate_acc5 = record[f"{candidate_alias}_correct5_count"] / support
		rows.append(
			{
				"class_idx": record["class_idx"],
				"class_name": class_name,
				"support": support,
				f"{baseline_alias}_correct1_count": record[f"{baseline_alias}_correct1_count"],
				f"{candidate_alias}_correct1_count": record[f"{candidate_alias}_correct1_count"],
				f"{baseline_alias}_correct5_count": record[f"{baseline_alias}_correct5_count"],
				f"{candidate_alias}_correct5_count": record[f"{candidate_alias}_correct5_count"],
				f"{baseline_alias}_error_count": support - record[f"{baseline_alias}_correct1_count"],
				f"{candidate_alias}_error_count": support - record[f"{candidate_alias}_correct1_count"],
				f"{baseline_alias}_acc1": _fmt_float(baseline_acc1),
				f"{candidate_alias}_acc1": _fmt_float(candidate_acc1),
				"acc1_delta": _fmt_float(candidate_acc1 - baseline_acc1),
				f"{baseline_alias}_acc5": _fmt_float(baseline_acc5),
				f"{candidate_alias}_acc5": _fmt_float(candidate_acc5),
				"acc5_delta": _fmt_float(candidate_acc5 - baseline_acc5),
				"fixed_count": record["fixed_count"],
				"regressed_count": record["regressed_count"],
				"persistent_error_count": record["persistent_error_count"],
			}
		)

	rows.sort(key=lambda row: (-_safe_float(row["acc1_delta"]), row["class_name"]))
	return rows


def _build_confusion_counts(prediction_rows: list[dict[str, Any]]) -> tuple[Counter[tuple[str, str]], dict[str, int]]:
	counts: Counter[tuple[str, str]] = Counter()
	support_by_class: dict[str, int] = {}
	for row in prediction_rows:
		true_label = row["true_label"]
		support_by_class[true_label] = support_by_class.get(true_label, 0) + 1
		if not row["is_correct"]:
			counts[(true_label, row["pred_label"])] += 1
	return counts, support_by_class


def _build_confusion_delta_rows(
	baseline_rows: list[dict[str, Any]],
	candidate_rows: list[dict[str, Any]],
	baseline_alias: str,
	candidate_alias: str,
) -> list[dict[str, Any]]:
	baseline_counts, baseline_supports = _build_confusion_counts(baseline_rows)
	candidate_counts, candidate_supports = _build_confusion_counts(candidate_rows)
	if baseline_supports != candidate_supports:
		raise ValueError("Support mismatch between baseline and candidate predictions.")

	rows: list[dict[str, Any]] = []
	for true_label, pred_label in sorted(set(baseline_counts) | set(candidate_counts)):
		baseline_count = int(baseline_counts.get((true_label, pred_label), 0))
		candidate_count = int(candidate_counts.get((true_label, pred_label), 0))
		support_true = int(baseline_supports.get(true_label, 0))
		baseline_rate = (baseline_count / support_true) if support_true else 0.0
		candidate_rate = (candidate_count / support_true) if support_true else 0.0
		rows.append(
			{
				"true_label": true_label,
				"pred_label": pred_label,
				"support_true": support_true,
				f"{baseline_alias}_count": baseline_count,
				f"{candidate_alias}_count": candidate_count,
				"count_delta": candidate_count - baseline_count,
				f"{baseline_alias}_rate": _fmt_float(baseline_rate),
				f"{candidate_alias}_rate": _fmt_float(candidate_rate),
				"rate_delta": _fmt_float(candidate_rate - baseline_rate),
			}
		)

	rows.sort(
		key=lambda row: (
			-abs(_safe_int(row["count_delta"])),
			-safe_float_abs(row["rate_delta"]),
			row["true_label"],
			row["pred_label"],
		)
	)
	return rows


def safe_float_abs(value: Any) -> float:
	return abs(_safe_float(value))


def _top5_rescues(prediction_rows: list[dict[str, Any]]) -> tuple[int, int]:
	error_rows = [row for row in prediction_rows if not row["is_correct"]]
	return sum(row["is_top5_correct"] for row in error_rows), len(error_rows)


def _high_confidence_error_counts(prediction_rows: list[dict[str, Any]]) -> dict[str, int]:
	counts: dict[str, int] = {}
	for threshold in HIGH_CONFIDENCE_THRESHOLDS:
		counts[f"{threshold:.2f}"] = sum(
			1
			for row in prediction_rows
			if (not row["is_correct"]) and row["pred_confidence"] >= threshold
		)
	return counts


def _validate_summary_metrics(
	summary_payload: dict[str, Any],
	prediction_rows: list[dict[str, Any]],
	label: str,
) -> None:
	num_samples = len(prediction_rows)
	acc1 = sum(row["is_correct"] for row in prediction_rows) / num_samples
	acc5 = sum(row["is_top5_correct"] for row in prediction_rows) / num_samples
	if int(summary_payload["num_samples"]) != num_samples:
		raise ValueError(f"{label} summary num_samples mismatch.")
	if abs(float(summary_payload["acc1"]) - acc1) > 1e-9:
		raise ValueError(f"{label} summary acc1 mismatch.")
	if abs(float(summary_payload["acc5"]) - acc5) > 1e-9:
		raise ValueError(f"{label} summary acc5 mismatch.")


def _validate_per_class_metrics(
	per_class_rows: list[dict[str, str]],
	prediction_rows: list[dict[str, Any]],
	label: str,
) -> None:
	aggregates: dict[str, dict[str, int]] = {}
	for row in prediction_rows:
		record = aggregates.setdefault(
			row["true_label"],
			{"support": 0, "correct1_count": 0, "correct5_count": 0},
		)
		record["support"] += 1
		record["correct1_count"] += row["is_correct"]
		record["correct5_count"] += row["is_top5_correct"]

	for row in per_class_rows:
		class_name = row["class_name"]
		if class_name not in aggregates:
			raise ValueError(f"{label} per-class row references unknown class: {class_name}")
		record = aggregates[class_name]
		if int(row["support"]) != record["support"]:
			raise ValueError(f"{label} per-class support mismatch for {class_name}")
		if int(row["correct1_count"]) != record["correct1_count"]:
			raise ValueError(f"{label} per-class correct1 mismatch for {class_name}")
		if int(row["correct5_count"]) != record["correct5_count"]:
			raise ValueError(f"{label} per-class correct5 mismatch for {class_name}")


def _validate_top_confusions(
	top_conf_rows: list[dict[str, str]],
	prediction_rows: list[dict[str, Any]],
	label: str,
) -> None:
	confusion_counts, supports = _build_confusion_counts(prediction_rows)
	for row in top_conf_rows:
		true_label = row["true_label"]
		pred_label = row["pred_label"]
		expected_count = confusion_counts.get((true_label, pred_label), 0)
		expected_support = supports.get(true_label, 0)
		expected_rate = (expected_count / expected_support) if expected_support else 0.0
		if int(row["count"]) != expected_count:
			raise ValueError(f"{label} top confusion count mismatch for {true_label} -> {pred_label}")
		if int(row["support_true"]) != expected_support:
			raise ValueError(f"{label} top confusion support mismatch for {true_label} -> {pred_label}")
		if abs(float(row["row_normalized_rate"]) - expected_rate) > 1e-6:
			raise ValueError(f"{label} top confusion rate mismatch for {true_label} -> {pred_label}")


def _build_summary(
	baseline_run: str,
	candidate_run: str,
	baseline_alias: str,
	candidate_alias: str,
	joined_rows: list[dict[str, Any]],
	baseline_rows: list[dict[str, Any]],
	candidate_rows: list[dict[str, Any]],
	per_class_rows: list[dict[str, Any]],
	confusion_rows: list[dict[str, Any]],
	source_paths: dict[str, str],
) -> dict[str, Any]:
	transition_counts = Counter()
	top5_transition_counts = Counter()
	persistent_rank_delta = Counter({"improved": 0, "same": 0, "worse": 0})
	for joined in joined_rows:
		baseline = joined["baseline"]
		candidate = joined["candidate"]
		transition_counts[_transition_label(baseline["is_correct"], candidate["is_correct"])] += 1
		top5_transition_counts[
			_top5_transition_label(baseline["is_top5_correct"], candidate["is_top5_correct"])
		] += 1
		if not baseline["is_correct"] and not candidate["is_correct"]:
			if candidate["true_rank"] < baseline["true_rank"]:
				persistent_rank_delta["improved"] += 1
			elif candidate["true_rank"] > baseline["true_rank"]:
				persistent_rank_delta["worse"] += 1
			else:
				persistent_rank_delta["same"] += 1

	baseline_rescues, baseline_errors = _top5_rescues(baseline_rows)
	candidate_rescues, candidate_errors = _top5_rescues(candidate_rows)
	baseline_high_conf = _high_confidence_error_counts(baseline_rows)
	candidate_high_conf = _high_confidence_error_counts(candidate_rows)

	top_class_improvements = [
		{
			"class_name": row["class_name"],
			f"{baseline_alias}_acc1": float(row[f"{baseline_alias}_acc1"]),
			f"{candidate_alias}_acc1": float(row[f"{candidate_alias}_acc1"]),
			"acc1_delta": float(row["acc1_delta"]),
			"fixed_count": int(row["fixed_count"]),
			"regressed_count": int(row["regressed_count"]),
		}
		for row in per_class_rows[:5]
	]
	top_class_regressions = [
		{
			"class_name": row["class_name"],
			f"{baseline_alias}_acc1": float(row[f"{baseline_alias}_acc1"]),
			f"{candidate_alias}_acc1": float(row[f"{candidate_alias}_acc1"]),
			"acc1_delta": float(row["acc1_delta"]),
			"fixed_count": int(row["fixed_count"]),
			"regressed_count": int(row["regressed_count"]),
		}
		for row in sorted(per_class_rows, key=lambda item: (float(item["acc1_delta"]), item["class_name"]))[:5]
	]

	improved_confusions = [
		{
			"true_label": row["true_label"],
			"pred_label": row["pred_label"],
			f"{baseline_alias}_count": int(row[f"{baseline_alias}_count"]),
			f"{candidate_alias}_count": int(row[f"{candidate_alias}_count"]),
			"count_delta": int(row["count_delta"]),
			"rate_delta": float(row["rate_delta"]),
		}
		for row in sorted(
			[row for row in confusion_rows if int(row["count_delta"]) < 0],
			key=lambda item: (int(item["count_delta"]), item["true_label"], item["pred_label"]),
		)[:5]
	]
	regressed_confusions = [
		{
			"true_label": row["true_label"],
			"pred_label": row["pred_label"],
			f"{baseline_alias}_count": int(row[f"{baseline_alias}_count"]),
			f"{candidate_alias}_count": int(row[f"{candidate_alias}_count"]),
			"count_delta": int(row["count_delta"]),
			"rate_delta": float(row["rate_delta"]),
		}
		for row in sorted(
			[row for row in confusion_rows if int(row["count_delta"]) > 0],
			key=lambda item: (-int(item["count_delta"]), item["true_label"], item["pred_label"]),
		)[:5]
	]

	improved_classes = sum(float(row["acc1_delta"]) > 0 for row in per_class_rows)
	regressed_classes = sum(float(row["acc1_delta"]) < 0 for row in per_class_rows)
	unchanged_classes = len(per_class_rows) - improved_classes - regressed_classes

	return {
		"packet_id": "P-004",
		"baseline_run": baseline_run,
		"candidate_run": candidate_run,
		"baseline_alias": baseline_alias,
		"candidate_alias": candidate_alias,
		"source_paths": source_paths,
		"num_samples": len(joined_rows),
		"metrics": {
			baseline_alias: {
				"acc1": sum(row["is_correct"] for row in baseline_rows) / len(baseline_rows),
				"acc5": sum(row["is_top5_correct"] for row in baseline_rows) / len(baseline_rows),
				"top1_errors": baseline_errors,
				"top5_rescues_among_errors": baseline_rescues,
				"top5_rescue_rate_among_errors": (baseline_rescues / baseline_errors) if baseline_errors else 0.0,
			},
			candidate_alias: {
				"acc1": sum(row["is_correct"] for row in candidate_rows) / len(candidate_rows),
				"acc5": sum(row["is_top5_correct"] for row in candidate_rows) / len(candidate_rows),
				"top1_errors": candidate_errors,
				"top5_rescues_among_errors": candidate_rescues,
				"top5_rescue_rate_among_errors": (candidate_rescues / candidate_errors) if candidate_errors else 0.0,
			},
		},
		"transition_counts": {key: int(transition_counts.get(key, 0)) for key in ("C->C", "C->W", "W->C", "W->W")},
		"top5_transition_counts": {
			key: int(top5_transition_counts.get(key, 0)) for key in ("T->T", "T->F", "F->T", "F->F")
		},
		"persistent_error_rank_delta": dict(persistent_rank_delta),
		"class_delta_overview": {
			"improved_classes": improved_classes,
			"regressed_classes": regressed_classes,
			"unchanged_classes": unchanged_classes,
		},
		"high_confidence_error_counts": {
			threshold: {
				baseline_alias: baseline_high_conf[threshold],
				candidate_alias: candidate_high_conf[threshold],
				"delta": candidate_high_conf[threshold] - baseline_high_conf[threshold],
			}
			for threshold in baseline_high_conf
		},
		"top_class_improvements": top_class_improvements,
		"top_class_regressions": top_class_regressions,
		"top_confusion_improvements": improved_confusions,
		"top_confusion_regressions": regressed_confusions,
	}


def _plot_transition_counts(summary: dict[str, Any], out_path: str) -> None:
	labels = ["C->C", "C->W", "W->C", "W->W"]
	values = [int(summary["transition_counts"][label]) for label in labels]
	colors = ["#0072b2", "#d55e00", "#009e73", "#7f7f7f"]

	plt.figure(figsize=(8, 5))
	bars = plt.bar(labels, values, color=colors, alpha=0.9)
	plt.ylabel("Sample count")
	plt.title("Outcome Transitions on the Shared Test Split")
	plt.grid(axis="y", alpha=0.2)
	offset = max(0.1, max(values) * 0.03) if values else 0.1
	plt.ylim(0, max(values) + offset * 4 if values else 1)
	for bar, value in zip(bars, values):
		plt.text(bar.get_x() + bar.get_width() / 2, value + offset, str(value), ha="center", va="bottom", fontsize=9)
	plt.tight_layout()
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=180)
	plt.close()


def _select_class_delta_rows(per_class_rows: list[dict[str, Any]], limit_each: int = 6) -> list[dict[str, Any]]:
	negative = sorted(per_class_rows, key=lambda row: (float(row["acc1_delta"]), row["class_name"]))[:limit_each]
	positive = sorted(per_class_rows, key=lambda row: (-float(row["acc1_delta"]), row["class_name"]))[:limit_each]
	selected = negative + positive
	deduped: list[dict[str, Any]] = []
	seen: set[str] = set()
	for row in selected:
		if row["class_name"] in seen:
			continue
		seen.add(row["class_name"])
		deduped.append(row)
	deduped.sort(key=lambda row: float(row["acc1_delta"]))
	return deduped


def _plot_class_acc1_delta(per_class_rows: list[dict[str, Any]], out_path: str) -> None:
	rows = _select_class_delta_rows(per_class_rows)
	labels = [row["class_name"] for row in rows]
	values = [float(row["acc1_delta"]) * 100.0 for row in rows]
	colors = ["#009e73" if value >= 0 else "#d55e00" for value in values]
	min_value = min(values) if values else -1.0
	max_value = max(values) if values else 1.0
	padding = max(0.6, (max_value - min_value) * 0.08)
	text_offset = max(0.2, padding * 0.25)

	fig_h = max(6, 0.45 * len(rows) + 2)
	plt.figure(figsize=(10, fig_h))
	y = list(range(len(rows)))
	bars = plt.barh(y, values, color=colors, alpha=0.9)
	plt.yticks(y, labels, fontsize=9)
	plt.axvline(0.0, color="#333333", linewidth=1.0)
	plt.xlim(min_value - padding, max_value + padding)
	plt.xlabel("Top-1 accuracy delta (percentage points, exp17 - exp02)")
	plt.title("Largest Class-Level Top-1 Accuracy Shifts")
	plt.grid(axis="x", alpha=0.2)
	for bar, value in zip(bars, values):
		x = value + (text_offset if value >= 0 else -text_offset)
		ha = "left" if value >= 0 else "right"
		plt.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.1f}", va="center", ha=ha, fontsize=8)
	plt.tight_layout()
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=180)
	plt.close()


def _plot_confusion_delta(confusion_rows: list[dict[str, Any]], out_path: str, limit: int = 12) -> None:
	rows = [row for row in confusion_rows if int(row["count_delta"]) != 0][:limit]
	labels = [f"{row['true_label']} -> {row['pred_label']}" for row in rows]
	values = [int(row["count_delta"]) for row in rows]
	colors = ["#d55e00" if value > 0 else "#009e73" for value in values]
	min_value = min(values) if values else -1
	max_value = max(values) if values else 1
	padding = max(0.5, (max_value - min_value) * 0.08)
	text_offset = max(0.15, padding * 0.2)

	fig_h = max(6, 0.45 * len(rows) + 2)
	plt.figure(figsize=(12, fig_h))
	y = list(range(len(rows)))
	bars = plt.barh(y, values, color=colors, alpha=0.9)
	plt.yticks(y, labels, fontsize=8)
	plt.axvline(0.0, color="#333333", linewidth=1.0)
	plt.xlim(min_value - padding, max_value + padding)
	plt.xlabel("Confusion count delta (exp17 - exp02)")
	plt.title("Largest Confusion-Pair Shifts")
	plt.grid(axis="x", alpha=0.2)
	for bar, value in zip(bars, values):
		x = value + (text_offset if value >= 0 else -text_offset)
		ha = "left" if value >= 0 else "right"
		plt.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+d}", va="center", ha=ha, fontsize=8)
	plt.tight_layout()
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	plt.savefig(out_path, dpi=180)
	plt.close()


def _resolve_input_paths(error_analysis_dir: str) -> dict[str, str]:
	return {
		"predictions": os.path.join(error_analysis_dir, "predictions_test.csv"),
		"summary": os.path.join(error_analysis_dir, "summary_test.json"),
		"per_class": os.path.join(error_analysis_dir, "per_class_metrics_test.csv"),
		"top_confusions": os.path.join(error_analysis_dir, "top_confusions_test.csv"),
	}


def _write_summary_json(path: str, payload: dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Compare exp17 vs exp02 error-analysis exports on the shared test split")
	parser.add_argument("--baseline-dir", required=True, help="Path to baseline artifacts/error_analysis directory")
	parser.add_argument("--candidate-dir", required=True, help="Path to candidate artifacts/error_analysis directory")
	parser.add_argument("--out-dir", required=True, help="Output directory for machine-readable comparison artifacts")
	parser.add_argument("--public-assets-dir", required=True, help="Output directory for public plots")
	parser.add_argument("--public-prefix", default="exp17_vs_exp02", help="Filename prefix for public plots")
	args = parser.parse_args(argv)

	baseline_run = _infer_run_name(args.baseline_dir)
	candidate_run = _infer_run_name(args.candidate_dir)
	baseline_alias = _infer_alias(baseline_run)
	candidate_alias = _infer_alias(candidate_run)

	baseline_paths = _resolve_input_paths(args.baseline_dir)
	candidate_paths = _resolve_input_paths(args.candidate_dir)

	baseline_rows = _load_predictions(baseline_paths["predictions"])
	candidate_rows = _load_predictions(candidate_paths["predictions"])
	joined_rows = _join_prediction_rows(baseline_rows, candidate_rows)

	_validate_summary_metrics(_load_json(baseline_paths["summary"]), baseline_rows, "Baseline")
	_validate_summary_metrics(_load_json(candidate_paths["summary"]), candidate_rows, "Candidate")
	_validate_per_class_metrics(_load_csv_rows(baseline_paths["per_class"]), baseline_rows, "Baseline")
	_validate_per_class_metrics(_load_csv_rows(candidate_paths["per_class"]), candidate_rows, "Candidate")
	_validate_top_confusions(_load_csv_rows(baseline_paths["top_confusions"]), baseline_rows, "Baseline")
	_validate_top_confusions(_load_csv_rows(candidate_paths["top_confusions"]), candidate_rows, "Candidate")

	sample_rows = _build_sample_transition_rows(joined_rows, baseline_alias=baseline_alias, candidate_alias=candidate_alias)
	per_class_rows = _build_per_class_delta_rows(joined_rows, baseline_alias=baseline_alias, candidate_alias=candidate_alias)
	confusion_rows = _build_confusion_delta_rows(
		baseline_rows,
		candidate_rows,
		baseline_alias=baseline_alias,
		candidate_alias=candidate_alias,
	)

	summary = _build_summary(
		baseline_run=baseline_run,
		candidate_run=candidate_run,
		baseline_alias=baseline_alias,
		candidate_alias=candidate_alias,
		joined_rows=joined_rows,
		baseline_rows=baseline_rows,
		candidate_rows=candidate_rows,
		per_class_rows=per_class_rows,
		confusion_rows=confusion_rows,
		source_paths={
			"baseline_predictions": baseline_paths["predictions"],
			"candidate_predictions": candidate_paths["predictions"],
			"baseline_summary": baseline_paths["summary"],
			"candidate_summary": candidate_paths["summary"],
			"baseline_per_class": baseline_paths["per_class"],
			"candidate_per_class": candidate_paths["per_class"],
			"baseline_top_confusions": baseline_paths["top_confusions"],
			"candidate_top_confusions": candidate_paths["top_confusions"],
		},
	)

	os.makedirs(args.out_dir, exist_ok=True)
	_write_csv(
		os.path.join(args.out_dir, "sample_transitions.csv"),
		fieldnames=list(sample_rows[0].keys()),
		rows=sample_rows,
	)
	_write_csv(
		os.path.join(args.out_dir, "per_class_delta.csv"),
		fieldnames=list(per_class_rows[0].keys()),
		rows=per_class_rows,
	)
	_write_csv(
		os.path.join(args.out_dir, "confusion_delta.csv"),
		fieldnames=list(confusion_rows[0].keys()),
		rows=confusion_rows,
	)
	_write_summary_json(os.path.join(args.out_dir, "summary.json"), summary)

	os.makedirs(args.public_assets_dir, exist_ok=True)
	transition_path = os.path.join(args.public_assets_dir, f"{args.public_prefix}_transition_counts.png")
	class_delta_path = os.path.join(args.public_assets_dir, f"{args.public_prefix}_class_acc1_delta.png")
	confusion_delta_path = os.path.join(args.public_assets_dir, f"{args.public_prefix}_confusion_delta.png")
	_plot_transition_counts(summary, transition_path)
	_plot_class_acc1_delta(per_class_rows, class_delta_path)
	_plot_confusion_delta(confusion_rows, confusion_delta_path)

	print(f"Sample transitions saved: {os.path.join(args.out_dir, 'sample_transitions.csv')}")
	print(f"Per-class delta saved: {os.path.join(args.out_dir, 'per_class_delta.csv')}")
	print(f"Confusion delta saved: {os.path.join(args.out_dir, 'confusion_delta.csv')}")
	print(f"Summary saved: {os.path.join(args.out_dir, 'summary.json')}")
	print(f"Public assets saved: {transition_path}, {class_delta_path}, {confusion_delta_path}")


if __name__ == "__main__":
	main()
