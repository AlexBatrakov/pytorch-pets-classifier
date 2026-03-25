import csv
import json
from collections import Counter
from pathlib import Path

from src.error_delta import (
	_build_confusion_delta_rows,
	_build_per_class_delta_rows,
	_build_sample_transition_rows,
	_build_summary,
	_join_prediction_rows,
	_load_predictions,
	main,
)

PREDICTION_FIELDNAMES = [
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


def _prediction_row(
	row_index: int,
	image_path: str,
	true_idx: int,
	true_label: str,
	pred_idx: int,
	pred_label: str,
	is_correct: int,
	pred_confidence: float,
	true_class_confidence: float,
	true_rank: int,
	is_top5_correct: int,
) -> dict[str, str]:
	return {
		"row_index": str(row_index),
		"image_path": image_path,
		"true_idx": str(true_idx),
		"true_label": true_label,
		"pred_idx": str(pred_idx),
		"pred_label": pred_label,
		"is_correct": str(is_correct),
		"pred_confidence": f"{pred_confidence:.6f}",
		"true_class_confidence": f"{true_class_confidence:.6f}",
		"true_rank": str(true_rank),
		"is_top5_correct": str(is_top5_correct),
		"top5_indices": "[]",
		"top5_labels": "[]",
		"top5_confidences": "[]",
	}


def _write_predictions_csv(path: Path, rows: list[dict[str, str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES)
		writer.writeheader()
		writer.writerows(rows)


def _build_sidecars(prediction_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, float | int]]:
	per_class: dict[str, dict[str, int | float]] = {}
	confusions: Counter[tuple[str, str]] = Counter()
	supports: Counter[str] = Counter()
	acc1 = 0
	acc5 = 0
	for row in prediction_rows:
		true_label = row["true_label"]
		supports[true_label] += 1
		record = per_class.setdefault(
			true_label,
			{
				"class_idx": int(row["true_idx"]),
				"class_name": true_label,
				"support": 0,
				"correct1_count": 0,
				"correct5_count": 0,
			},
		)
		record["support"] += 1
		record["correct1_count"] += int(row["is_correct"])
		record["correct5_count"] += int(row["is_top5_correct"])
		acc1 += int(row["is_correct"])
		acc5 += int(row["is_top5_correct"])
		if int(row["is_correct"]) == 0:
			confusions[(true_label, row["pred_label"])] += 1

	per_class_rows = []
	for record in per_class.values():
		support = int(record["support"])
		correct1 = int(record["correct1_count"])
		correct5 = int(record["correct5_count"])
		per_class_rows.append(
			{
				"class_idx": str(record["class_idx"]),
				"class_name": str(record["class_name"]),
				"support": str(support),
				"correct1_count": str(correct1),
				"correct5_count": str(correct5),
				"error_count": str(support - correct1),
				"acc1": f"{correct1 / support:.6f}",
				"acc5": f"{correct5 / support:.6f}",
			}
		)

	confusion_rows = []
	for rank, ((true_label, pred_label), count) in enumerate(
		sorted(confusions.items(), key=lambda item: (-item[1], item[0][0], item[0][1])),
		start=1,
	):
		support_true = supports[true_label]
		confusion_rows.append(
			{
				"rank": str(rank),
				"true_idx": str(next(int(row["true_idx"]) for row in prediction_rows if row["true_label"] == true_label)),
				"true_label": true_label,
				"pred_idx": str(next(int(row["pred_idx"]) for row in prediction_rows if row["pred_label"] == pred_label)),
				"pred_label": pred_label,
				"count": str(count),
				"support_true": str(support_true),
				"row_normalized_rate": f"{count / support_true:.6f}",
			}
		)

	summary = {
		"num_samples": len(prediction_rows),
		"acc1": acc1 / len(prediction_rows),
		"acc5": acc5 / len(prediction_rows),
	}
	return per_class_rows, confusion_rows, summary


def _write_sidecars(base_dir: Path, prediction_rows: list[dict[str, str]]) -> None:
	per_class_rows, confusion_rows, summary = _build_sidecars(prediction_rows)
	with (base_dir / "summary_test.json").open("w", encoding="utf-8") as f:
		json.dump(summary, f)
	with (base_dir / "per_class_metrics_test.csv").open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
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
		)
		writer.writeheader()
		writer.writerows(per_class_rows)
	with (base_dir / "top_confusions_test.csv").open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
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
		)
		writer.writeheader()
		writer.writerows(confusion_rows)


def _build_fixture_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
	baseline_rows = [
		_prediction_row(0, "img1.jpg", 0, "A", 0, "A", 1, 0.90, 0.90, 1, 1),
		_prediction_row(1, "img2.jpg", 0, "A", 1, "B", 0, 0.80, 0.20, 2, 1),
		_prediction_row(2, "img3.jpg", 1, "B", 1, "B", 1, 0.88, 0.88, 1, 1),
		_prediction_row(3, "img4.jpg", 1, "B", 0, "A", 0, 0.95, 0.01, 6, 0),
		_prediction_row(4, "img5.jpg", 2, "C", 0, "A", 0, 0.92, 0.05, 3, 1),
	]
	candidate_rows = [
		_prediction_row(0, "img1.jpg", 0, "A", 0, "A", 1, 0.92, 0.92, 1, 1),
		_prediction_row(1, "img2.jpg", 0, "A", 0, "A", 1, 0.70, 0.70, 1, 1),
		_prediction_row(2, "img3.jpg", 1, "B", 0, "A", 0, 0.81, 0.12, 2, 1),
		_prediction_row(3, "img4.jpg", 1, "B", 0, "A", 0, 0.85, 0.20, 4, 1),
		_prediction_row(4, "img5.jpg", 2, "C", 2, "C", 1, 0.89, 0.89, 1, 1),
	]
	return baseline_rows, candidate_rows


def test_load_predictions_parses_numeric_columns(tmp_path):
	rows, _ = _build_fixture_rows()
	path = tmp_path / "predictions_test.csv"
	_write_predictions_csv(path, rows)

	loaded = _load_predictions(str(path))
	assert loaded[0]["row_index"] == 0
	assert loaded[1]["pred_idx"] == 1
	assert loaded[3]["pred_confidence"] == 0.95
	assert loaded[3]["true_rank"] == 6


def test_join_prediction_rows_validates_identity_mismatches():
	baseline_rows, candidate_rows = _build_fixture_rows()
	loaded_baseline = _load_predictions_from_rows(baseline_rows)
	loaded_candidate = _load_predictions_from_rows(candidate_rows)
	loaded_candidate[0]["row_index"] = 99

	try:
		_join_prediction_rows(loaded_baseline, loaded_candidate)
		assert False, "Expected identity mismatch to raise ValueError"
	except ValueError as exc:
		assert "row_index" in str(exc)


def test_build_sample_transition_rows_counts_outcomes():
	baseline_rows, candidate_rows = _build_fixture_rows()
	joined = _join_prediction_rows(_load_predictions_from_rows(baseline_rows), _load_predictions_from_rows(candidate_rows))
	rows = _build_sample_transition_rows(joined, baseline_alias="exp02", candidate_alias="exp17")
	counts = Counter(row["outcome_transition"] for row in rows)
	top5_counts = Counter(row["top5_transition"] for row in rows)

	assert counts == Counter({"W->C": 2, "C->C": 1, "C->W": 1, "W->W": 1})
	assert top5_counts == Counter({"T->T": 4, "F->T": 1})


def test_build_per_class_delta_rows_tracks_fixed_and_regressed():
	baseline_rows, candidate_rows = _build_fixture_rows()
	joined = _join_prediction_rows(_load_predictions_from_rows(baseline_rows), _load_predictions_from_rows(candidate_rows))
	rows = _build_per_class_delta_rows(joined, baseline_alias="exp02", candidate_alias="exp17")
	by_class = {row["class_name"]: row for row in rows}

	assert by_class["A"]["acc1_delta"] == "0.500000"
	assert by_class["A"]["fixed_count"] == 1
	assert by_class["B"]["acc1_delta"] == "-0.500000"
	assert by_class["B"]["regressed_count"] == 1
	assert by_class["B"]["persistent_error_count"] == 1
	assert by_class["C"]["acc1_delta"] == "1.000000"


def test_build_confusion_delta_rows_uses_full_prediction_union():
	baseline_rows, candidate_rows = _build_fixture_rows()
	rows = _build_confusion_delta_rows(
		_load_predictions_from_rows(baseline_rows),
		_load_predictions_from_rows(candidate_rows),
		baseline_alias="exp02",
		candidate_alias="exp17",
	)
	by_pair = {(row["true_label"], row["pred_label"]): row for row in rows}

	assert by_pair[("A", "B")]["count_delta"] == -1
	assert by_pair[("C", "A")]["count_delta"] == -1
	assert by_pair[("B", "A")]["count_delta"] == 1


def test_build_summary_reports_known_transition_counts():
	baseline_rows, candidate_rows = _build_fixture_rows()
	loaded_baseline = _load_predictions_from_rows(baseline_rows)
	loaded_candidate = _load_predictions_from_rows(candidate_rows)
	joined = _join_prediction_rows(loaded_baseline, loaded_candidate)
	per_class_rows = _build_per_class_delta_rows(joined, baseline_alias="exp02", candidate_alias="exp17")
	confusion_rows = _build_confusion_delta_rows(
		loaded_baseline,
		loaded_candidate,
		baseline_alias="exp02",
		candidate_alias="exp17",
	)
	summary = _build_summary(
		baseline_run="exp02_example",
		candidate_run="exp17_example",
		baseline_alias="exp02",
		candidate_alias="exp17",
		joined_rows=joined,
		baseline_rows=loaded_baseline,
		candidate_rows=loaded_candidate,
		per_class_rows=per_class_rows,
		confusion_rows=confusion_rows,
		source_paths={"baseline_predictions": "a", "candidate_predictions": "b"},
	)

	assert summary["transition_counts"] == {"C->C": 1, "C->W": 1, "W->C": 2, "W->W": 1}
	assert summary["top5_transition_counts"] == {"T->T": 4, "T->F": 0, "F->T": 1, "F->F": 0}
	assert summary["class_delta_overview"] == {"improved_classes": 2, "regressed_classes": 1, "unchanged_classes": 0}


def test_main_writes_expected_outputs(tmp_path):
	baseline_rows, candidate_rows = _build_fixture_rows()
	baseline_dir = tmp_path / "runs" / "exp02_example" / "artifacts" / "error_analysis"
	candidate_dir = tmp_path / "runs" / "exp17_example" / "artifacts" / "error_analysis"
	_write_predictions_csv(baseline_dir / "predictions_test.csv", baseline_rows)
	_write_predictions_csv(candidate_dir / "predictions_test.csv", candidate_rows)
	_write_sidecars(baseline_dir, baseline_rows)
	_write_sidecars(candidate_dir, candidate_rows)

	out_dir = tmp_path / "artifacts" / "error_delta" / "exp17_vs_exp02"
	assets_dir = tmp_path / "docs" / "experiments" / "assets"
	main(
		[
			"--baseline-dir",
			str(baseline_dir),
			"--candidate-dir",
			str(candidate_dir),
			"--out-dir",
			str(out_dir),
			"--public-assets-dir",
			str(assets_dir),
			"--public-prefix",
			"exp17_vs_exp02",
		]
	)

	assert (out_dir / "sample_transitions.csv").exists()
	assert (out_dir / "per_class_delta.csv").exists()
	assert (out_dir / "confusion_delta.csv").exists()
	assert (out_dir / "summary.json").exists()
	assert (assets_dir / "exp17_vs_exp02_transition_counts.png").exists()
	assert (assets_dir / "exp17_vs_exp02_class_acc1_delta.png").exists()
	assert (assets_dir / "exp17_vs_exp02_confusion_delta.png").exists()

	with (out_dir / "summary.json").open(encoding="utf-8") as f:
		summary = json.load(f)
	assert summary["transition_counts"] == {"C->C": 1, "C->W": 1, "W->C": 2, "W->W": 1}
	assert summary["metrics"]["exp02"]["top1_errors"] == 3
	assert summary["metrics"]["exp17"]["top1_errors"] == 2


def _load_predictions_from_rows(rows: list[dict[str, str]]):
	parsed = []
	for row in rows:
		parsed.append(
			{
				"row_index": int(row["row_index"]),
				"image_path": row["image_path"],
				"true_idx": int(row["true_idx"]),
				"true_label": row["true_label"],
				"pred_idx": int(row["pred_idx"]),
				"pred_label": row["pred_label"],
				"is_correct": int(row["is_correct"]),
				"pred_confidence": float(row["pred_confidence"]),
				"true_class_confidence": float(row["true_class_confidence"]),
				"true_rank": int(row["true_rank"]),
				"is_top5_correct": int(row["is_top5_correct"]),
			}
		)
	return parsed
