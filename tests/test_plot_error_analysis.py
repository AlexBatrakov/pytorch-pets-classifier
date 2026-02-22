from src.plot_error_analysis import (
	_infer_related_csv_paths,
	_infer_split_from_predictions_path,
	_select_error_rows_for_gallery,
)


def test_infer_split_from_predictions_path():
	assert _infer_split_from_predictions_path("/tmp/predictions_test.csv") == "test"
	assert _infer_split_from_predictions_path("/tmp/other.csv") == "unknown"


def test_infer_related_csv_paths_from_predictions():
	per_class, top_conf = _infer_related_csv_paths(
		"/tmp/error_analysis/predictions_val.csv",
		None,
		None,
	)
	assert per_class.endswith("/tmp/error_analysis/per_class_metrics_val.csv")
	assert top_conf.endswith("/tmp/error_analysis/top_confusions_val.csv")


def test_select_error_rows_for_gallery_picks_most_confident_errors():
	rows = [
		{"is_correct": "1", "pred_confidence": "0.99", "id": "a"},
		{"is_correct": "0", "pred_confidence": "0.60", "id": "b"},
		{"is_correct": "0", "pred_confidence": "0.95", "id": "c"},
		{"is_correct": "0", "pred_confidence": "0.80", "id": "d"},
	]
	selected = _select_error_rows_for_gallery(rows, max_items=2)
	assert [r["id"] for r in selected] == ["c", "d"]
