import csv
from collections import Counter
from pathlib import Path

from torch.utils.data import Subset

from src.error_analysis import (
	_build_per_class_rows,
	_build_top_confusion_rows,
	_resolve_loader_image_paths,
	_write_predictions_csv,
)


class _FakeDataset:
	def __init__(self):
		self._images = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg"), Path("/tmp/c.jpg")]

	def __len__(self):
		return len(self._images)


class _FakeLoader:
	def __init__(self, dataset):
		self.dataset = dataset


def test_resolve_loader_image_paths_for_dataset():
	loader = _FakeLoader(_FakeDataset())
	paths = _resolve_loader_image_paths(loader)
	assert paths == ["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"]


def test_resolve_loader_image_paths_for_subset():
	subset = Subset(_FakeDataset(), [2, 0])
	loader = _FakeLoader(subset)
	paths = _resolve_loader_image_paths(loader)
	assert paths == ["/tmp/c.jpg", "/tmp/a.jpg"]


def test_write_predictions_csv_writes_header_and_rows(tmp_path):
	out = tmp_path / "predictions.csv"
	rows = [
		{
			"row_index": 0,
			"image_path": "/tmp/a.jpg",
			"true_idx": 1,
			"true_label": "A",
			"pred_idx": 2,
			"pred_label": "B",
			"is_correct": 0,
			"pred_confidence": "0.900000",
			"true_class_confidence": "0.010000",
			"true_rank": 4,
			"is_top5_correct": 1,
			"top5_indices": "[2,1]",
			"top5_labels": "[\"B\",\"A\"]",
			"top5_confidences": "[0.9,0.01]",
		}
	]
	_write_predictions_csv(str(out), rows)

	with open(out, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		loaded = list(reader)

	assert len(loaded) == 1
	assert loaded[0]["image_path"] == "/tmp/a.jpg"
	assert loaded[0]["pred_label"] == "B"


def test_build_per_class_rows_sorts_hardest_first():
	rows = _build_per_class_rows(
		class_names=["A", "B", "C"],
		supports=[10, 5, 5],
		correct1_counts=[9, 2, 2],
		correct5_counts=[10, 5, 4],
	)
	assert [r["class_name"] for r in rows] == ["B", "C", "A"]
	assert rows[0]["acc1"] == "0.400000"
	assert rows[0]["error_count"] == 3


def test_build_top_confusion_rows_uses_counts_and_normalized_rate():
	rows = _build_top_confusion_rows(
		class_names=["A", "B", "C"],
		confusion_counts=Counter({(0, 1): 4, (2, 1): 3, (0, 2): 2}),
		supports=[10, 8, 4],
		limit=2,
	)
	assert len(rows) == 2
	assert rows[0]["true_label"] == "A"
	assert rows[0]["pred_label"] == "B"
	assert rows[0]["count"] == 4
	assert rows[0]["row_normalized_rate"] == "0.400000"
