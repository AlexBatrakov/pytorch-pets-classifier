from __future__ import annotations

import csv

from src.plot_metrics import _load_metrics, _plot_metrics


def test_load_metrics_and_plot_output(tmp_path) -> None:
	metrics_path = tmp_path / "metrics.csv"
	with metrics_path.open("w", newline="", encoding="utf-8") as f:
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
		writer.writerow([1, 1.6, 0.56, 0.84, 0.93, 0.72, 0.96, 0.0003])
		writer.writerow([2, 0.89, 0.75, 0.94, 0.52, 0.83, 0.99, 0.0003])

	metrics = _load_metrics(str(metrics_path))
	assert len(metrics["epoch"]) == 2
	assert metrics["val_acc1"][1] == 0.83

	out_path = tmp_path / "training_curves.png"
	_plot_metrics(metrics, out_path=str(out_path), title="Test Curves")
	assert out_path.exists()
	assert out_path.stat().st_size > 0
