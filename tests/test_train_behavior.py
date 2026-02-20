from __future__ import annotations

import csv

import torch.nn as nn

from src.train import (
	_append_metrics_csv,
	_build_checkpoint_payload,
	_count_parameters,
	_init_metrics_csv,
	_update_best_val_acc,
)


def test_update_best_val_acc_only_on_strict_improvement() -> None:
	best, is_best = _update_best_val_acc(0.80, 0.82)
	assert is_best is True
	assert best == 0.82

	best, is_best = _update_best_val_acc(best, 0.82)
	assert is_best is False
	assert best == 0.82

	best, is_best = _update_best_val_acc(best, 0.81)
	assert is_best is False
	assert best == 0.82


def test_build_checkpoint_payload_contains_required_keys() -> None:
	model = nn.Linear(4, 2)
	epoch_metrics = {
		"train_loss": 0.5,
		"train_acc1": 0.8,
		"train_acc5": 1.0,
		"val_loss": 0.6,
		"val_acc1": 0.75,
		"val_acc5": 0.95,
		"lr": 3e-4,
	}
	run_metadata = {"git_commit": "abc123", "device": "cpu"}
	payload = _build_checkpoint_payload(
		model=model,
		class_names=["cat", "dog"],
		cfg={"seed": 42},
		epoch=3,
		best_val_acc=0.75,
		epoch_metrics=epoch_metrics,
		run_metadata=run_metadata,
	)

	assert "model_state_dict" in payload
	assert "class_names" in payload
	assert "config" in payload
	assert "epoch" in payload
	assert "best_val_acc" in payload
	assert "epoch_metrics" in payload
	assert "run_metadata" in payload
	assert "saved_at_utc" in payload
	assert payload["epoch"] == 3
	assert payload["best_val_acc"] == 0.75
	assert payload["run_metadata"]["git_commit"] == "abc123"


def test_count_parameters_reports_total_and_trainable() -> None:
	model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
	for p in model[0].parameters():
		p.requires_grad = False

	total, trainable = _count_parameters(model)
	assert total > 0
	assert trainable > 0
	assert trainable < total


def test_metrics_csv_init_and_append(tmp_path) -> None:
	metrics_path = tmp_path / "metrics.csv"
	_init_metrics_csv(str(metrics_path))
	_append_metrics_csv(
		path=str(metrics_path),
		epoch=1,
		train_loss=1.0,
		train_acc1=0.5,
		train_acc5=0.8,
		val_loss=0.9,
		val_acc1=0.6,
		val_acc5=0.85,
		lr=3e-4,
	)

	with metrics_path.open("r", encoding="utf-8") as f:
		rows = list(csv.reader(f))

	assert rows[0] == [
		"epoch",
		"train_loss",
		"train_acc1",
		"train_acc5",
		"val_loss",
		"val_acc1",
		"val_acc5",
		"lr",
	]
	assert len(rows) == 2
	assert rows[1][0] == "1"
	assert rows[1][-1] == "0.00030000"
