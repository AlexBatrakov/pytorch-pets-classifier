from __future__ import annotations

import csv
import json
from pathlib import Path

from src.experiment_analytics import (
	build_seed_family_rows,
	collect_run_rows,
	_render_report,
	_build_run_row,
	_write_csv,
)


def _write_config(path: Path, payload: dict) -> None:
	import yaml

	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def _make_run(tmp_path: Path, name: str, config: dict, metrics: list[dict[str, object]], val: dict | None = None, test: dict | None = None, mlflow: dict | None = None) -> Path:
	run_dir = tmp_path / "runs" / name
	cfg_path = tmp_path / "configs" / "experiments" / f"{name}.yaml"
	config = {
		**config,
		"paths": {
			"checkpoints_dir": str(run_dir / "checkpoints"),
			"artifacts_dir": str(run_dir / "artifacts"),
		},
	}
	_write_config(cfg_path, config)
	_write_metrics_csv(run_dir / "artifacts" / "metrics.csv", metrics)
	if val is not None:
		(run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
		(run_dir / "artifacts" / "val_metrics.json").write_text(json.dumps(val), encoding="utf-8")
	if test is not None:
		(run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
		(run_dir / "artifacts" / "test_metrics.json").write_text(json.dumps(test), encoding="utf-8")
	if mlflow is not None:
		(run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
		(run_dir / "artifacts" / "mlflow_run.json").write_text(json.dumps(mlflow), encoding="utf-8")
	return cfg_path


def _default_config(seed: int) -> dict:
	return {
		"seed": seed,
		"deterministic": False,
		"data": {"batch_size": 32, "image_size": 224, "eval_resize_size": 256},
		"train": {
			"epochs": 3,
			"lr": 0.0003,
			"weight_decay": 0.01,
			"freeze_backbone": False,
			"freeze_epochs": 0,
			"scheduler": {"name": "cosine"},
			"early_stopping": {"enabled": True},
		},
	}


def _default_metrics() -> list[dict[str, object]]:
	return [
		{"epoch": 1, "train_loss": 1.0, "train_acc1": 0.5, "train_acc5": 0.8, "val_loss": 0.6, "val_acc1": 0.7, "val_acc5": 0.9, "lr": 0.0003},
		{"epoch": 2, "train_loss": 0.8, "train_acc1": 0.6, "train_acc5": 0.85, "val_loss": 0.5, "val_acc1": 0.72, "val_acc5": 0.91, "lr": 0.0003},
	]


def test_collect_run_rows_and_missing_test_status_match_repo_history(tmp_path):
	_make_run(
		tmp_path,
		"exp02_cosine_es_e30_s42",
		_default_config(seed=42),
		_default_metrics(),
		val={"split": "val", "loss": 0.5, "acc1": 0.72, "acc5": 0.91},
		test={"split": "test", "loss": 0.48, "acc1": 0.71, "acc5": 0.92},
	)
	_make_run(
		tmp_path,
		"exp17_cosine_es_img256_wd1e3_s42",
		_default_config(seed=42),
		_default_metrics(),
		val={"split": "val", "loss": 0.46, "acc1": 0.75, "acc5": 0.93},
		test={"split": "test", "loss": 0.44, "acc1": 0.74, "acc5": 0.94},
	)
	_make_run(
		tmp_path,
		"exp18_cosine_es_img256_wd1e3_cj_s42",
		_default_config(seed=42),
		_default_metrics(),
		val={"split": "val", "loss": 0.45, "acc1": 0.73, "acc5": 0.92},
		test={"split": "test", "loss": 0.46, "acc1": 0.72, "acc5": 0.93},
	)
	_make_run(tmp_path, "exp03_plateau_es_e30_s42", _default_config(seed=42), _default_metrics())
	_make_run(tmp_path, "exp03b_plateau_noes_e30_s42", _default_config(seed=42), _default_metrics())
	_make_run(tmp_path, "exp04_step_es_e30_s42", _default_config(seed=42), _default_metrics())

	rows = collect_run_rows(tmp_path)
	rows_by_name = {row["run_name"]: row for row in rows}
	assert "exp02_cosine_es_e30_s42" in rows_by_name
	assert "exp17_cosine_es_img256_wd1e3_s42" in rows_by_name
	assert "exp18_cosine_es_img256_wd1e3_cj_s42" in rows_by_name

	missing = {row["run_name"]: row for row in rows if row["test_metrics_available"] != "true"}
	assert set(missing) == {
		"exp03_plateau_es_e30_s42",
		"exp03b_plateau_noes_e30_s42",
		"exp04_step_es_e30_s42",
	}
	assert missing["exp03_plateau_es_e30_s42"]["test_metrics_source"] == "missing"
	assert missing["exp03b_plateau_noes_e30_s42"]["test_metrics_source"] == "missing"
	assert missing["exp04_step_es_e30_s42"]["test_metrics_source"] == "missing"


def test_build_run_row_uses_csv_fallback_for_missing_val_and_preserves_missing_test(tmp_path, monkeypatch):
	cfg_path = _make_run(
		tmp_path,
		"exp99_demo_s42",
		{
			"seed": 42,
			"deterministic": False,
			"data": {"batch_size": 32},
			"train": {
				"epochs": 3,
				"lr": 0.0003,
				"weight_decay": 0.01,
				"freeze_backbone": False,
				"freeze_epochs": 0,
				"scheduler": {"name": "cosine"},
				"early_stopping": {"enabled": True},
			},
		},
		[
			{"epoch": 1, "train_loss": 1.0, "train_acc1": 0.5, "train_acc5": 0.8, "val_loss": 0.6, "val_acc1": 0.7, "val_acc5": 0.9, "lr": 0.0003},
			{"epoch": 2, "train_loss": 0.8, "train_acc1": 0.6, "train_acc5": 0.85, "val_loss": 0.5, "val_acc1": 0.72, "val_acc5": 0.91, "lr": 0.0003},
		],
		val=None,
		test=None,
		mlflow={"run_id": "abc", "experiment_name": "demo"},
	)

	monkeypatch.setattr("src.experiment_analytics.REPO_ROOT", tmp_path)
	row = _build_run_row(cfg_path)

	assert row["val_metrics_available"] == "false"
	assert row["val_metrics_source"] == "metrics.csv (best val row)"
	assert row["val_acc1"] == "0.720000"
	assert row["test_metrics_available"] == "false"
	assert row["test_metrics_source"] == "missing"
	assert row["mlflow_run_id"] == "abc"


def test_collect_run_rows_respects_custom_root(tmp_path):
	_make_run(
		tmp_path,
		"exp98_root_demo_s123",
		{
			"seed": 123,
			"deterministic": False,
			"data": {"batch_size": 16, "image_size": 256, "eval_resize_size": 292},
			"train": {
				"epochs": 2,
				"lr": 0.00015,
				"weight_decay": 0.001,
				"freeze_backbone": False,
				"freeze_epochs": 0,
				"scheduler": {"name": "cosine"},
				"early_stopping": {"enabled": True},
			},
		},
		[
			{"epoch": 1, "train_loss": 1.0, "train_acc1": 0.5, "train_acc5": 0.8, "val_loss": 0.6, "val_acc1": 0.7, "val_acc5": 0.9, "lr": 0.00015},
			{"epoch": 2, "train_loss": 0.8, "train_acc1": 0.6, "train_acc5": 0.85, "val_loss": 0.5, "val_acc1": 0.72, "val_acc5": 0.91, "lr": 0.00015},
		],
		val={"split": "val", "loss": 0.5, "acc1": 0.72, "acc5": 0.91},
		test={"split": "test", "loss": 0.48, "acc1": 0.71, "acc5": 0.92},
	)

	rows = collect_run_rows(tmp_path)

	assert len(rows) == 1
	assert rows[0]["config_path"] == "configs/experiments/exp98_root_demo_s123.yaml"
	assert rows[0]["run_dir"] == "runs/exp98_root_demo_s123"
	assert rows[0]["image_size"] == 256
	assert rows[0]["test_acc1"] == "0.710000"


def test_family_summary_and_report_include_expected_grouping(tmp_path):
	_make_run(
		tmp_path,
		"exp02_cosine_es_e30_s42",
		_default_config(seed=42),
		_default_metrics(),
		val={"split": "val", "loss": 0.5, "acc1": 0.72, "acc5": 0.91},
		test={"split": "test", "loss": 0.48, "acc1": 0.71, "acc5": 0.92},
		mlflow={"run_id": "exp02-run", "experiment_name": "pets"},
	)
	_make_run(
		tmp_path,
		"exp05_cosine_es_e30_s123",
		_default_config(seed=123),
		_default_metrics(),
		val={"split": "val", "loss": 0.49, "acc1": 0.73, "acc5": 0.91},
		test={"split": "test", "loss": 0.47, "acc1": 0.72, "acc5": 0.92},
	)
	_make_run(
		tmp_path,
		"exp06_cosine_es_e30_s777",
		_default_config(seed=777),
		_default_metrics(),
		val={"split": "val", "loss": 0.48, "acc1": 0.74, "acc5": 0.92},
		test={"split": "test", "loss": 0.46, "acc1": 0.73, "acc5": 0.93},
	)
	_make_run(
		tmp_path,
		"exp07_cosine_es_bs16_lr15e4_s42",
		{
			**_default_config(seed=42),
			"data": {"batch_size": 16, "image_size": 224, "eval_resize_size": 256},
			"train": {**_default_config(seed=42)["train"], "lr": 0.00015},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.47, "acc1": 0.75, "acc5": 0.92},
		test={"split": "test", "loss": 0.45, "acc1": 0.74, "acc5": 0.93},
	)
	_make_run(
		tmp_path,
		"exp10_cosine_es_bs16_lr15e4_s123",
		{
			**_default_config(seed=123),
			"data": {"batch_size": 16, "image_size": 224, "eval_resize_size": 256},
			"train": {**_default_config(seed=123)["train"], "lr": 0.00015},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.46, "acc1": 0.76, "acc5": 0.93},
		test={"split": "test", "loss": 0.44, "acc1": 0.75, "acc5": 0.94},
	)
	_make_run(
		tmp_path,
		"exp11_cosine_es_bs16_lr15e4_s777",
		{
			**_default_config(seed=777),
			"data": {"batch_size": 16, "image_size": 224, "eval_resize_size": 256},
			"train": {**_default_config(seed=777)["train"], "lr": 0.00015},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.45, "acc1": 0.77, "acc5": 0.94},
		test={"split": "test", "loss": 0.43, "acc1": 0.76, "acc5": 0.95},
	)
	_make_run(
		tmp_path,
		"exp12_cosine_es_img224_s42",
		_default_config(seed=42),
		_default_metrics(),
		val={"split": "val", "loss": 0.47, "acc1": 0.74, "acc5": 0.92},
		test={"split": "test", "loss": 0.45, "acc1": 0.73, "acc5": 0.93},
	)
	_make_run(
		tmp_path,
		"exp13_cosine_es_img256_s42",
		{
			**_default_config(seed=42),
			"data": {"batch_size": 32, "image_size": 256, "eval_resize_size": 292},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.46, "acc1": 0.75, "acc5": 0.93},
		test={"split": "test", "loss": 0.44, "acc1": 0.74, "acc5": 0.94},
	)
	_make_run(
		tmp_path,
		"exp14_cosine_es_img320_s42",
		{
			**_default_config(seed=42),
			"data": {"batch_size": 16, "image_size": 320, "eval_resize_size": 352},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.48, "acc1": 0.73, "acc5": 0.92},
		test={"split": "test", "loss": 0.46, "acc1": 0.72, "acc5": 0.93},
	)
	_make_run(
		tmp_path,
		"exp17_cosine_es_img256_wd1e3_s42",
		{
			**_default_config(seed=42),
			"data": {"batch_size": 32, "image_size": 256, "eval_resize_size": 292},
			"train": {**_default_config(seed=42)["train"], "weight_decay": 0.001},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.44, "acc1": 0.78, "acc5": 0.94},
		test={"split": "test", "loss": 0.42, "acc1": 0.77, "acc5": 0.95},
		mlflow={"run_id": "exp17-run", "experiment_name": "pets"},
	)
	_make_run(
		tmp_path,
		"exp18_cosine_es_img256_wd1e3_cj_s42",
		{
			**_default_config(seed=42),
			"data": {"batch_size": 32, "image_size": 256, "eval_resize_size": 292, "aug": {"color_jitter": {"enabled": True}}},
			"train": {**_default_config(seed=42)["train"], "weight_decay": 0.001},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.45, "acc1": 0.76, "acc5": 0.93},
		test={"split": "test", "loss": 0.43, "acc1": 0.75, "acc5": 0.94},
	)
	_make_run(
		tmp_path,
		"exp19_cosine_es_img256_wd1e3_s123",
		{
			**_default_config(seed=123),
			"data": {"batch_size": 32, "image_size": 256, "eval_resize_size": 292},
			"train": {**_default_config(seed=123)["train"], "weight_decay": 0.001},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.43, "acc1": 0.79, "acc5": 0.95},
		test={"split": "test", "loss": 0.41, "acc1": 0.78, "acc5": 0.96},
	)
	_make_run(
		tmp_path,
		"exp20_cosine_es_img256_wd1e3_s777",
		{
			**_default_config(seed=777),
			"data": {"batch_size": 32, "image_size": 256, "eval_resize_size": 292},
			"train": {**_default_config(seed=777)["train"], "weight_decay": 0.001},
		},
		_default_metrics(),
		val={"split": "val", "loss": 0.42, "acc1": 0.80, "acc5": 0.95},
		test={"split": "test", "loss": 0.40, "acc1": 0.79, "acc5": 0.96},
	)
	_make_run(tmp_path, "exp03_plateau_es_e30_s42", _default_config(seed=42), _default_metrics())
	_make_run(tmp_path, "exp03b_plateau_noes_e30_s42", _default_config(seed=42), _default_metrics())
	_make_run(tmp_path, "exp04_step_es_e30_s42", _default_config(seed=42), _default_metrics())

	run_rows = collect_run_rows(tmp_path)
	family_rows = build_seed_family_rows(run_rows)

	assert {row["seed_family"] for row in family_rows} == {"exp02", "exp07", "exp17"}
	assert len(family_rows) == 3
	assert float(next(row for row in family_rows if row["seed_family"] == "exp17")["test_acc1_mean"]) > float(
		next(row for row in family_rows if row["seed_family"] == "exp02")["test_acc1_mean"]
	)
	report = _render_report(run_rows, family_rows)
	eval_json_count = sum(1 for row in run_rows if row["val_metrics_available"] == "true" and row["test_metrics_available"] == "true")
	mlflow_json_count = sum(1 for row in run_rows if row["mlflow_json_available"] == "true")
	assert "Cross-Run Experiment Analytics" in report
	assert "exp17" in report
	assert "ColorJitter" in report
	assert f"Total committed runs: `{len(run_rows)}`" in report
	assert f"Runs with retained `val_metrics.json` and `test_metrics.json`: `{eval_json_count}`" in report
	assert f"Runs with `mlflow_run.json`: `{mlflow_json_count}`" in report


def test_write_csv_round_trips_rows(tmp_path):
	out = tmp_path / "summary.csv"
	rows = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
	_write_csv(out, ["a", "b"], rows)
	with out.open(newline="", encoding="utf-8") as f:
		loaded = list(csv.DictReader(f))
	assert loaded == rows
