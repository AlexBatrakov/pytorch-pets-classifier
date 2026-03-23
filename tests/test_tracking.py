from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from src.log_run_artifacts import _build_run_paths, _load_final_metrics
from src.tracking import (
	DEFAULT_EXPERIMENT_NAME,
	DEFAULT_TRACKING_URI,
	TrackingSettings,
	build_path_tags,
	create_tracker,
	flatten_mapping,
	infer_run_name,
	load_run_info,
	normalize_paths,
	resolve_tracking_settings,
	write_run_info,
)


def test_resolve_tracking_settings_defaults_off() -> None:
	settings = resolve_tracking_settings({})
	assert settings == TrackingSettings(enabled=False)


def test_resolve_tracking_settings_defaults_to_local_sqlite_store() -> None:
	settings = resolve_tracking_settings({"MLFLOW_TRACKING": "1"})
	assert settings.enabled is True
	assert settings.tracking_uri == DEFAULT_TRACKING_URI
	assert settings.experiment_name == DEFAULT_EXPERIMENT_NAME


def test_flatten_mapping_handles_nested_scalars_and_sequences() -> None:
	payload = {
		"seed": 42,
		"deterministic": False,
		"train": {"epochs": 5, "scheduler": {"name": "cosine"}},
		"labels": ["cat", "dog"],
	}

	flat = flatten_mapping(payload)

	assert flat["seed"] == "42"
	assert flat["deterministic"] == "false"
	assert flat["train.epochs"] == "5"
	assert flat["train.scheduler.name"] == "cosine"
	assert flat["labels"] == '["cat", "dog"]'


def test_normalize_paths_and_build_path_tags_are_repo_relative(tmp_path) -> None:
	run_dir = tmp_path / "runs" / "exp17"
	paths = {
		"metrics_csv": str(run_dir / "artifacts" / "metrics.csv"),
		"best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
	}

	normalized = normalize_paths(paths, repo_root=str(tmp_path))
	tags = build_path_tags(paths, repo_root=str(tmp_path))

	assert normalized["metrics_csv"] == "runs/exp17/artifacts/metrics.csv"
	assert tags["path.best_checkpoint"] == "runs/exp17/checkpoints/best.pt"


def test_normalize_paths_uses_absolute_path_outside_repo_root(tmp_path) -> None:
	repo_root = tmp_path / "repo"
	external_path = tmp_path / "external" / "metrics.csv"
	normalized = normalize_paths({"metrics_csv": str(external_path)}, repo_root=str(repo_root))
	assert normalized["metrics_csv"] == external_path.resolve().as_posix()


def test_run_info_round_trip(tmp_path) -> None:
	run_info_path = tmp_path / "mlflow_run.json"
	payload = {
		"run_id": "run-123",
		"tracking_uri": DEFAULT_TRACKING_URI,
		"experiment_name": DEFAULT_EXPERIMENT_NAME,
		"paths": {"metrics_csv": "runs/exp17/artifacts/metrics.csv"},
	}

	write_run_info(str(run_info_path), payload)
	loaded = load_run_info(str(run_info_path))

	assert loaded == payload


def test_infer_run_name_falls_back_to_config_stem_for_default_checkpoints(tmp_path, monkeypatch) -> None:
	monkeypatch.chdir(tmp_path)
	run_name = infer_run_name("configs/default.yaml", "./checkpoints")
	assert run_name == "default"


def test_create_tracker_uses_mlflow_module_when_enabled(monkeypatch, tmp_path) -> None:
	calls: list[tuple[object, ...]] = []

	class _FakeRun:
		def __init__(self, run_id: str) -> None:
			self.info = SimpleNamespace(run_id=run_id)

	class _FakeMlflow:
		def set_tracking_uri(self, uri: str) -> None:
			calls.append(("set_tracking_uri", uri))

		def set_experiment(self, name: str) -> None:
			calls.append(("set_experiment", name))

		def start_run(self, run_name: str | None = None, run_id: str | None = None) -> _FakeRun:
			calls.append(("start_run", run_name, run_id))
			return _FakeRun(run_id or "run-123")

		def log_params(self, params: dict[str, str]) -> None:
			calls.append(("log_params", params))

		def set_tags(self, tags: dict[str, str]) -> None:
			calls.append(("set_tags", tags))

		def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
			calls.append(("log_metrics", metrics, step))

		def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
			calls.append(("log_artifact", Path(path).name, artifact_path))

		def end_run(self, status: str | None = None) -> None:
			calls.append(("end_run", status))

	monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow())
	artifact_path = tmp_path / "metrics.csv"
	artifact_path.write_text("epoch,train_loss\n1,1.0\n", encoding="utf-8")

	tracker = create_tracker(
		TrackingSettings(
			enabled=True,
			tracking_uri=DEFAULT_TRACKING_URI,
			experiment_name=DEFAULT_EXPERIMENT_NAME,
		)
	)
	run_id = tracker.start_run(run_name="exp17")
	tracker.log_params({"seed": 42})
	tracker.set_tags({"path.metrics_csv": "runs/exp17/artifacts/metrics.csv"})
	tracker.log_metrics({"val_acc1": 0.9}, step=1)
	tracker.log_artifact(str(artifact_path), artifact_path="artifacts")
	tracker.end_run(status="FINISHED")

	assert run_id == "run-123"
	assert ("set_experiment", DEFAULT_EXPERIMENT_NAME) in calls
	assert ("log_metrics", {"val_acc1": 0.9}, 1) in calls
	assert ("log_artifact", "metrics.csv", "artifacts") in calls
	assert ("end_run", "FINISHED") in calls


def test_load_final_metrics_uses_best_val_and_test_prefixes(tmp_path) -> None:
	val_json = tmp_path / "val_metrics.json"
	test_json = tmp_path / "test_metrics.json"
	val_json.write_text('{"split":"val","loss":0.2,"acc1":0.9,"acc5":0.99}', encoding="utf-8")
	test_json.write_text('{"split":"test","loss":0.3,"acc1":0.88,"acc5":0.98}', encoding="utf-8")

	val_metrics = _load_final_metrics(str(val_json), "val")
	test_metrics = _load_final_metrics(str(test_json), "test")

	assert val_metrics["best_val_acc1"] == 0.9
	assert test_metrics["test_acc5"] == 0.98


def test_build_run_paths_matches_existing_runner_layout(tmp_path) -> None:
	run_dir = tmp_path / "runs" / "exp17"
	paths = _build_run_paths(str(run_dir))
	assert paths["metrics_csv"].endswith("runs/exp17/artifacts/metrics.csv")
	assert paths["training_curves_png"].endswith("runs/exp17/assets/training_curves.png")
