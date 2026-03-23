from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "pytorch-pets-classifier"
RUN_INFO_FILENAME = "mlflow_run.json"


def _is_truthy(value: str | None) -> bool:
	if value is None:
		return False
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _stringify_value(value: Any) -> str:
	if value is None:
		return "null"
	if isinstance(value, bool):
		return "true" if value else "false"
	if isinstance(value, (list, tuple)):
		return json.dumps(list(value), sort_keys=True)
	if isinstance(value, dict):
		return json.dumps(value, sort_keys=True)
	return str(value)


@dataclass(frozen=True)
class TrackingSettings:
	enabled: bool
	tracking_uri: str | None = None
	experiment_name: str | None = None


def resolve_tracking_settings(env: Mapping[str, str] | None = None) -> TrackingSettings:
	env_map = env if env is not None else os.environ
	enabled = _is_truthy(env_map.get("MLFLOW_TRACKING"))
	if not enabled:
		return TrackingSettings(enabled=False)
	tracking_uri = env_map.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
	experiment_name = env_map.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME
	return TrackingSettings(
		enabled=True,
		tracking_uri=tracking_uri,
		experiment_name=experiment_name,
	)


def flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
	flat: dict[str, str] = {}
	for key, value in mapping.items():
		full_key = f"{prefix}.{key}" if prefix else str(key)
		if isinstance(value, Mapping):
			flat.update(flatten_mapping(value, prefix=full_key))
		else:
			flat[full_key] = _stringify_value(value)
	return flat


def to_repo_relative_path(path: str, repo_root: str | None = None) -> str:
	root = Path(repo_root or os.getcwd()).resolve()
	absolute_path = Path(path).resolve(strict=False)
	try:
		return absolute_path.relative_to(root).as_posix()
	except ValueError:
		return absolute_path.as_posix()


def normalize_paths(paths: Mapping[str, str], repo_root: str | None = None) -> dict[str, str]:
	return {key: to_repo_relative_path(value, repo_root=repo_root) for key, value in paths.items()}


def build_path_tags(paths: Mapping[str, str], repo_root: str | None = None) -> dict[str, str]:
	normalized = normalize_paths(paths, repo_root=repo_root)
	return {f"path.{key}": value for key, value in normalized.items()}


def infer_run_name(config_path: str, checkpoints_dir: str, cwd: str | None = None) -> str:
	current_dir_name = Path(cwd or os.getcwd()).resolve().name
	checkpoints_path = Path(checkpoints_dir)
	if checkpoints_path.name == "checkpoints":
		candidate = checkpoints_path.parent.name
		if candidate and candidate not in {".", current_dir_name}:
			return candidate
		return Path(config_path).stem or "train"
	if checkpoints_path.name and checkpoints_path.name not in {".", ""}:
		return checkpoints_path.name
	return Path(config_path).stem or "train"


def write_run_info(path: str, payload: Mapping[str, Any]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(dict(payload), f, indent=2, sort_keys=True)


def load_run_info(path: str) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


class NoOpTracker:
	def __init__(self, settings: TrackingSettings) -> None:
		self.settings = settings
		self.run_id: str | None = None
		self.has_active_run = False

	def start_run(self, run_name: str | None = None, run_id: str | None = None) -> str | None:
		_ = run_name, run_id
		return None

	def log_params(self, params: Mapping[str, Any]) -> None:
		_ = params

	def set_tags(self, tags: Mapping[str, Any]) -> None:
		_ = tags

	def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
		_ = metrics, step

	def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
		_ = path, artifact_path

	def end_run(self, status: str | None = None) -> None:
		_ = status


class MlflowTracker:
	def __init__(self, settings: TrackingSettings) -> None:
		try:
			import mlflow  # type: ignore
		except ImportError as exc:
			raise RuntimeError(
				"MLflow tracking is enabled but 'mlflow' is not installed. "
				"Install it via 'pip install -r requirements-dev.txt'."
			) from exc

		self.settings = settings
		self._mlflow = mlflow
		self.run_id: str | None = None
		self.has_active_run = False

	def start_run(self, run_name: str | None = None, run_id: str | None = None) -> str:
		if not self.settings.tracking_uri:
			raise RuntimeError("MLflow tracking URI is required when tracking is enabled.")
		self._mlflow.set_tracking_uri(self.settings.tracking_uri)
		if run_id is None:
			if not self.settings.experiment_name:
				raise RuntimeError("MLflow experiment name is required when tracking is enabled.")
			self._mlflow.set_experiment(self.settings.experiment_name)
			run = self._mlflow.start_run(run_name=run_name)
		else:
			run = self._mlflow.start_run(run_id=run_id)
		self.run_id = str(run.info.run_id)
		self.has_active_run = True
		return self.run_id

	def log_params(self, params: Mapping[str, Any]) -> None:
		if not params:
			return
		self._mlflow.log_params({key: _stringify_value(value) for key, value in params.items()})

	def set_tags(self, tags: Mapping[str, Any]) -> None:
		if not tags:
			return
		self._mlflow.set_tags({key: _stringify_value(value) for key, value in tags.items()})

	def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
		if not metrics:
			return
		payload = {key: float(value) for key, value in metrics.items()}
		self._mlflow.log_metrics(payload, step=step)

	def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
		self._mlflow.log_artifact(path, artifact_path=artifact_path)

	def end_run(self, status: str | None = None) -> None:
		if not self.has_active_run:
			return
		self._mlflow.end_run(status=status)
		self.has_active_run = False


def create_tracker(settings: TrackingSettings | None = None) -> NoOpTracker | MlflowTracker:
	tracker_settings = settings if settings is not None else resolve_tracking_settings()
	if not tracker_settings.enabled:
		return NoOpTracker(tracker_settings)
	return MlflowTracker(tracker_settings)
