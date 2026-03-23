from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tracking import (
	RUN_INFO_FILENAME,
	TrackingSettings,
	build_path_tags,
	create_tracker,
	load_run_info,
)


def _build_run_paths(run_dir: str) -> dict[str, str]:
	base = Path(run_dir)
	return {
		"metrics_csv": str(base / "artifacts" / "metrics.csv"),
		"val_metrics_json": str(base / "artifacts" / "val_metrics.json"),
		"test_metrics_json": str(base / "artifacts" / "test_metrics.json"),
		"training_curves_png": str(base / "assets" / "training_curves.png"),
		"confusion_matrix_png": str(base / "assets" / "confusion_matrix.png"),
		"best_checkpoint": str(base / "checkpoints" / "best.pt"),
		"last_checkpoint": str(base / "checkpoints" / "last.pt"),
	}


def _load_final_metrics(path: str, split: str) -> dict[str, float]:
	with open(path, "r", encoding="utf-8") as f:
		payload = json.load(f)

	if payload.get("split") != split:
		raise ValueError(f"Expected split '{split}' in {path}, found '{payload.get('split')}'.")

	prefix = "best_val" if split == "val" else "test"
	return {
		f"{prefix}_loss": float(payload["loss"]),
		f"{prefix}_acc1": float(payload["acc1"]),
		f"{prefix}_acc5": float(payload["acc5"]),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Log post-train MLflow artifacts for an experiment run")
	parser.add_argument("--run-dir", required=True, help="Run directory produced by scripts/run_experiment.sh")
	args = parser.parse_args()

	run_dir = Path(args.run_dir)
	run_info_path = run_dir / "artifacts" / RUN_INFO_FILENAME
	if not run_info_path.exists():
		print("MLflow tracking not enabled for this run; skipping artifact logging.")
		return

	run_info = load_run_info(str(run_info_path))
	run_id = str(run_info["run_id"])
	tracking_uri = str(run_info["tracking_uri"])
	experiment_name = str(run_info.get("experiment_name") or "")

	tracker = create_tracker(
		TrackingSettings(
			enabled=True,
			tracking_uri=tracking_uri,
			experiment_name=experiment_name,
		)
	)

	paths = _build_run_paths(str(run_dir))
	artifacts = [
		(paths["metrics_csv"], "artifacts", False),
		(paths["val_metrics_json"], "artifacts", False),
		(paths["test_metrics_json"], "artifacts", False),
		(paths["training_curves_png"], "assets", True),
		(paths["confusion_matrix_png"], "assets", True),
	]

	run_failed = False
	try:
		tracker.start_run(run_id=run_id)
		tracker.set_tags(build_path_tags(paths))
		tracker.log_metrics(_load_final_metrics(paths["val_metrics_json"], "val"))
		tracker.log_metrics(_load_final_metrics(paths["test_metrics_json"], "test"))

		for artifact_path, target_dir, optional in artifacts:
			file_path = Path(artifact_path)
			if file_path.exists():
				tracker.log_artifact(str(file_path), artifact_path=target_dir)
			elif not optional:
				raise FileNotFoundError(f"Required MLflow artifact not found: {file_path}")
			else:
				print(f"Optional artifact missing; skipping: {file_path}")
	except Exception:
		run_failed = True
		raise
	finally:
		tracker.end_run(status="FAILED" if run_failed else "FINISHED")

	print(f"Logged final MLflow metrics and artifacts for run: {run_id}")


if __name__ == "__main__":
	main()
