from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "experiments"
RUNS_DIR = REPO_ROOT / "runs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "experiment_analytics"
DOCS_DIR = REPO_ROOT / "docs" / "experiments"
PUBLIC_ASSETS_DIR = DOCS_DIR / "assets"

SEED_FAMILIES: dict[str, dict[str, Any]] = {
	"exp02": {
		"recipe_label": "cosine + early stopping",
		"runs": [
			"exp02_cosine_es_e30_s42",
			"exp05_cosine_es_e30_s123",
			"exp06_cosine_es_e30_s777",
		],
	},
	"exp07": {
		"recipe_label": "cosine + early stopping, batch=16, lr=1.5e-4",
		"runs": [
			"exp07_cosine_es_bs16_lr15e4_s42",
			"exp10_cosine_es_bs16_lr15e4_s123",
			"exp11_cosine_es_bs16_lr15e4_s777",
		],
	},
	"exp17": {
		"recipe_label": "cosine + early stopping, img256, wd=1e-3",
		"runs": [
			"exp17_cosine_es_img256_wd1e3_s42",
			"exp19_cosine_es_img256_wd1e3_s123",
			"exp20_cosine_es_img256_wd1e3_s777",
		],
	},
}

RUN_GROUPS: dict[str, str] = {
	"exp01_baseline_e15_s42": "baseline",
	"exp02_cosine_es_e30_s42": "scheduler",
	"exp03_plateau_es_e30_s42": "scheduler",
	"exp03b_plateau_noes_e30_s42": "scheduler",
	"exp04_step_es_e30_s42": "scheduler",
	"exp05_cosine_es_e30_s123": "seed_support",
	"exp06_cosine_es_e30_s777": "seed_support",
	"exp07_cosine_es_bs16_lr15e4_s42": "batch_lr",
	"exp08_cosine_es_bs64_lr6e4_s42": "batch_lr",
	"exp09_cosine_es_freeze2_s42": "freeze",
	"exp10_cosine_es_bs16_lr15e4_s123": "seed_support",
	"exp11_cosine_es_bs16_lr15e4_s777": "seed_support",
	"exp12_cosine_es_img224_s42": "resolution",
	"exp13_cosine_es_img256_s42": "resolution",
	"exp14_cosine_es_img320_s42": "resolution",
	"exp15_cosine_es_img256_wd0_s42": "weight_decay",
	"exp16_cosine_es_img256_wd1e4_s42": "weight_decay",
	"exp17_cosine_es_img256_wd1e3_s42": "weight_decay",
	"exp18_cosine_es_img256_wd1e3_cj_s42": "augmentation",
	"exp19_cosine_es_img256_wd1e3_s123": "seed_support",
	"exp20_cosine_es_img256_wd1e3_s777": "seed_support",
}

CSV_FIELDNAMES = [
	"run_name",
	"experiment_id",
	"config_path",
	"run_dir",
	"analysis_group",
	"seed_family",
	"seed",
	"scheduler",
	"early_stopping",
	"batch_size",
	"lr",
	"weight_decay",
	"image_size",
	"eval_resize_size",
	"freeze_backbone",
	"freeze_epochs",
	"color_jitter",
	"epochs_planned",
	"epochs_logged",
	"best_epoch",
	"stopped_epoch",
	"val_metrics_available",
	"val_metrics_source",
	"test_metrics_available",
	"test_metrics_source",
	"metrics_csv_available",
	"mlflow_json_available",
	"mlflow_run_id",
	"mlflow_experiment_name",
	"val_loss",
	"val_acc1",
	"val_acc5",
	"test_loss",
	"test_acc1",
	"test_acc5",
]

FAMILY_FIELDNAMES = [
	"seed_family",
	"recipe_label",
	"run_count",
	"runs",
	"seeds",
	"test_acc1_mean",
	"test_acc1_std",
	"test_acc5_mean",
	"test_acc5_std",
	"test_loss_mean",
	"test_loss_std",
]


@dataclass(frozen=True)
class MetricsRecord:
	split: str
	loss: float
	acc1: float
	acc5: float
	source: str
	available: bool


def _fmt_float(value: float | None, digits: int = 6) -> str:
	if value is None:
		return ""
	return f"{value:.{digits}f}"


def _fmt_bool(value: bool | None) -> str:
	if value is None:
		return ""
	return "true" if value else "false"


def _load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def _read_metrics_csv(path: Path) -> list[dict[str, Any]]:
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)
	if not rows:
		raise ValueError(f"Metrics file is empty: {path}")
	return rows


def _load_val_metrics(run_dir: Path) -> MetricsRecord:
	val_path = run_dir / "artifacts" / "val_metrics.json"
	if val_path.exists():
		payload = _load_json(val_path)
		return MetricsRecord(
			split="val",
			loss=float(payload["loss"]),
			acc1=float(payload["acc1"]),
			acc5=float(payload["acc5"]),
			source="val_metrics.json",
			available=True,
		)

	metrics_path = run_dir / "artifacts" / "metrics.csv"
	rows = _read_metrics_csv(metrics_path)
	best_row = max(rows, key=lambda row: float(row["val_acc1"]))
	return MetricsRecord(
		split="val",
		loss=float(best_row["val_loss"]),
		acc1=float(best_row["val_acc1"]),
		acc5=float(best_row["val_acc5"]),
		source="metrics.csv (best val row)",
		available=False,
	)


def _load_test_metrics(run_dir: Path) -> MetricsRecord:
	test_path = run_dir / "artifacts" / "test_metrics.json"
	if test_path.exists():
		payload = _load_json(test_path)
		return MetricsRecord(
			split="test",
			loss=float(payload["loss"]),
			acc1=float(payload["acc1"]),
			acc5=float(payload["acc5"]),
			source="test_metrics.json",
			available=True,
		)

	return MetricsRecord(
		split="test",
		loss=float("nan"),
		acc1=float("nan"),
		acc5=float("nan"),
		source="missing",
		available=False,
	)


def _resolve_run_name(config_path: Path, cfg: dict[str, Any]) -> str:
	paths = cfg.get("paths", {}) or {}
	checkpoints_dir = Path(paths.get("checkpoints_dir", "checkpoints"))
	if checkpoints_dir.name == "checkpoints":
		candidate = checkpoints_dir.parent.name
		if candidate:
			return candidate
	return config_path.stem


def _resolve_run_dir(cfg: dict[str, Any], root: Path | None = None) -> Path:
	root = root or REPO_ROOT
	paths = cfg.get("paths", {}) or {}
	run_dir = Path(paths.get("artifacts_dir", "artifacts")).parent
	if not run_dir.is_absolute():
		run_dir = root / run_dir
	return run_dir.resolve(strict=False)


def _seed_family_for_run(run_name: str) -> str:
	for family, payload in SEED_FAMILIES.items():
		if run_name in payload["runs"]:
			return family
	return ""


def _analysis_group_for_run(run_name: str) -> str:
	return RUN_GROUPS.get(run_name, "unassigned")


def _extract_factor_value(cfg: dict[str, Any], *keys: str, default: Any = "") -> Any:
	node: Any = cfg
	for key in keys:
		if not isinstance(node, dict):
			return default
		node = node.get(key)
		if node is None:
			return default
	return node


def _build_run_row(config_path: Path, root: Path | None = None) -> dict[str, Any]:
	root = root or REPO_ROOT
	cfg = load_config(str(config_path))
	run_name = _resolve_run_name(config_path, cfg)
	run_dir = _resolve_run_dir(cfg, root=root)
	metrics_path = run_dir / "artifacts" / "metrics.csv"
	rows = _read_metrics_csv(metrics_path)
	best_row = max(rows, key=lambda row: float(row["val_acc1"]))
	val_metrics = _load_val_metrics(run_dir)
	test_metrics = _load_test_metrics(run_dir)
	mlflow_path = run_dir / "artifacts" / "mlflow_run.json"
	mlflow_payload = _load_json(mlflow_path) if mlflow_path.exists() else {}

	run_id = run_name.split("_", 1)[0]

	return {
		"run_name": run_name,
		"experiment_id": run_id,
		"config_path": str(config_path.relative_to(root)),
		"run_dir": str(run_dir.relative_to(root)),
		"analysis_group": _analysis_group_for_run(run_name),
		"seed_family": _seed_family_for_run(run_name),
		"seed": _extract_factor_value(cfg, "seed"),
		"scheduler": _extract_factor_value(cfg, "train", "scheduler", "name", default="none"),
		"early_stopping": _fmt_bool(bool(_extract_factor_value(cfg, "train", "early_stopping", "enabled", default=False))),
		"batch_size": _extract_factor_value(cfg, "data", "batch_size", default=32),
		"lr": _extract_factor_value(cfg, "train", "lr", default=0.0),
		"weight_decay": _extract_factor_value(cfg, "train", "weight_decay", default=0.0),
		"image_size": _extract_factor_value(cfg, "data", "image_size", default=224),
		"eval_resize_size": _extract_factor_value(cfg, "data", "eval_resize_size", default=256),
		"freeze_backbone": _fmt_bool(bool(_extract_factor_value(cfg, "train", "freeze_backbone", default=False))),
		"freeze_epochs": _extract_factor_value(cfg, "train", "freeze_epochs", default=0),
		"color_jitter": _fmt_bool(bool(_extract_factor_value(cfg, "data", "aug", "color_jitter", "enabled", default=False))),
		"epochs_planned": _extract_factor_value(cfg, "train", "epochs", default=len(rows)),
		"epochs_logged": len(rows),
		"best_epoch": int(best_row["epoch"]),
		"stopped_epoch": int(rows[-1]["epoch"]),
		"val_metrics_available": _fmt_bool(val_metrics.available),
		"val_metrics_source": val_metrics.source,
		"test_metrics_available": _fmt_bool(test_metrics.available),
		"test_metrics_source": test_metrics.source,
		"metrics_csv_available": _fmt_bool(metrics_path.exists()),
		"mlflow_json_available": _fmt_bool(mlflow_path.exists()),
		"mlflow_run_id": mlflow_payload.get("run_id", ""),
		"mlflow_experiment_name": mlflow_payload.get("experiment_name", ""),
		"val_loss": _fmt_float(val_metrics.loss),
		"val_acc1": _fmt_float(val_metrics.acc1),
		"val_acc5": _fmt_float(val_metrics.acc5),
		"test_loss": _fmt_float(test_metrics.loss if test_metrics.available else None),
		"test_acc1": _fmt_float(test_metrics.acc1 if test_metrics.available else None),
		"test_acc5": _fmt_float(test_metrics.acc5 if test_metrics.available else None),
	}


def collect_run_rows(root: Path | None = None) -> list[dict[str, Any]]:
	root = root or REPO_ROOT
	config_paths = sorted((root / "configs" / "experiments").glob("*.yaml"))
	rows = [_build_run_row(path, root=root) for path in config_paths]
	rows.sort(key=lambda row: row["run_name"])
	return rows


def _format_family_row(seed_family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
	test_acc1 = [float(row["test_acc1"]) for row in rows if row["test_acc1"]]
	test_acc5 = [float(row["test_acc5"]) for row in rows if row["test_acc5"]]
	test_loss = [float(row["test_loss"]) for row in rows if row["test_loss"]]
	return {
		"seed_family": seed_family,
		"recipe_label": SEED_FAMILIES[seed_family]["recipe_label"],
		"run_count": len(rows),
		"runs": " | ".join(row["run_name"] for row in rows),
		"seeds": " | ".join(str(row["seed"]) for row in rows),
		"test_acc1_mean": _fmt_float(sum(test_acc1) / len(test_acc1)),
		"test_acc1_std": _fmt_float(_sample_std(test_acc1)),
		"test_acc5_mean": _fmt_float(sum(test_acc5) / len(test_acc5)),
		"test_acc5_std": _fmt_float(_sample_std(test_acc5)),
		"test_loss_mean": _fmt_float(sum(test_loss) / len(test_loss)),
		"test_loss_std": _fmt_float(_sample_std(test_loss)),
	}


def _sample_std(values: list[float]) -> float:
	if len(values) < 2:
		return 0.0
	mean = sum(values) / len(values)
	variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
	return variance**0.5


def build_seed_family_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	rows_by_name = {row["run_name"]: row for row in run_rows}
	family_rows: list[dict[str, Any]] = []
	for seed_family, payload in SEED_FAMILIES.items():
		selected_rows = [rows_by_name[run_name] for run_name in payload["runs"] if run_name in rows_by_name]
		family_rows.append(_format_family_row(seed_family, selected_rows))
	return family_rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _screening_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	return [
		row
		for row in sorted(
			run_rows,
			key=lambda row: (
				-float(row["test_acc1"]) if row["test_acc1"] else 1.0,
				row["run_name"],
			),
		)
		if row["test_metrics_available"] == "true"
	]


def _plot_screening_overview(run_rows: list[dict[str, Any]], out_path: Path) -> None:
	sorted_rows = _screening_rows(run_rows)
	if not sorted_rows:
		return

	run_names = [row["run_name"] for row in sorted_rows]
	test_acc1 = [float(row["test_acc1"]) for row in sorted_rows]
	test_loss = [float(row["test_loss"]) for row in sorted_rows]
	colors = ["#1f77b4" if row["analysis_group"] != "augmentation" else "#d62728" for row in sorted_rows]

	fig, axes = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)

	ax = axes[0]
	ax.barh(run_names, test_acc1, color=colors)
	ax.invert_yaxis()
	ax.set_title("Test Top-1 Accuracy by Run")
	ax.set_xlabel("test_acc1")
	ax.grid(axis="x", alpha=0.25)
	for idx, value in enumerate(test_acc1):
		ax.text(value + 0.001, idx, f"{value:.3f}", va="center", fontsize=8)

	ax = axes[1]
	ax.barh(run_names, test_loss, color=colors)
	ax.invert_yaxis()
	ax.set_title("Test Loss by Run")
	ax.set_xlabel("test_loss")
	ax.grid(axis="x", alpha=0.25)
	for idx, value in enumerate(test_loss):
		ax.text(value + 0.002, idx, f"{value:.3f}", va="center", fontsize=8)

	fig.suptitle("Cross-Run Screening Overview", fontsize=15)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def _plot_seed_family_summary(family_rows: list[dict[str, Any]], out_path: Path) -> None:
	if not family_rows:
		return

	labels = [row["seed_family"] for row in family_rows]
	acc1 = [float(row["test_acc1_mean"]) for row in family_rows]
	acc1_std = [float(row["test_acc1_std"]) for row in family_rows]
	loss = [float(row["test_loss_mean"]) for row in family_rows]
	loss_std = [float(row["test_loss_std"]) for row in family_rows]

	fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

	ax = axes[0]
	ax.bar(labels, acc1, yerr=acc1_std, capsize=6, color=["#4c78a8", "#f58518", "#54a24b"])
	ax.set_title("Seed Families: Test Accuracy")
	ax.set_ylabel("test_acc1 mean ± std")
	ax.set_ylim(0.84, 0.89)
	ax.grid(axis="y", alpha=0.25)
	for idx, value in enumerate(acc1):
		ax.text(idx, value + 0.0008, f"{value:.3f}", ha="center", fontsize=8)

	ax = axes[1]
	ax.bar(labels, loss, yerr=loss_std, capsize=6, color=["#4c78a8", "#f58518", "#54a24b"])
	ax.set_title("Seed Families: Test Loss")
	ax.set_ylabel("test_loss mean ± std")
	ax.grid(axis="y", alpha=0.25)
	for idx, value in enumerate(loss):
		ax.text(idx, value + 0.002, f"{value:.3f}", ha="center", fontsize=8)

	fig.suptitle("Robustness-First Seed-Family Summary", fontsize=15)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


def _render_report(run_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]) -> str:
	screened = [row for row in run_rows if row["test_metrics_available"] == "true"]
	total_runs = len(run_rows)
	runs_with_eval_json = sum(
		1 for row in run_rows if row["val_metrics_available"] == "true" and row["test_metrics_available"] == "true"
	)
	runs_with_mlflow_json = sum(1 for row in run_rows if row["mlflow_json_available"] == "true")
	missing_test = [row["run_name"] for row in run_rows if row["test_metrics_available"] != "true"]
	best = next(row for row in screened if row["run_name"] == "exp17_cosine_es_img256_wd1e3_s42")
	exp13 = next(row for row in run_rows if row["run_name"] == "exp13_cosine_es_img256_s42")
	exp12 = next(row for row in run_rows if row["run_name"] == "exp12_cosine_es_img224_s42")
	exp14 = next(row for row in run_rows if row["run_name"] == "exp14_cosine_es_img320_s42")
	exp18 = next(row for row in run_rows if row["run_name"] == "exp18_cosine_es_img256_wd1e3_cj_s42")
	fam_exp02 = next(row for row in family_rows if row["seed_family"] == "exp02")
	fam_exp07 = next(row for row in family_rows if row["seed_family"] == "exp07")
	fam_exp17 = next(row for row in family_rows if row["seed_family"] == "exp17")

	screening_table = "\n".join(
		[
			"| Run | Change | Val acc@1 | Test acc@1 | Test loss |",
			"| --- | --- | --- | --- | --- |",
			f"| `exp12` | `img224` control | `{exp12['val_acc1']}` | `{exp12['test_acc1']}` | `{exp12['test_loss']}` |",
			f"| `exp13` | `img256` | `{exp13['val_acc1']}` | `{exp13['test_acc1']}` | `{exp13['test_loss']}` |",
			f"| `exp14` | `img320` | `{exp14['val_acc1']}` | `{exp14['test_acc1']}` | `{exp14['test_loss']}` |",
			f"| `exp17` | `img256 + wd=1e-3` | `{best['val_acc1']}` | `{best['test_acc1']}` | `{best['test_loss']}` |",
			f"| `exp18` | `img256 + wd=1e-3 + ColorJitter` | `{exp18['val_acc1']}` | `{exp18['test_acc1']}` | `{exp18['test_loss']}` |",
		]
	)

	family_table = "\n".join(
		[
			"| Seed family | Recipe | Test acc@1 mean ± std | Test loss mean ± std |",
			"| --- | --- | --- | --- |",
			f"| `exp02` | {fam_exp02['recipe_label']} | `{fam_exp02['test_acc1_mean']} ± {fam_exp02['test_acc1_std']}` | `{fam_exp02['test_loss_mean']} ± {fam_exp02['test_loss_std']}` |",
			f"| `exp07` | {fam_exp07['recipe_label']} | `{fam_exp07['test_acc1_mean']} ± {fam_exp07['test_acc1_std']}` | `{fam_exp07['test_loss_mean']} ± {fam_exp07['test_loss_std']}` |",
			f"| `exp17` | {fam_exp17['recipe_label']} | `{fam_exp17['test_acc1_mean']} ± {fam_exp17['test_acc1_std']}` | `{fam_exp17['test_loss_mean']} ± {fam_exp17['test_loss_std']}` |",
		]
	)

	missing_note = ", ".join(f"`{name}`" for name in missing_test)
	return f"""# Cross-Run Experiment Analytics

## Goal

Summarize the committed experiment history as a compact, reviewer-friendly analytics slice built from repo-native configs and run artifacts.

## Coverage

- Total committed runs: `{total_runs}`
- Runs with retained `val_metrics.json` and `test_metrics.json`: `{runs_with_eval_json}`
- Runs with `mlflow_run.json`: `{runs_with_mlflow_json}`
- Missing holdout sidecars: {missing_note}

MLflow is used only as optional sidecar metadata. The analytics slice treats configs and run artifacts as the source of truth.

## What Changed

{screening_table}

Interpretation:

- `img256` improved over `img224`.
- `img320` regressed and required a smaller batch size.
- `wd=1e-3` was the best single-seed screening candidate.
- mild `ColorJitter` was a clear negative result.

## What Helped Most

{family_table}

Interpretation:

- `exp07` looked strong on one seed, but the family mean and spread were weaker.
- `exp17` is the current showcase winner because it combines the best mean quality with the tightest robustness spread.
- The packet keeps single-run screening separate from seed-family conclusions on purpose.

## Notes

- The run-level artifact set stays explicit about missing holdout JSON for `exp03`, `exp03b`, and `exp04`.
- The public output stays compact: two figures and one short report.

## Figures

![Cross-run screening overview](assets/cross_run_screening_overview.png)

![Seed-family robustness summary](assets/cross_run_seed_family_summary.png)
"""


def write_outputs(root: Path = REPO_ROOT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	run_rows = collect_run_rows(root)
	family_rows = build_seed_family_rows(run_rows)

	artifacts_dir = root / "artifacts" / "experiment_analytics"
	_write_csv(artifacts_dir / "run_summary.csv", CSV_FIELDNAMES, run_rows)
	_write_csv(artifacts_dir / "seed_family_summary.csv", FAMILY_FIELDNAMES, family_rows)

	_plot_screening_overview(run_rows, root / "docs" / "experiments" / "assets" / "cross_run_screening_overview.png")
	_plot_seed_family_summary(family_rows, root / "docs" / "experiments" / "assets" / "cross_run_seed_family_summary.png")

	report_path = root / "docs" / "experiments" / "cross_run_analytics.md"
	report_path.write_text(_render_report(run_rows, family_rows), encoding="utf-8")

	return run_rows, family_rows


def main() -> None:
	parser = argparse.ArgumentParser(description="Build compact cross-run experiment analytics")
	parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root to analyze and write into")
	args = parser.parse_args()

	root = Path(args.root).resolve()
	run_rows, family_rows = write_outputs(root)
	print(f"Wrote {len(run_rows)} run rows and {len(family_rows)} seed-family rows to {root}")


if __name__ == "__main__":
	main()
