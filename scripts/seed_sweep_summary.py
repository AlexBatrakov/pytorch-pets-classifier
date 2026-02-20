from __future__ import annotations

import argparse
import json
import subprocess
import statistics
import sys
from pathlib import Path


def _read_metric(run_dir: str, filename: str) -> dict:
	path = Path(run_dir) / "artifacts" / filename
	if not path.exists():
		raise FileNotFoundError
	return json.loads(path.read_text(encoding="utf-8"))


def _read_best_val_from_csv(run_dir: str) -> dict:
	import csv

	path = Path(run_dir) / "artifacts" / "metrics.csv"
	if not path.exists():
		raise FileNotFoundError(f"Missing fallback CSV file: {path}")
	rows = []
	with path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(
				{
					"epoch": int(row["epoch"]),
					"val_loss": float(row["val_loss"]),
					"val_acc1": float(row["val_acc1"]),
					"val_acc5": float(row["val_acc5"]),
				}
			)
	if not rows:
		raise ValueError(f"No rows in metrics file: {path}")
	best = max(rows, key=lambda r: r["val_acc1"])
	return {"split": "val", "loss": best["val_loss"], "acc1": best["val_acc1"], "acc5": best["val_acc5"]}


def _infer_test_from_eval(run_dir: str) -> dict:
	ckpt = Path(run_dir) / "checkpoints" / "best.pt"
	if not ckpt.exists():
		raise FileNotFoundError(f"Missing checkpoint for fallback eval: {ckpt}")
	cmd = [sys.executable, "-m", "src.eval", "--ckpt", str(ckpt), "--split", "test"]
	res = subprocess.run(cmd, check=True, capture_output=True, text=True)
	out = res.stdout.strip().splitlines()
	metric_line = next((line for line in out if line.startswith("Test loss ")), "")
	if not metric_line:
		raise ValueError(f"Could not parse test metrics from eval output for run: {run_dir}")
	# Example: "Test loss 0.4570 | acc@1 0.875 | acc@5 0.984"
	parts = metric_line.replace("Test loss ", "").split("|")
	loss = float(parts[0].strip())
	acc1 = float(parts[1].replace("acc@1", "").strip())
	acc5 = float(parts[2].replace("acc@5", "").strip())
	return {"split": "test", "loss": loss, "acc1": acc1, "acc5": acc5}


def main() -> None:
	parser = argparse.ArgumentParser(description="Summarize seed sweep metrics from run directories")
	parser.add_argument(
		"--runs",
		nargs="+",
		required=True,
		help="Run directories, e.g. runs/exp02... runs/exp05... runs/exp06...",
	)
	args = parser.parse_args()

	test_rows = []
	for run in args.runs:
		try:
			test = _read_metric(run, "test_metrics.json")
		except FileNotFoundError:
			test = _infer_test_from_eval(run)
		try:
			val = _read_metric(run, "val_metrics.json")
		except FileNotFoundError:
			val = _read_best_val_from_csv(run)
		test_rows.append(
			{
				"run": run,
				"val_acc1": float(val["acc1"]),
				"test_acc1": float(test["acc1"]),
				"test_acc5": float(test["acc5"]),
				"test_loss": float(test["loss"]),
			}
		)

	print("Per-run metrics:")
	for row in test_rows:
		print(
			f"- {row['run']}: "
			f"val_acc1={row['val_acc1']:.4f}, "
			f"test_acc1={row['test_acc1']:.4f}, "
			f"test_acc5={row['test_acc5']:.4f}, "
			f"test_loss={row['test_loss']:.4f}"
		)

	test_acc1 = [r["test_acc1"] for r in test_rows]
	test_acc5 = [r["test_acc5"] for r in test_rows]
	test_loss = [r["test_loss"] for r in test_rows]

	def _mean_std(values: list[float]) -> tuple[float, float]:
		if len(values) == 1:
			return values[0], 0.0
		return statistics.mean(values), statistics.stdev(values)

	acc1_m, acc1_s = _mean_std(test_acc1)
	acc5_m, acc5_s = _mean_std(test_acc5)
	loss_m, loss_s = _mean_std(test_loss)

	print("\nSeed sweep summary:")
	print(f"- test_acc1: {acc1_m:.4f} +/- {acc1_s:.4f}")
	print(f"- test_acc5: {acc5_m:.4f} +/- {acc5_s:.4f}")
	print(f"- test_loss: {loss_m:.4f} +/- {loss_s:.4f}")


if __name__ == "__main__":
	main()
