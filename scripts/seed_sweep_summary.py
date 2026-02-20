from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _read_metric(run_dir: str, filename: str) -> dict:
	path = Path(run_dir) / "artifacts" / filename
	if not path.exists():
		raise FileNotFoundError(f"Missing metrics file: {path}")
	return json.loads(path.read_text(encoding="utf-8"))


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
		test = _read_metric(run, "test_metrics.json")
		val = _read_metric(run, "val_metrics.json")
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
