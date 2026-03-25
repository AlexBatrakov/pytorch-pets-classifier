import torch

from src.calibration import (
	apply_temperature,
	build_reliability_bins,
	confidence_threshold_sweep,
	ece_from_logits,
	fit_temperature,
	negative_log_likelihood,
)


def test_apply_temperature_requires_positive_scalar():
	logits = torch.tensor([[1.0, 0.0]])
	try:
		apply_temperature(logits, 0.0)
		assert False, "Expected ValueError for zero temperature."
	except ValueError:
		pass

	try:
		apply_temperature(logits, -1.0)
		assert False, "Expected ValueError for negative temperature."
	except ValueError:
		pass


def test_fit_temperature_improves_or_matches_val_nll_on_synthetic_data():
	logits = torch.tensor(
		[
			[5.0, 1.0],
			[1.0, 5.0],
			[5.0, 1.0],
			[1.0, 5.0],
			[5.0, 1.0],
			[1.0, 5.0],
		],
		dtype=torch.float32,
	)
	targets = torch.tensor([0, 1, 1, 1, 0, 0], dtype=torch.long)

	temp = fit_temperature(logits, targets, max_iter=80, lr=0.1)
	assert temp > 0.0

	uncal_nll = negative_log_likelihood(logits, targets)
	cal_nll = negative_log_likelihood(apply_temperature(logits, temp), targets)
	assert cal_nll <= uncal_nll


def test_ece_is_near_zero_for_high_confidence_perfect_predictions():
	logits = torch.tensor(
		[
			[20.0, 0.0],
			[0.0, 20.0],
			[20.0, 0.0],
			[0.0, 20.0],
		],
		dtype=torch.float32,
	)
	targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
	ece = ece_from_logits(logits, targets, num_bins=10)
	assert ece < 1e-4


def test_reliability_bins_handle_empty_bins_and_preserve_total_count():
	confidences = torch.tensor([0.05, 0.95], dtype=torch.float32)
	correctness = torch.tensor([1.0, 0.0], dtype=torch.float32)
	bins = build_reliability_bins(confidences, correctness, num_bins=5)

	assert len(bins) == 5
	assert sum(int(row["count"]) for row in bins) == 2
	assert any(int(row["count"]) == 0 for row in bins)


def test_confidence_threshold_sweep_coverage_is_monotonic():
	confidences = torch.tensor([0.10, 0.60, 0.90], dtype=torch.float32)
	correctness = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
	thresholds = [0.0, 0.5, 0.7, 1.0]
	rows = confidence_threshold_sweep(confidences, correctness, thresholds=thresholds)

	coverages = [float(row["coverage"]) for row in rows]
	for left, right in zip(coverages, coverages[1:]):
		assert left >= right
