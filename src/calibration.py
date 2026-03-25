from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
	if isinstance(temperature, torch.Tensor):
		if temperature.numel() != 1:
			raise ValueError("Temperature tensor must be a scalar.")
		temp_value = float(temperature.detach().item())
	else:
		temp_value = float(temperature)

	if not math.isfinite(temp_value) or temp_value <= 0.0:
		raise ValueError(f"Temperature must be a finite positive scalar, got {temp_value}.")
	return logits / temp_value


def negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor) -> float:
	return float(F.cross_entropy(logits, targets).item())


def top1_confidence_and_correctness(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	if logits.ndim != 2:
		raise ValueError(f"Expected logits with shape [N, C], got {tuple(logits.shape)}.")
	if targets.ndim != 1:
		raise ValueError(f"Expected targets with shape [N], got {tuple(targets.shape)}.")
	if logits.size(0) != targets.size(0):
		raise ValueError("Logits and targets must have the same number of samples.")

	probs = F.softmax(logits, dim=1)
	confidences, pred_indices = probs.max(dim=1)
	correctness = pred_indices.eq(targets).to(dtype=torch.float32)
	return confidences.detach(), correctness.detach()


def build_reliability_bins(
	confidences: torch.Tensor,
	correctness: torch.Tensor,
	num_bins: int = 15,
) -> list[dict[str, float | int]]:
	if confidences.ndim != 1 or correctness.ndim != 1:
		raise ValueError("Confidences and correctness must be 1D tensors.")
	if confidences.size(0) != correctness.size(0):
		raise ValueError("Confidences and correctness must have matching lengths.")
	if num_bins <= 0:
		raise ValueError("num_bins must be positive.")

	total = int(confidences.size(0))
	if total == 0:
		return []

	conf = confidences.detach().cpu()
	corr = correctness.detach().cpu()
	boundaries = torch.linspace(0.0, 1.0, num_bins + 1)

	rows: list[dict[str, float | int]] = []
	for bin_idx in range(num_bins):
		bin_lower = float(boundaries[bin_idx].item())
		bin_upper = float(boundaries[bin_idx + 1].item())
		if bin_idx == num_bins - 1:
			mask = (conf >= bin_lower) & (conf <= bin_upper)
		else:
			mask = (conf >= bin_lower) & (conf < bin_upper)

		count = int(mask.sum().item())
		if count > 0:
			bin_conf = float(conf[mask].mean().item())
			bin_acc = float(corr[mask].mean().item())
		else:
			bin_conf = 0.0
			bin_acc = 0.0

		rows.append(
			{
				"bin_index": bin_idx,
				"bin_lower": bin_lower,
				"bin_upper": bin_upper,
				"count": count,
				"mean_confidence": bin_conf,
				"accuracy": bin_acc,
			}
		)
	return rows


def expected_calibration_error(bins: Sequence[dict[str, float | int]]) -> float:
	total = sum(int(b["count"]) for b in bins)
	if total == 0:
		return 0.0

	ece = 0.0
	for row in bins:
		count = int(row["count"])
		if count <= 0:
			continue
		acc = float(row["accuracy"])
		conf = float(row["mean_confidence"])
		ece += abs(acc - conf) * (count / total)
	return float(ece)


def ece_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> float:
	confidences, correctness = top1_confidence_and_correctness(logits, targets)
	bins = build_reliability_bins(confidences=confidences, correctness=correctness, num_bins=num_bins)
	return expected_calibration_error(bins)


def calibration_metrics(logits: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> dict[str, float]:
	confidences, correctness = top1_confidence_and_correctness(logits, targets)
	bins = build_reliability_bins(confidences=confidences, correctness=correctness, num_bins=num_bins)
	return {
		"nll": negative_log_likelihood(logits, targets),
		"ece": expected_calibration_error(bins),
		"acc1": float(correctness.mean().item()) if correctness.numel() else 0.0,
	}


def fit_temperature(
	val_logits: torch.Tensor,
	val_targets: torch.Tensor,
	max_iter: int = 100,
	lr: float = 0.1,
) -> float:
	if val_logits.ndim != 2:
		raise ValueError(f"Expected val_logits with shape [N, C], got {tuple(val_logits.shape)}.")
	if val_targets.ndim != 1:
		raise ValueError(f"Expected val_targets with shape [N], got {tuple(val_targets.shape)}.")
	if val_logits.size(0) != val_targets.size(0):
		raise ValueError("Validation logits and targets must have matching sample counts.")
	if val_logits.size(0) == 0:
		raise ValueError("Validation split is empty; cannot fit temperature.")

	logits = val_logits.detach()
	targets = val_targets.detach()
	log_temperature = nn.Parameter(torch.zeros(1, device=logits.device, dtype=logits.dtype))
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.LBFGS(
		[log_temperature],
		lr=float(lr),
		max_iter=int(max_iter),
		line_search_fn="strong_wolfe",
	)

	def closure() -> torch.Tensor:
		optimizer.zero_grad()
		temperature = torch.exp(log_temperature).clamp(min=1e-6, max=1e6)
		loss = criterion(logits / temperature, targets)
		loss.backward()
		return loss

	optimizer.step(closure)
	temperature = float(torch.exp(log_temperature.detach()).item())
	if not math.isfinite(temperature) or temperature <= 0.0:
		raise RuntimeError(f"Fitted invalid temperature: {temperature}")
	return temperature


def confidence_threshold_sweep(
	confidences: torch.Tensor,
	correctness: torch.Tensor,
	thresholds: Sequence[float] | None = None,
) -> list[dict[str, float | int]]:
	if confidences.ndim != 1 or correctness.ndim != 1:
		raise ValueError("Confidences and correctness must be 1D tensors.")
	if confidences.size(0) != correctness.size(0):
		raise ValueError("Confidences and correctness must have matching lengths.")

	if thresholds is None:
		thresholds = [i / 100.0 for i in range(101)]
	if not thresholds:
		raise ValueError("thresholds must not be empty.")

	conf = confidences.detach().cpu()
	corr = correctness.detach().cpu()
	total = int(conf.numel())
	if total == 0:
		return []

	rows: list[dict[str, float | int]] = []
	for threshold in thresholds:
		thr = float(threshold)
		mask = conf >= thr
		retained_count = int(mask.sum().item())
		coverage = retained_count / total
		if retained_count > 0:
			retained_accuracy = float(corr[mask].mean().item())
		else:
			retained_accuracy = 0.0

		rows.append(
			{
				"threshold": thr,
				"retained_count": retained_count,
				"coverage": float(coverage),
				"retained_accuracy": retained_accuracy,
			}
		)
	return rows


def threshold_sweep_from_logits(
	logits: torch.Tensor,
	targets: torch.Tensor,
	thresholds: Sequence[float] | None = None,
) -> list[dict[str, float | int]]:
	confidences, correctness = top1_confidence_and_correctness(logits, targets)
	return confidence_threshold_sweep(confidences=confidences, correctness=correctness, thresholds=thresholds)
