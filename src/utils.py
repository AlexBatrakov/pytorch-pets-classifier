from __future__ import annotations

import os
import random
from typing import Any, Iterable, Tuple

import numpy as np
import torch


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Any:
	"""
	Load a full training checkpoint across PyTorch versions.

	PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
	checkpoints that include metadata beyond tensors.
	"""
	try:
		return torch.load(path, map_location=map_location, weights_only=False)
	except TypeError:
		# Older PyTorch versions may not support the weights_only argument.
		return torch.load(path, map_location=map_location)


def set_seed(seed: int, deterministic: bool = False) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	os.environ["PYTHONHASHSEED"] = str(seed)
	if deterministic:
		torch.use_deterministic_algorithms(True)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
	if torch.backends.mps.is_available():
		return torch.device("mps")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def accuracy_topk(
	logits: torch.Tensor,
	targets: torch.Tensor,
	topk: Tuple[int, ...] = (1,),
) -> Iterable[float]:
	with torch.no_grad():
		max_k = max(topk)
		_, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
		pred = pred.t()
		correct = pred.eq(targets.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0)
			res.append((correct_k / targets.size(0)).item())
		return res


class AverageMeter:
	def __init__(self) -> None:
		self.reset()

	def reset(self) -> None:
		self.val = 0.0
		self.avg = 0.0
		self.sum = 0.0
		self.count = 0

	def update(self, val: float, n: int = 1) -> None:
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count if self.count else 0.0
