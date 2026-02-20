from __future__ import annotations

import torch

from src.config import apply_overrides
from src.utils import accuracy_topk, get_device


def test_apply_overrides_updates_nested_and_keeps_original() -> None:
	base_cfg = {
		"seed": 42,
		"train": {"epochs": 5, "lr": 1e-3},
		"data": {"batch_size": 32},
	}
	overrides = {
		"seed": 7,
		"train.lr": 3e-4,
		"train.weight_decay": 0.01,
		"new_field.enabled": True,
	}

	new_cfg = apply_overrides(base_cfg, overrides)

	assert new_cfg["seed"] == 7
	assert new_cfg["train"]["lr"] == 3e-4
	assert new_cfg["train"]["weight_decay"] == 0.01
	assert new_cfg["new_field"]["enabled"] is True

	# Ensure we do not mutate the original config dict.
	assert base_cfg["seed"] == 42
	assert base_cfg["train"]["lr"] == 1e-3
	assert "weight_decay" not in base_cfg["train"]
	assert "new_field" not in base_cfg


def test_apply_overrides_skips_none_and_creates_deep_branch() -> None:
	base_cfg = {
		"train": {"epochs": 5, "lr": 1e-3},
		"logging": {"print_every": 50},
	}
	overrides = {
		"train.lr": None,
		"train.epochs": 10,
		"model.head.dropout": 0.2,
		"logging.print_every": None,
	}

	new_cfg = apply_overrides(base_cfg, overrides)

	assert new_cfg["train"]["epochs"] == 10
	assert new_cfg["train"]["lr"] == 1e-3
	assert new_cfg["model"]["head"]["dropout"] == 0.2
	assert new_cfg["logging"]["print_every"] == 50


def test_accuracy_topk_returns_expected_values() -> None:
	logits = torch.tensor(
		[
			[5.0, 1.0, 0.0],
			[0.2, 0.1, 2.0],
			[0.1, 3.0, 0.2],
			[2.0, 1.9, 1.8],
		]
	)
	targets = torch.tensor([0, 2, 1, 1])

	acc1, acc2 = accuracy_topk(logits, targets, topk=(1, 2))
	assert abs(acc1 - 0.75) < 1e-8
	assert abs(acc2 - 1.0) < 1e-8


def test_get_device_prefers_mps_then_cuda_then_cpu(monkeypatch) -> None:
	monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
	monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
	assert get_device().type == "cpu"

	monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
	assert get_device().type == "cuda"

	monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
	assert get_device().type == "mps"
