from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


def _train_transform() -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def _val_transform() -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def _resolve_target_type(data_dir: str, download: bool) -> str:
	try:
		_ = OxfordIIITPet(
			root=data_dir,
			split="trainval",
			target_types="breed",
			download=download,
			transform=None,
		)
		return "breed"
	except ValueError:
		return "category"


def build_datasets(config) -> Tuple[Subset, Subset, List[str]]:
	data_cfg = config["data"]
	seed = config.get("seed", 42)
	data_dir = config["paths"]["data_dir"]
	download = data_cfg.get("download", True)

	target_type = _resolve_target_type(data_dir, download)

	base_ds = OxfordIIITPet(
		root=data_dir,
		split="trainval",
		target_types=target_type,
		download=download,
		transform=None,
	)

	class_names = list(base_ds.classes)

	total = len(base_ds)
	val_size = int(total * data_cfg.get("val_split", 0.2))
	train_size = total - val_size
	gen = torch.Generator().manual_seed(seed)
	indices = torch.randperm(total, generator=gen).tolist()
	train_idx = indices[:train_size]
	val_idx = indices[train_size:]

	train_ds = OxfordIIITPet(
		root=data_dir,
		split="trainval",
		target_types=target_type,
		download=False,
		transform=_train_transform(),
	)
	val_ds = OxfordIIITPet(
		root=data_dir,
		split="trainval",
		target_types=target_type,
		download=False,
		transform=_val_transform(),
	)

	train_subset = Subset(train_ds, train_idx)
	val_subset = Subset(val_ds, val_idx)

	return train_subset, val_subset, class_names


def build_loaders(config):
	data_cfg = config["data"]
	train_ds, val_ds, class_names = build_datasets(config)

	train_loader = DataLoader(
		train_ds,
		batch_size=data_cfg.get("batch_size", 32),
		shuffle=True,
		num_workers=data_cfg.get("num_workers", 0),
		pin_memory=data_cfg.get("pin_memory", False),
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=data_cfg.get("batch_size", 32),
		shuffle=False,
		num_workers=data_cfg.get("num_workers", 0),
		pin_memory=data_cfg.get("pin_memory", False),
	)

	return train_loader, val_loader, class_names


def build_test_loader(config):
	data_cfg = config["data"]
	data_dir = config["paths"]["data_dir"]
	download = data_cfg.get("download", True)

	target_type = _resolve_target_type(data_dir, download)
	base_ds = OxfordIIITPet(
		root=data_dir,
		split="trainval",
		target_types=target_type,
		download=download,
		transform=None,
	)
	class_names = list(base_ds.classes)

	test_ds = OxfordIIITPet(
		root=data_dir,
		split="test",
		target_types=target_type,
		download=False,
		transform=_val_transform(),
	)

	test_loader = DataLoader(
		test_ds,
		batch_size=data_cfg.get("batch_size", 32),
		shuffle=False,
		num_workers=data_cfg.get("num_workers", 0),
		pin_memory=data_cfg.get("pin_memory", False),
	)

	return test_loader, class_names
