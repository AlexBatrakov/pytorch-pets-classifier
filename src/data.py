from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


DEFAULT_IMAGE_SIZE = 224
DEFAULT_EVAL_RESIZE_SIZE = 256


def _get_image_sizes(data_cfg) -> Tuple[int, int]:
	image_size = int(data_cfg.get("image_size", DEFAULT_IMAGE_SIZE))
	eval_resize_size = int(data_cfg.get("eval_resize_size", DEFAULT_EVAL_RESIZE_SIZE))
	return image_size, eval_resize_size


def _train_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.RandomResizedCrop(image_size),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def _val_transform(
	image_size: int = DEFAULT_IMAGE_SIZE,
	eval_resize_size: int = DEFAULT_EVAL_RESIZE_SIZE,
) -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize(eval_resize_size),
			transforms.CenterCrop(image_size),
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
	image_size, eval_resize_size = _get_image_sizes(data_cfg)
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
		transform=_train_transform(image_size=image_size),
	)
	val_ds = OxfordIIITPet(
		root=data_dir,
		split="trainval",
		target_types=target_type,
		download=False,
		transform=_val_transform(image_size=image_size, eval_resize_size=eval_resize_size),
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
	image_size, eval_resize_size = _get_image_sizes(data_cfg)
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
		transform=_val_transform(image_size=image_size, eval_resize_size=eval_resize_size),
	)

	test_loader = DataLoader(
		test_ds,
		batch_size=data_cfg.get("batch_size", 32),
		shuffle=False,
		num_workers=data_cfg.get("num_workers", 0),
		pin_memory=data_cfg.get("pin_memory", False),
	)

	return test_loader, class_names
