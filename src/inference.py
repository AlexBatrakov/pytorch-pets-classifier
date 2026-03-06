from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import torch
from PIL import Image
from torchvision import transforms

from .model import build_model
from .utils import get_device, load_checkpoint


DEFAULT_IMAGE_SIZE = 224
DEFAULT_EVAL_RESIZE_SIZE = 256
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class TopPrediction:
	label: str
	score: float


@dataclass(frozen=True)
class InferenceResult:
	label: str
	score: float
	top_k: list[TopPrediction]
	inference_ms: float


def _predict_transform(image_size: int, eval_resize_size: int) -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize(eval_resize_size),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def _resolve_preprocess_sizes(ckpt: dict) -> tuple[int, int]:
	cfg = ckpt.get("config", {}) or {}
	data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
	image_size = int(data_cfg.get("image_size", DEFAULT_IMAGE_SIZE))
	eval_resize_size = int(data_cfg.get("eval_resize_size", DEFAULT_EVAL_RESIZE_SIZE))
	return image_size, eval_resize_size


def _resolve_class_names(ckpt: dict) -> list[str]:
	class_names = ckpt.get("class_names") or []
	if not class_names:
		raise ValueError("Checkpoint missing class names.")
	return list(class_names)


def _topk_predictions(
	probs: torch.Tensor,
	class_names: Sequence[str],
	top_k: int = DEFAULT_TOP_K,
) -> list[TopPrediction]:
	if probs.ndim != 1:
		raise ValueError("Expected a 1D probabilities tensor.")
	if not class_names:
		raise ValueError("Class names must not be empty.")

	limit = max(1, min(int(top_k), len(class_names)))
	top_probs, top_idx = torch.topk(probs, k=limit)
	return [
		TopPrediction(label=class_names[idx], score=float(prob))
		for prob, idx in zip(top_probs.tolist(), top_idx.tolist())
	]


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
	if device is None:
		return get_device()
	if isinstance(device, torch.device):
		return device
	return torch.device(device)


class Predictor:
	def __init__(
		self,
		model: torch.nn.Module,
		class_names: Sequence[str],
		transform: Callable[[Image.Image], torch.Tensor],
		device: str | torch.device | None = None,
	) -> None:
		self.class_names = list(class_names)
		if not self.class_names:
			raise ValueError("Class names must not be empty.")

		self.device = _resolve_device(device)
		self.transform = transform
		self.model = model.to(self.device)
		self.model.eval()

	@classmethod
	def from_checkpoint(
		cls,
		ckpt_path: str,
		device: str | torch.device | None = None,
	) -> Predictor:
		ckpt = load_checkpoint(ckpt_path, map_location="cpu")
		class_names = _resolve_class_names(ckpt)
		image_size, eval_resize_size = _resolve_preprocess_sizes(ckpt)
		model = build_model(
			num_classes=len(class_names),
			pretrained=False,
			freeze_backbone=False,
		)
		model.load_state_dict(ckpt["model_state_dict"])
		return cls(
			model=model,
			class_names=class_names,
			transform=_predict_transform(
				image_size=image_size,
				eval_resize_size=eval_resize_size,
			),
			device=device,
		)

	def predict_pil(
		self,
		image: Image.Image,
		top_k: int = DEFAULT_TOP_K,
	) -> InferenceResult:
		x = self.transform(image.convert("RGB"))
		x = x.unsqueeze(0).to(self.device)

		with torch.no_grad():
			started_at = perf_counter()
			logits = self.model(x)
			probs = torch.softmax(logits, dim=1).squeeze(0)
			inference_ms = (perf_counter() - started_at) * 1000.0

		top_predictions = _topk_predictions(probs, self.class_names, top_k=top_k)
		top1 = top_predictions[0]
		return InferenceResult(
			label=top1.label,
			score=top1.score,
			top_k=top_predictions,
			inference_ms=inference_ms,
		)

	def predict_path(
		self,
		image_path: str,
		top_k: int = DEFAULT_TOP_K,
	) -> InferenceResult:
		with Image.open(image_path) as image:
			return self.predict_pil(image, top_k=top_k)
