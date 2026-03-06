from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from PIL import Image
from torchvision import transforms

from src.inference import (
	Predictor,
	_predict_transform,
	_resolve_class_names,
	_resolve_preprocess_sizes,
)


def _size_tuple(value) -> tuple[int, int]:
	if isinstance(value, int):
		return (value, value)
	return tuple(value)


def test_predict_preprocess_defaults_match_legacy_behavior() -> None:
	image_size, eval_resize_size = _resolve_preprocess_sizes({})
	assert image_size == 224
	assert eval_resize_size == 256

	tf = _predict_transform(image_size=image_size, eval_resize_size=eval_resize_size)
	assert isinstance(tf.transforms[0], transforms.Resize)
	assert _size_tuple(tf.transforms[0].size) == (256, 256)
	assert isinstance(tf.transforms[1], transforms.CenterCrop)
	assert _size_tuple(tf.transforms[1].size) == (224, 224)


def test_predict_preprocess_uses_sizes_from_checkpoint_config() -> None:
	ckpt = {"config": {"data": {"image_size": 256, "eval_resize_size": 292}}}
	image_size, eval_resize_size = _resolve_preprocess_sizes(ckpt)
	assert image_size == 256
	assert eval_resize_size == 292


def test_resolve_class_names_requires_checkpoint_classes() -> None:
	with pytest.raises(ValueError, match="Checkpoint missing class names."):
		_resolve_class_names({})


class _DummyModel(nn.Module):
	def __init__(self, logits: torch.Tensor) -> None:
		super().__init__()
		self.register_buffer("_logits", logits)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self._logits.unsqueeze(0).expand(x.size(0), -1)


def test_predictor_returns_topk_predictions_and_latency() -> None:
	predictor = Predictor(
		model=_DummyModel(torch.tensor([0.2, 2.0, -0.5])),
		class_names=["cat", "dog", "bird"],
		transform=lambda image: torch.zeros(3, 4, 4),
		device="cpu",
	)

	result = predictor.predict_pil(Image.new("RGB", (8, 8), color="white"), top_k=2)

	assert result.label == "dog"
	assert result.score == pytest.approx(result.top_k[0].score)
	assert [prediction.label for prediction in result.top_k] == ["dog", "cat"]
	assert len(result.top_k) == 2
	assert result.inference_ms >= 0.0


def test_predictor_from_checkpoint_uses_checkpoint_metadata(monkeypatch) -> None:
	ckpt = {
		"model_state_dict": {"mock_weight": torch.tensor([1.0])},
		"class_names": ["cat", "dog"],
		"config": {"data": {"image_size": 256, "eval_resize_size": 292}},
	}

	class _CheckpointModel(_DummyModel):
		def __init__(self) -> None:
			super().__init__(torch.tensor([0.0, 1.0]))
			self.loaded_state_dict = None

		def load_state_dict(self, state_dict):  # type: ignore[override]
			self.loaded_state_dict = state_dict
			return None

	model = _CheckpointModel()

	monkeypatch.setattr("src.inference.load_checkpoint", lambda path, map_location="cpu": ckpt)
	monkeypatch.setattr(
		"src.inference.build_model",
		lambda num_classes, pretrained, freeze_backbone: model,
	)
	monkeypatch.setattr("src.inference.get_device", lambda: torch.device("cpu"))

	predictor = Predictor.from_checkpoint("dummy.pt")

	assert predictor.class_names == ["cat", "dog"]
	assert isinstance(predictor.transform.transforms[0], transforms.Resize)
	assert _size_tuple(predictor.transform.transforms[0].size) == (292, 292)
	assert isinstance(predictor.transform.transforms[1], transforms.CenterCrop)
	assert _size_tuple(predictor.transform.transforms[1].size) == (256, 256)
	assert model.loaded_state_dict == ckpt["model_state_dict"]
