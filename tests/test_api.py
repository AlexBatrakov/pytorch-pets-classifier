from __future__ import annotations

from io import BytesIO

import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from src.api import create_app
from src.inference import Predictor


class _DummyModel(nn.Module):
	def __init__(self, logits: torch.Tensor) -> None:
		super().__init__()
		self.register_buffer("_logits", logits)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self._logits.unsqueeze(0).expand(x.size(0), -1)


def _build_predictor() -> Predictor:
	return Predictor(
		model=_DummyModel(torch.tensor([0.2, 2.0, -0.5])),
		class_names=["cat", "dog", "bird"],
		transform=lambda image: torch.zeros(3, 4, 4),
		device="cpu",
	)


def _png_bytes() -> bytes:
	buffer = BytesIO()
	Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
	return buffer.getvalue()


def test_health_returns_status_and_model_metadata() -> None:
	app = create_app(predictor=_build_predictor(), model_version="test-model")

	with TestClient(app) as client:
		response = client.get("/health")

	assert response.status_code == 200
	assert response.json() == {
		"status": "ok",
		"model_version": "test-model",
		"device": "cpu",
	}


def test_predict_returns_label_scores_and_latency() -> None:
	app = create_app(
		predictor=_build_predictor(),
		model_version="test-model",
		default_top_k=2,
	)

	with TestClient(app) as client:
		response = client.post(
			"/predict",
			files={"file": ("pet.png", _png_bytes(), "image/png")},
		)

	assert response.status_code == 200
	body = response.json()
	assert body["label"] == "dog"
	assert body["model_version"] == "test-model"
	assert len(body["top_k"]) == 2
	assert [item["label"] for item in body["top_k"]] == ["dog", "cat"]
	assert body["score"] == body["top_k"][0]["score"]
	assert body["inference_ms"] >= 0.0


def test_predict_rejects_invalid_image_upload() -> None:
	app = create_app(predictor=_build_predictor())

	with TestClient(app) as client:
		response = client.post(
			"/predict",
			files={"file": ("notes.txt", b"not an image", "text/plain")},
		)

	assert response.status_code == 400
	assert response.json()["detail"] == "Uploaded file is not a valid image."


def test_predict_rejects_oversized_upload() -> None:
	app = create_app(predictor=_build_predictor(), max_upload_bytes=16)

	with TestClient(app) as client:
		response = client.post(
			"/predict",
			files={"file": ("pet.png", _png_bytes(), "image/png")},
		)

	assert response.status_code == 413
	assert "exceeds the 16-byte limit" in response.json()["detail"]
