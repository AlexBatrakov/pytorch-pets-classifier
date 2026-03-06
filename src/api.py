from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError
import uvicorn

from .inference import DEFAULT_TOP_K, Predictor


DEFAULT_MODEL_PATH = "runs/exp17_cosine_es_img256_wd1e3_s42/checkpoints/best.pt"
DEFAULT_MODEL_VERSION = "exp17_cosine_es_img256_wd1e3_s42"
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_PORT = 8080

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiSettings:
	model_path: str
	device: str | None
	model_version: str
	max_upload_bytes: int
	default_top_k: int
	port: int

	@classmethod
	def from_env(cls) -> ApiSettings:
		return cls(
			model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
			device=os.getenv("DEVICE"),
			model_version=os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION),
			max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
			default_top_k=max(1, int(os.getenv("DEFAULT_TOP_K", DEFAULT_TOP_K))),
			port=int(os.getenv("PORT", DEFAULT_PORT)),
		)


def _load_image_from_bytes(contents: bytes) -> Image.Image:
	try:
		with Image.open(BytesIO(contents)) as image:
			image.load()
			return image.copy()
	except (UnidentifiedImageError, OSError) as exc:
		raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


async def _read_upload_bytes(upload_file: UploadFile, max_upload_bytes: int) -> bytes:
	contents = await upload_file.read(max_upload_bytes + 1)
	if not contents:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")
	if len(contents) > max_upload_bytes:
		raise HTTPException(
			status_code=413,
			detail=f"Uploaded file exceeds the {max_upload_bytes}-byte limit.",
		)
	return contents


def create_app(
	*,
	predictor: Predictor | None = None,
	model_path: str | None = None,
	device: str | None = None,
	model_version: str | None = None,
	max_upload_bytes: int | None = None,
	default_top_k: int | None = None,
) -> FastAPI:
	env_settings = ApiSettings.from_env()
	settings = ApiSettings(
		model_path=model_path or env_settings.model_path,
		device=device if device is not None else env_settings.device,
		model_version=model_version or env_settings.model_version,
		max_upload_bytes=max_upload_bytes or env_settings.max_upload_bytes,
		default_top_k=max(1, default_top_k or env_settings.default_top_k),
		port=env_settings.port,
	)

	@asynccontextmanager
	async def lifespan(app: FastAPI):
		resolved_predictor = predictor
		if resolved_predictor is None:
			resolved_predictor = Predictor.from_checkpoint(
				settings.model_path,
				device=settings.device,
			)
		app.state.predictor = resolved_predictor
		app.state.settings = settings
		yield

	app = FastAPI(title="PyTorch Pets Classifier API", version="0.1.0", lifespan=lifespan)

	@app.middleware("http")
	async def log_requests(request: Request, call_next):
		started_at = perf_counter()
		try:
			response = await call_next(request)
		except Exception:
			duration_ms = (perf_counter() - started_at) * 1000.0
			logger.exception(
				"request_failed method=%s path=%s duration_ms=%.2f",
				request.method,
				request.url.path,
				duration_ms,
			)
			raise

		duration_ms = (perf_counter() - started_at) * 1000.0
		logger.info(
			"request_completed method=%s path=%s status=%s duration_ms=%.2f",
			request.method,
			request.url.path,
			response.status_code,
			duration_ms,
		)
		return response

	@app.get("/health")
	async def health(request: Request) -> dict[str, str]:
		loaded_predictor: Predictor = request.app.state.predictor
		app_settings: ApiSettings = request.app.state.settings
		return {
			"status": "ok",
			"model_version": app_settings.model_version,
			"device": str(loaded_predictor.device),
		}

	@app.post("/predict")
	async def predict(
		request: Request,
		file: UploadFile = File(...),
		top_k: int | None = Query(default=None, ge=1),
	) -> dict[str, object]:
		contents = await _read_upload_bytes(
			file,
			request.app.state.settings.max_upload_bytes,
		)
		image = _load_image_from_bytes(contents)

		loaded_predictor: Predictor = request.app.state.predictor
		app_settings: ApiSettings = request.app.state.settings
		requested_top_k = top_k or app_settings.default_top_k
		result = loaded_predictor.predict_pil(image, top_k=requested_top_k)

		logger.info(
			"prediction_completed filename=%s label=%s score=%.4f inference_ms=%.2f",
			file.filename or "<unknown>",
			result.label,
			result.score,
			result.inference_ms,
		)

		return {
			"label": result.label,
			"score": result.score,
			"top_k": [
				{"label": prediction.label, "score": prediction.score}
				for prediction in result.top_k
			],
			"inference_ms": result.inference_ms,
			"model_version": app_settings.model_version,
		}

	return app


app = create_app()


def main() -> None:
	settings = ApiSettings.from_env()
	uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
	main()
