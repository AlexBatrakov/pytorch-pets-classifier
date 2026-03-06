FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PORT=8080 \
	DEVICE=cpu \
	MODEL_VERSION=exp17_cosine_es_img256_wd1e3_s42 \
	MODEL_PATH=/app/models/exp17_best.pt

WORKDIR /app

COPY requirements.txt pyproject.toml ./

# Use CPU-only PyTorch wheels for Azure Container Apps linux/amd64 runtime.
RUN pip install --no-cache-dir --upgrade pip \
	&& grep -vE '^(torch|torchvision)(==.*)?$' requirements.txt > /tmp/requirements.runtime.txt \
	&& pip install --no-cache-dir -r /tmp/requirements.runtime.txt \
	&& pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0

COPY src ./src
COPY runs/exp17_cosine_es_img256_wd1e3_s42/checkpoints/best.pt ./models/exp17_best.pt

EXPOSE 8080

CMD ["python", "-m", "src.api"]
