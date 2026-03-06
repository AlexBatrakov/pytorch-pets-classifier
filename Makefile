SHELL := /bin/bash

.DEFAULT_GOAL := help

ifneq ("$(wildcard .env)","")
include .env
endif

PYTHON ?= .venv/bin/python
PIP ?= $(PYTHON) -m pip
MODEL_PATH ?= runs/exp17_cosine_es_img256_wd1e3_s42/checkpoints/best.pt
CKPT ?= $(MODEL_PATH)
MODEL_VERSION ?= exp17_cosine_es_img256_wd1e3_s42
DEVICE ?= cpu
PORT ?= 8080
HOST ?= 127.0.0.1
DOCKER_API_HOST ?= 127.0.0.1
SPLIT ?= test
DEFAULT_TOP_K ?= 5
TOP_K ?= 3
CONFIG ?= configs/default.yaml
SHOWCASE_CONFIG ?= configs/experiments/exp17_cosine_es_img256_wd1e3_s42.yaml
RUN_DIR ?= runs/exp17_cosine_es_img256_wd1e3_s42
DEFAULT_SAMPLE_IMAGE_PATH ?= /tmp/pytorch_pets_makefile_smoke.png
IMAGE_PATH ?= $(DEFAULT_SAMPLE_IMAGE_PATH)
IMAGE_NAME ?= pytorch-pets-api:local
CONTAINER_NAME ?= pytorch-pets-api-local
DOCKER_PORT ?= 18080
LIVE_API_URL ?= https://petsdsdemo-api.salmondune-59471bd6.germanywestcentral.azurecontainerapps.io

export MODEL_PATH CKPT MODEL_VERSION DEVICE PORT HOST DOCKER_API_HOST SPLIT DEFAULT_TOP_K TOP_K
export CONFIG SHOWCASE_CONFIG RUN_DIR IMAGE_PATH IMAGE_NAME CONTAINER_NAME
export DEFAULT_SAMPLE_IMAGE_PATH DOCKER_PORT LIVE_API_URL

.PHONY: help doctor setup setup-dev test test-api train eval predict run-exp17 run-exp17-force
.PHONY: sample-image ensure-image serve health-local predict-local
.PHONY: docker-build docker-run docker-stop docker-smoke docker-build-amd64
.PHONY: live-health live-predict

##@ Base / Meta
help: ## Show available Make targets
	@awk 'BEGIN {FS = ":.*## "}; \
		/^##@/ {if (seen) printf "\n"; printf "%s\n", substr($$0, 5); seen=1; next} \
		/^[a-zA-Z0-9_.-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

doctor: ## Check local prerequisites and useful defaults
	@if [ -x "$(PYTHON)" ]; then echo "[ok] Python executable: $(PYTHON)"; else echo "[error] Missing Python executable: $(PYTHON)"; exit 1; fi
	@if [ -d ".venv" ]; then echo "[ok] Virtualenv directory: .venv"; else echo "[warn] .venv directory not found"; fi
	@if [ -f "$(MODEL_PATH)" ]; then echo "[ok] Model checkpoint: $(MODEL_PATH)"; else echo "[warn] Model checkpoint not found: $(MODEL_PATH)"; fi
	@if command -v docker >/dev/null 2>&1; then echo "[ok] Docker available: $$(command -v docker)"; else echo "[warn] Docker not found"; fi
	@if command -v az >/dev/null 2>&1; then echo "[ok] Azure CLI available: $$(command -v az)"; else echo "[info] Azure CLI not found (optional for local work)"; fi
	@if [ -f ".env" ]; then echo "[ok] Loaded .env overrides"; else echo "[info] .env not found; using Makefile defaults / .env.example reference"; fi
	@echo "[info] Local API address: http://$(HOST):$(PORT)"
	@echo "[info] Live API URL: $(LIVE_API_URL)"

##@ Setup / Test
setup: ## Install runtime dependencies into the configured virtualenv
	$(PIP) install -r requirements.txt

setup-dev: ## Install runtime and development dependencies into the virtualenv
	$(PIP) install -r requirements-dev.txt

test: ## Run the full pytest suite
	$(PYTHON) -m pytest -q

test-api: ## Run the API-focused pytest subset
	$(PYTHON) -m pytest tests/test_api.py -q

##@ Core ML Workflow
train: ## Run training with CONFIG=<path>
	$(PYTHON) -m src.train --config "$(CONFIG)"

eval: ## Evaluate CKPT=<path> on SPLIT=val|test
	$(PYTHON) -m src.eval --ckpt "$(CKPT)" --split "$(SPLIT)"

predict: ensure-image ## Run single-image prediction with CKPT=<path> IMAGE_PATH=<path>
	$(PYTHON) -m src.predict --ckpt "$(CKPT)" --image "$(IMAGE_PATH)" --top-k "$(TOP_K)"

run-exp17: ## Reproduce the showcase experiment in RUN_DIR=<path>
	@if [ -f "$(RUN_DIR)/artifacts/metrics.csv" ]; then \
		echo "[info] Showcase run already exists at $(RUN_DIR)"; \
		echo "[info] Use 'make run-exp17-force' to rerun into the same directory."; \
	else \
		./scripts/run_experiment.sh "$(SHOWCASE_CONFIG)" "$(RUN_DIR)"; \
	fi

run-exp17-force: ## Force rerun the showcase experiment into RUN_DIR=<path>
	./scripts/run_experiment.sh --force "$(SHOWCASE_CONFIG)" "$(RUN_DIR)"

##@ Service / API
sample-image: ## Generate a deterministic smoke-test image at IMAGE_PATH=<path>
	@mkdir -p "$(dir $(IMAGE_PATH))"
	$(PYTHON) -c "from PIL import Image; Image.new('RGB', (256, 256), color=(128, 96, 160)).save('$(IMAGE_PATH)')"
	@echo "[ok] Wrote sample image: $(IMAGE_PATH)"

ensure-image: ## Ensure IMAGE_PATH exists; auto-create the default smoke image if needed
	@if [ -f "$(IMAGE_PATH)" ]; then \
		echo "[ok] Using image: $(IMAGE_PATH)"; \
	elif [ "$(IMAGE_PATH)" = "$(DEFAULT_SAMPLE_IMAGE_PATH)" ]; then \
		$(MAKE) --no-print-directory sample-image IMAGE_PATH="$(IMAGE_PATH)"; \
	else \
		echo "[error] Image not found: $(IMAGE_PATH)"; \
		echo "[info] Either provide IMAGE_PATH=<real-file> or use the default smoke image path."; \
		exit 1; \
	fi

serve: ## Start the local HTTP API in the foreground with the configured MODEL_PATH and PORT
	MODEL_PATH="$(MODEL_PATH)" MODEL_VERSION="$(MODEL_VERSION)" DEVICE="$(DEVICE)" PORT="$(PORT)" DEFAULT_TOP_K="$(DEFAULT_TOP_K)" $(PYTHON) -m src.api

health-local: ## Call the local API health endpoint
	curl -sS "http://$(HOST):$(PORT)/health"

predict-local: ensure-image ## Call the local API prediction endpoint
	curl -sS -X POST "http://$(HOST):$(PORT)/predict?top_k=$(TOP_K)" \
		-F "file=@$(IMAGE_PATH);type=image/png"

##@ Docker
docker-build: ## Build the local Docker image
	docker build -t "$(IMAGE_NAME)" .

docker-run: ## Start the local Docker container on DOCKER_PORT=<port>
	docker run --rm -d -p "$(DOCKER_PORT):8080" --name "$(CONTAINER_NAME)" "$(IMAGE_NAME)"

docker-stop: ## Stop the local Docker container if it is running
	@if docker ps -q -f name="^/$(CONTAINER_NAME)$$" | grep -q .; then \
		docker stop "$(CONTAINER_NAME)"; \
	else \
		echo "[info] Container not running: $(CONTAINER_NAME)"; \
	fi

docker-smoke: ensure-image ## Call the Dockerized API health and prediction endpoints
	@curl -sS "http://$(DOCKER_API_HOST):$(DOCKER_PORT)/health"
	@echo
	@curl -sS -X POST "http://$(DOCKER_API_HOST):$(DOCKER_PORT)/predict?top_k=$(TOP_K)" \
		-F "file=@$(IMAGE_PATH);type=image/png"
	@echo

docker-build-amd64: ## Build an Azure-compatible linux/amd64 image locally
	docker buildx build --platform linux/amd64 -t "$(IMAGE_NAME)-amd64" .

##@ Live Demo
live-health: ## Call the public Azure health endpoint
	curl -sS "$(LIVE_API_URL)/health"

live-predict: ensure-image ## Call the public Azure prediction endpoint
	curl -sS -X POST "$(LIVE_API_URL)/predict?top_k=$(TOP_K)" \
		-F "file=@$(IMAGE_PATH);type=image/png"
