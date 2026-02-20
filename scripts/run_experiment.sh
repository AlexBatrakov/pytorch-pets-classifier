#!/usr/bin/env bash
set -euo pipefail

FORCE_OVERWRITE=false
if [[ "${1:-}" == "--force" ]]; then
  FORCE_OVERWRITE=true
  shift
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [--force] <config_path> <run_dir>"
  echo "Example: $0 configs/experiments/exp01_baseline_e15_s42.yaml runs/exp01_baseline_e15_s42"
  exit 1
fi

CONFIG_PATH="$1"
RUN_DIR="$2"
CKPT_PATH="${RUN_DIR}/checkpoints/best.pt"
METRICS_PATH="${RUN_DIR}/artifacts/metrics.csv"
ASSETS_DIR="${RUN_DIR}/assets"

if [[ -f "${METRICS_PATH}" && "${FORCE_OVERWRITE}" != "true" ]]; then
  echo "Run directory already has metrics: ${METRICS_PATH}"
  echo "Refusing to overwrite existing experiment."
  echo "Use a new run_dir or rerun with --force."
  exit 2
fi

mkdir -p "${ASSETS_DIR}"

python -m src.train --config "${CONFIG_PATH}"
python -m src.eval --ckpt "${CKPT_PATH}" --split val --json-out "${RUN_DIR}/artifacts/val_metrics.json"
python -m src.eval --ckpt "${CKPT_PATH}" --split test --json-out "${RUN_DIR}/artifacts/test_metrics.json" --cm-out "${ASSETS_DIR}/confusion_matrix.png" --cm-normalize
python -m src.plot_metrics --metrics "${METRICS_PATH}" --out "${ASSETS_DIR}/training_curves.png"

echo "Done. Run artifacts:"
echo "  checkpoint: ${CKPT_PATH}"
echo "  metrics:    ${METRICS_PATH}"
echo "  assets:     ${ASSETS_DIR}"
