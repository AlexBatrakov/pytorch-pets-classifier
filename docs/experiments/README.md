# Experiments Log

This folder keeps compact experiment records that are linked from the main README.
Use one file per experiment when you want to keep details (settings, metrics, notes, artifacts).

## Suggested format

| Field | Example |
| --- | --- |
| Name | `baseline_15_epochs` |
| Config | `configs/default.yaml` |
| Command | `python -m src.train --config configs/default.yaml --epochs 15` |
| Best epoch | `7` |
| Val metrics | `loss 0.4513, acc@1 0.863, acc@5 0.986` |
| Test metrics | `loss 0.5881, acc@1 0.817, acc@5 0.974` |
| Artifacts | `assets/confusion_matrix.png`, `assets/training_curves.png` |

## Current experiments

- Baseline 15 epochs: summarized in main README results section.
- Cosine + early stopping template config: `configs/cosine_earlystop.yaml`.
- Isolated baseline run template: `docs/experiments/exp01_baseline_e15_s42.md`.
- Runner script: `scripts/run_experiment.sh`.
- Isolated cosine+ES run template: `docs/experiments/exp02_cosine_es_e30_s42.md`.

## Comparison table

| Experiment | Config | Best epoch | Stopped at epoch | Val (loss / acc@1 / acc@5) | Test (loss / acc@1 / acc@5) |
| --- | --- | --- | --- | --- | --- |
| `exp01_baseline_e15_s42` | `configs/experiments/exp01_baseline_e15_s42.yaml` | 7 | 15 | `0.4513 / 0.863 / 0.986` | `0.5881 / 0.817 / 0.974` |
| `exp02_cosine_es_e30_s42` | `configs/experiments/exp02_cosine_es_e30_s42.yaml` | 23 | 29 (early stop) | `0.2890 / 0.914 / 0.988` | `0.4570 / 0.875 / 0.984` |
