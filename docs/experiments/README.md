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
