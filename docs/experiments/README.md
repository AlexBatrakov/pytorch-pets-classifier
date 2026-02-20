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
- Isolated plateau+ES run template: `docs/experiments/exp03_plateau_es_e30_s42.md`.
- Isolated plateau-noES control run template: `docs/experiments/exp03b_plateau_noes_e30_s42.md`.
- Isolated step+ES run template: `docs/experiments/exp04_step_es_e30_s42.md`.
- Seed robustness page for cosine winner: `docs/experiments/seed_sweep_cosine.md`.
- Each experiment page includes embedded training curves and confusion matrix images.

## Comparison table

| Experiment | Config | Best epoch | Stopped at epoch | Val (loss / acc@1 / acc@5) | Test (loss / acc@1 / acc@5) |
| --- | --- | --- | --- | --- | --- |
| `exp01_baseline_e15_s42` | `configs/experiments/exp01_baseline_e15_s42.yaml` | 7 | 15 | `0.4513 / 0.863 / 0.986` | `0.5881 / 0.817 / 0.974` |
| `exp02_cosine_es_e30_s42` | `configs/experiments/exp02_cosine_es_e30_s42.yaml` | 23 | 29 (early stop) | `0.2890 / 0.914 / 0.988` | `0.4570 / 0.875 / 0.984` |
| `exp03_plateau_es_e30_s42` | `configs/experiments/exp03_plateau_es_e30_s42.yaml` | 12 | 18 (early stop) | `0.3341 / 0.906 / 0.989` | `0.4906 / 0.852 / 0.982` |
| `exp03b_plateau_noes_e30_s42` | `configs/experiments/exp03b_plateau_noes_e30_s42.yaml` | 21 | 30 | `0.3322 / 0.908 / 0.989` | `0.4864 / 0.859 / 0.981` |
| `exp04_step_es_e30_s42` | `configs/experiments/exp04_step_es_e30_s42.yaml` | 13 | 19 (early stop) | `0.3416 / 0.909 / 0.989` | `0.4977 / 0.851 / 0.982` |

## Current winner

- Scheduler winner: `cosine` (experiment `exp02_cosine_es_e30_s42`).
- Plateau and step improve over baseline but do not beat cosine on test metrics.
- Seed robustness (`42/123/777`) for cosine winner:
  - `test_acc1 = 0.8717 ± 0.0059`
  - `test_acc5 = 0.9830 ± 0.0014`
  - `test_loss = 0.4551 ± 0.0160`
