# PyTorch Pets Classifier
[![CI](https://github.com/AlexBatrakov/pytorch-pets-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexBatrakov/pytorch-pets-classifier/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Showcase **Data Science / ML** pet project for fine-grained image classification on **Oxford-IIIT Pets (37 breeds)** using **PyTorch + torchvision (ResNet18 transfer learning)**.

The goal of this repo is not only to train a model, but to demonstrate an evidence-based DS workflow:
- a clean PyTorch training/evaluation pipeline,
- experiment design, tracking, and comparison,
- robust model selection (seed sweeps, not only single-seed peaks),
- structured error analysis with plots, visual examples, and hypotheses,
- tests + CI for core pipeline behavior.

## Results Snapshot

### Final showcase model
- **Chosen model:** `exp02_cosine_es_e30_s42` (ResNet18 + cosine scheduler + early stopping)
- **Why this one:** better **3-seed mean ± std** than alternatives; selected for robustness, not cherry-picked peak score

### Metrics (best checkpoint)

| Split | loss | acc@1 | acc@5 |
| --- | --- | --- | --- |
| Val | 0.2890 | 0.914 | 0.988 |
| Test | 0.4570 | 0.875 | 0.984 |

### Robustness (seeds `42/123/777`) for showcase setup

| Metric | Mean ± std |
| --- | --- |
| test_acc1 | `0.8717 ± 0.0059` |
| test_acc5 | `0.9830 ± 0.0014` |
| test_loss | `0.4551 ± 0.0160` |

### Key takeaways
- Cosine LR + early stopping improved the baseline and generalized better on `test`.
- A smaller batch (`exp07`, `batch=16`) produced a better **single-seed** peak, but was less stable across seeds.
- Error analysis shows the main weakness is **fine-grained breed separation**, not complete feature failure.

### Showcase visuals

![Training curves](assets/training_curves_showcase.png)
![Error analysis: top confusion pairs](docs/experiments/assets/exp02_error_confusion_top_pairs.png)

## How Final Model Was Chosen

Selection rules used in this project:
- **Within one run:** save `best.pt` by highest `val_acc1` (early stopping also monitors `val_acc1`)
- **Across configs:** compare candidates on the same train/val/test split and log results in `docs/experiments/README.md`
- **Do not trust single-seed peaks:** rerun close contenders with seeds `42/123/777`
- **Prefer robustness over cherry-picking:** compare `mean ± std`, not only the best number
- **Use test metrics transparently:** reported for comparison and final justification

Final decision:
- `exp07` (`batch=16`) achieved the best **single-seed** `test_acc1` (`0.877`)
- `exp02` (`batch=32`) achieved better **3-seed average** and much lower variance
- therefore `exp02` is the main showcase model

Details:
- Experiment index: [docs/experiments/README.md](docs/experiments/README.md)
- Cosine winner seed sweep: [docs/experiments/seed_sweep_cosine.md](docs/experiments/seed_sweep_cosine.md)
- Small-batch seed sweep (`exp07`): [docs/experiments/seed_sweep_cosine_bs16.md](docs/experiments/seed_sweep_cosine_bs16.md)

## Error Analysis (Showcase Model `exp02`)

Detailed report:
- [docs/experiments/error_analysis_exp02.md](docs/experiments/error_analysis_exp02.md)

Key findings from the `test` split:
- Errors are concentrated in visually similar breeds (`American Pit Bull Terrier <-> Staffordshire Bull Terrier`, `Birman <-> Ragdoll`, `Basset Hound -> Beagle`)
- `460` top-1 errors were made, but `402` (`87.4%`) still contain the true class in top-5
- A subset of mistakes is overconfident (`20.4%` of errors have confidence `>= 0.90`)
- Main weakness is fine-grained class ranking/separation

![Overconfident mistakes gallery](docs/experiments/assets/exp02_error_gallery.png)

## What I Would Do Next (DS Perspective)

These are the next steps I would prioritize **for this project**, based on the current error analysis:

- **Targeted augmentation for fine-grained confusions**  
  The main errors are concentrated in visually similar breeds (for example `APBT <-> Staffordshire Bull Terrier`, `Birman <-> Ragdoll`), so I would test stronger crop/scale and lighting augmentations.

- **Stronger backbone in a controlled comparison**  
  Since top-5 is high while top-1 still fails on near-neighbor classes, I would run a controlled comparison with a stronger backbone (for example `ResNet34` / `EfficientNet`) using the same evaluation protocol and seed sweep.

- **Confidence calibration / uncertainty analysis**  
  The model makes overconfident mistakes, so I would add reliability diagrams and temperature scaling to improve confidence interpretability.

## Quickstart

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Development dependencies (tests):

```bash
pip install -r requirements-dev.txt
```

### Train (baseline)

```bash
python -m src.train --config configs/default.yaml
```

### Train (cosine + early stopping preset)

```bash
python -m src.train --config configs/cosine_earlystop.yaml
```

### Evaluate

```bash
python -m src.eval --ckpt checkpoints/best.pt --split val
python -m src.eval --ckpt checkpoints/best.pt --split test
```

Save confusion matrix:

```bash
python -m src.eval --ckpt checkpoints/best.pt --split test --cm-out assets/confusion_matrix.png --cm-normalize
```

### Predict one image

```bash
python -m src.predict --ckpt checkpoints/best.pt --image path/to/image.jpg
```

### Run tests

```bash
python -m pytest -q
```

Test scope:
- Unit: config overrides, metrics math, device selection, best-checkpoint logic
- Integration: metrics CSV writing, training-curves plotting
- Smoke: imports and basic model/dataset construction

## Reproduce Showcase Experiment (`exp02`)

Use isolated run folders to avoid overwriting `best.pt`, `last.pt`, and `metrics.csv`.

```bash
source .venv/bin/activate
./scripts/run_experiment.sh configs/experiments/exp02_cosine_es_e30_s42.yaml runs/exp02_cosine_es_e30_s42
```

Notes:
- `./scripts/run_experiment.sh` refuses to overwrite an existing run directory by default
- use `--force` only when intentionally rerunning into the same `run_dir`
- the script trains, evaluates on `val` and `test`, exports confusion matrix, and builds training curves

Related docs:
- Showcase experiment page: [docs/experiments/exp02_cosine_es_e30_s42.md](docs/experiments/exp02_cosine_es_e30_s42.md)
- Full experiment catalog: [docs/experiments/README.md](docs/experiments/README.md)

## Experiment Navigation

Start here:
- [Experiment index and comparison table](docs/experiments/README.md)
- [Showcase experiment (`exp02`)](docs/experiments/exp02_cosine_es_e30_s42.md)
- [Error analysis report (`exp02`)](docs/experiments/error_analysis_exp02.md)

Supporting runs:
- [Baseline (`exp01`)](docs/experiments/exp01_baseline_e15_s42.md)
- [Scheduler sweep pages / configs summary](docs/experiments/README.md)
- [Cosine seed sweep (`exp02`, seeds `42/123/777`)](docs/experiments/seed_sweep_cosine.md)
- [Small-batch seed sweep (`exp07`, seeds `42/123/777`)](docs/experiments/seed_sweep_cosine_bs16.md)

## Project Structure

```text
src/                  training/eval/inference/error-analysis scripts
tests/                unit/integration/smoke tests
configs/              default and preset configs
configs/experiments/  experiment-specific configs (exp01, exp02, ...)
scripts/              experiment runner + seed sweep summary
docs/experiments/     experiment logs, comparisons, error analysis
assets/               showcase images used in root README
```

## Implementation Notes

- Device selection: uses MPS on macOS if available, otherwise CPU/CUDA fallback
- Dataset splitting in training pipeline: `train/val` from `trainval`, plus official `test` split for final evaluation
- Checkpoints include run metadata (`git_commit`, timestamp, device, torch version, parameter counts, epoch metrics)
- Run artifacts (checkpoints/metrics/images) are kept out of git; docs assets are copied selectively for showcase pages

## Repo Hygiene

- Dataset downloads to `./data` (not committed)
- Checkpoints saved to `./checkpoints` or `./runs/.../checkpoints` (not committed)
- Experiment metrics/plots saved under `./runs/...` (not committed)

## Roadmap

### Modeling / Research

- Grad-CAM visualization
- Stronger backbone experiments (ResNet34 / EfficientNet)
- Targeted augmentations for hard breed pairs
- Confidence calibration (reliability diagram / temperature scaling)

### Training / Experimentation

- AMP training
- Optuna hyperparameter search
- Weights & Biases logging

### Deployment / Interop

- ONNX export
