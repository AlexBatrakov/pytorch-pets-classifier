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
- **Chosen model:** `exp17_cosine_es_img256_wd1e3_s42` (ResNet18 + cosine + early stopping + `img256` + `wd=1e-3`)
- **Why this one:** won a controlled Group A-short screening (resolution + `weight_decay`) and then improved the **3-seed mean ± std** versus the previous showcase recipe

### Metrics (best checkpoint)

| Split | loss | acc@1 | acc@5 |
| --- | --- | --- | --- |
| Val | 0.2918 | 0.925 | 0.992 |
| Test | 0.4292 | 0.881 | 0.985 |

### Robustness (seeds `42/123/777`) for showcase setup

| Metric | Mean ± std |
| --- | --- |
| test_acc1 | `0.8829 ± 0.0015` |
| test_acc5 | `0.9854 ± 0.0014` |
| test_loss | `0.4159 ± 0.0158` |

### Key takeaways
- A controlled Group A-short cycle (resolution -> `weight_decay` -> one augmentation test) improved the showcase recipe **without changing the backbone**.
- `img256 + wd=1e-3` improved both mean `test_acc1` and run-to-run stability vs the previous cosine showcase setup.
- A mild `ColorJitter` test was a useful negative result (it hurt accuracy), which helped avoid blind augmentation tuning.
- Error analysis still indicates the main weakness is **fine-grained breed separation**, not complete feature failure.

### Showcase visuals

![Training curves](assets/training_curves_showcase.png)
![Error analysis: top confusion pairs](docs/experiments/assets/exp17_error_confusion_top_pairs.png)

## How Final Model Was Chosen

Selection rules used in this project:
- **Within one run:** save `best.pt` by highest `val_acc1` (early stopping also monitors `val_acc1`)
- **Across configs:** compare candidates on the same train/val/test split and log results in `docs/experiments/README.md`
- **Do not trust single-seed peaks:** rerun close contenders with seeds `42/123/777`
- **Prefer robustness over cherry-picking:** compare `mean ± std`, not only the best number
- **Use test metrics transparently:** reported for comparison and final justification

Final decision:
- `exp07` (`batch=16`) achieved a strong **single-seed** result, but was less stable across seeds
- Group A-short then tested a controlled set of low-risk recipe changes on `ResNet18`:
  - resolution sweep (`224/256/320`)
  - `weight_decay` sweep on the best resolution
  - one augmentation recipe (`ColorJitter`, rejected)
- `exp17` (`img256 + wd=1e-3`) became the best screening candidate and was confirmed on seeds `42/123/777`
- the `exp17` recipe improved mean `test_acc1` and reduced variance vs the previous `exp02` showcase recipe
- therefore the `exp17` recipe is the current showcase model

Details:
- Experiment index: [docs/experiments/README.md](docs/experiments/README.md)
- Group A-short screening summary: [docs/experiments/group_a_short_resolution_wd_aug.md](docs/experiments/group_a_short_resolution_wd_aug.md)
- Current showcase seed sweep (`exp17` recipe): [docs/experiments/seed_sweep_img256_wd1e3.md](docs/experiments/seed_sweep_img256_wd1e3.md)
- Previous cosine winner seed sweep (`exp02` recipe): [docs/experiments/seed_sweep_cosine.md](docs/experiments/seed_sweep_cosine.md)
- Small-batch seed sweep (`exp07`): [docs/experiments/seed_sweep_cosine_bs16.md](docs/experiments/seed_sweep_cosine_bs16.md)

## Error Analysis (Showcase Model `exp17`)

Detailed report:
- [docs/experiments/error_analysis_exp17.md](docs/experiments/error_analysis_exp17.md)

Key findings from the `test` split:
- Errors are still concentrated in visually similar breeds (`American Pit Bull Terrier`, `Staffordshire Bull Terrier`, `Ragdoll`, `Basset Hound`, `Birman`)
- `436` top-1 errors were made, but `381` (`87.4%`) still contain the true class in top-5
- Overconfident mistakes still exist, but their frequency decreased vs `exp02` (for example `>= 0.90` confidence errors: `17.9%` vs `20.4%`)
- Main weakness remains fine-grained class ranking/separation, especially in a few hard breed clusters

Historical reference:
- Previous showcase error analysis (`exp02`): [docs/experiments/error_analysis_exp02.md](docs/experiments/error_analysis_exp02.md)

![Overconfident mistakes gallery](docs/experiments/assets/exp17_error_gallery.png)

## What I Would Do Next (DS Perspective)

These are the next steps I would prioritize **after completing Group A-short and refreshing error analysis for `exp17`**:

- **Confidence calibration / uncertainty analysis (Group C)**  
  The model still makes overconfident mistakes, so adding reliability diagrams and temperature scaling would deepen the portfolio story beyond raw accuracy.

- **Error-delta comparison (`exp17` vs `exp02`) with a short write-up**  
  The new analysis shows uneven gains across classes (for example improvements on `Birman` / `Staffordshire Bull Terrier`, but regression on `American Pit Bull Terrier`), which is useful material for a strong DS narrative.

- **Targeted augmentation for fine-grained confusions (later Group A / B bridge)**  
  The `ColorJitter` negative result suggests we should be more selective and error-driven (for example crop/scale robustness for specific confusion pairs) rather than broadly increasing augmentation strength.

- **Stronger backbone in a controlled comparison**  
  Since top-5 is high while top-1 still fails on near-neighbor classes, I would run a controlled comparison with a stronger backbone (for example `ResNet34` / `EfficientNet`) using the same evaluation protocol and seed sweep.

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

## Reproduce Showcase Experiment (`exp17`)

Use isolated run folders to avoid overwriting `best.pt`, `last.pt`, and `metrics.csv`.

```bash
source .venv/bin/activate
./scripts/run_experiment.sh configs/experiments/exp17_cosine_es_img256_wd1e3_s42.yaml runs/exp17_cosine_es_img256_wd1e3_s42
```

Notes:
- `./scripts/run_experiment.sh` refuses to overwrite an existing run directory by default
- use `--force` only when intentionally rerunning into the same `run_dir`
- the script trains, evaluates on `val` and `test`, exports confusion matrix, and builds training curves

Related docs:
- Showcase experiment page: [docs/experiments/exp17_cosine_es_img256_wd1e3_s42.md](docs/experiments/exp17_cosine_es_img256_wd1e3_s42.md)
- Group A-short summary: [docs/experiments/group_a_short_resolution_wd_aug.md](docs/experiments/group_a_short_resolution_wd_aug.md)
- Full experiment catalog: [docs/experiments/README.md](docs/experiments/README.md)

## Experiment Navigation

Start here:
- [Experiment index and comparison table](docs/experiments/README.md)
- [Showcase experiment (`exp17`)](docs/experiments/exp17_cosine_es_img256_wd1e3_s42.md)
- [Error analysis report (`exp17`, current showcase)](docs/experiments/error_analysis_exp17.md)
- [Group A-short summary (resolution / weight decay / augmentation)](docs/experiments/group_a_short_resolution_wd_aug.md)
- [Seed sweep for current showcase recipe (`exp17`, seeds `42/123/777`)](docs/experiments/seed_sweep_img256_wd1e3.md)
- [Error analysis report (`exp02`, historical reference)](docs/experiments/error_analysis_exp02.md)

Supporting runs:
- [Baseline (`exp01`)](docs/experiments/exp01_baseline_e15_s42.md)
- [Scheduler sweep pages / configs summary](docs/experiments/README.md)
- [Previous cosine showcase seed sweep (`exp02`, seeds `42/123/777`)](docs/experiments/seed_sweep_cosine.md)
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
