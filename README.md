# PyTorch Pets Classifier
[![CI](https://github.com/AlexBatrakov/pytorch-pets-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexBatrakov/pytorch-pets-classifier/actions/workflows/ci.yml)

Minimal, production-style baseline for multi-class image classification on the Oxford-IIIT Pets dataset (37 breeds). Uses transfer learning with torchvision ResNet18 ImageNet weights and runs on macOS MPS or CPU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install development dependencies (tests):

```bash
pip install -r requirements-dev.txt
```

## Testing

```bash
python -m pytest -q
```

Test scope:
- Unit: config overrides, metrics math, device selection, best-checkpoint selection logic.
- Integration: metrics CSV write path and training-curves plotting from CSV.
- Smoke: imports and basic model/dataset construction.

Experiment logs:
- Main summary stays in this README.
- Detailed run notes go to `docs/experiments/README.md`.

## Train

```bash
python -m src.train --config configs/default.yaml
```

Run with cosine scheduler + early stopping preset:

```bash
python -m src.train --config configs/cosine_earlystop.yaml
```

## Experiments

Use isolated run folders to avoid overwriting `best.pt`, `last.pt`, and `metrics.csv` between experiments.

Run the first isolated baseline experiment:

```bash
source .venv/bin/activate
./scripts/run_experiment.sh configs/experiments/exp01_baseline_e15_s42.yaml runs/exp01_baseline_e15_s42
```

The script refuses to overwrite an existing run folder by default.
Use `--force` only when you intentionally want to rerun into the same `run_dir`.

Run the second experiment (cosine scheduler + early stopping):

```bash
./scripts/run_experiment.sh configs/experiments/exp02_cosine_es_e30_s42.yaml runs/exp02_cosine_es_e30_s42
```

Run the third experiment (plateau scheduler + early stopping):

```bash
./scripts/run_experiment.sh configs/experiments/exp03_plateau_es_e30_s42.yaml runs/exp03_plateau_es_e30_s42
```

Run the fourth experiment (step scheduler + early stopping):

```bash
./scripts/run_experiment.sh configs/experiments/exp04_step_es_e30_s42.yaml runs/exp04_step_es_e30_s42
```

Seed sweep for the current winner (`cosine + early stopping`):

```bash
./scripts/run_experiment.sh configs/experiments/exp05_cosine_es_e30_s123.yaml runs/exp05_cosine_es_e30_s123
./scripts/run_experiment.sh configs/experiments/exp06_cosine_es_e30_s777.yaml runs/exp06_cosine_es_e30_s777
python scripts/seed_sweep_summary.py --runs runs/exp02_cosine_es_e30_s42 runs/exp05_cosine_es_e30_s123 runs/exp06_cosine_es_e30_s777
```

Batch/freeze sweep for the current winner:

```bash
./scripts/run_experiment.sh configs/experiments/exp07_cosine_es_bs16_lr15e4_s42.yaml runs/exp07_cosine_es_bs16_lr15e4_s42
./scripts/run_experiment.sh configs/experiments/exp08_cosine_es_bs64_lr6e4_s42.yaml runs/exp08_cosine_es_bs64_lr6e4_s42
./scripts/run_experiment.sh configs/experiments/exp09_cosine_es_freeze2_s42.yaml runs/exp09_cosine_es_freeze2_s42
```

Seed sweep for the best single-seed ablation (`exp07`, smaller batch):

```bash
./scripts/run_experiment.sh configs/experiments/exp10_cosine_es_bs16_lr15e4_s123.yaml runs/exp10_cosine_es_bs16_lr15e4_s123
./scripts/run_experiment.sh configs/experiments/exp11_cosine_es_bs16_lr15e4_s777.yaml runs/exp11_cosine_es_bs16_lr15e4_s777
python scripts/seed_sweep_summary.py --runs runs/exp07_cosine_es_bs16_lr15e4_s42 runs/exp10_cosine_es_bs16_lr15e4_s123 runs/exp11_cosine_es_bs16_lr15e4_s777
```

Recommended workflow:
- Keep one high-level summary in this README.
- Keep detailed per-run notes in `docs/experiments/*.md`.

Common overrides:

```bash
python -m src.train --epochs 10 --batch-size 64 --lr 3e-4 --freeze-epochs 2 --num-workers 0
```

Best checkpoint is saved to `./checkpoints/best.pt`.
Per-epoch metrics are saved to `./artifacts/metrics.csv`.
Each checkpoint also stores run metadata (`git_commit`, `created_at_utc`, `device`, `torch_version`, parameter counts, and epoch metrics).

Example `metrics.csv`:

```csv
epoch,train_loss,train_acc1,train_acc5,val_loss,val_acc1,val_acc5,lr
1,1.665000,0.566000,0.839000,0.930600,0.720000,0.965000,0.00030000
2,0.890800,0.749000,0.945000,0.516700,0.827000,0.990000,0.00030000
```

Build training curves from the metrics file:

```bash
python -m src.plot_metrics --metrics artifacts/metrics.csv --out assets/training_curves.png
```

## Evaluate

```bash
python -m src.eval --ckpt checkpoints/best.pt --split val
```

Evaluate on the official test split:

```bash
python -m src.eval --ckpt checkpoints/best.pt --split test
```

Save a confusion matrix image:

```bash
python -m src.eval --ckpt checkpoints/best.pt --split test --cm-out assets/confusion_matrix.png --cm-normalize
```

## Results

### Current best experiment (`exp02_cosine_es_e30_s42`)

Training command:

```bash
./scripts/run_experiment.sh --force configs/experiments/exp02_cosine_es_e30_s42.yaml runs/exp02_cosine_es_e30_s42
```

| Parameter | Value |
| --- | --- |
| Model | ResNet18 (ImageNet pretrained) |
| Dataset | Oxford-IIIT Pets (37 classes) |
| Optimizer | AdamW |
| Initial LR | 3e-4 |
| Scheduler | cosine (`t_max=30`) |
| Early stopping | `monitor=val_acc1`, `patience=6`, `min_delta=0.001` |
| Best epoch | 23 |
| Stopped epoch | 29 (early stop) |

### Metrics (best checkpoint)

| Split | loss | acc@1 | acc@5 |
| --- | --- | --- | --- |
| Val | 0.2890 | 0.914 | 0.988 |
| Test | 0.4570 | 0.875 | 0.984 |

### Improvement vs baseline (`exp01_baseline_e15_s42`)

| Metric | Baseline | Current best | Delta |
| --- | --- | --- | --- |
| val_acc1 | 0.8628 | 0.9144 | +5.16 pp |
| test_acc1 | 0.8170 | 0.8750 | +5.80 pp |
| test_loss | 0.5881 | 0.4570 | -0.1311 |
| test_acc5 | 0.9740 | 0.9840 | +1.00 pp |

### Scheduler sweep (seed 42)

| Experiment | Scheduler | Test acc@1 | Test loss |
| --- | --- | --- | --- |
| `exp02` | cosine + ES | 0.875 | 0.4570 |
| `exp03b` | plateau (no ES) | 0.859 | 0.4864 |
| `exp04` | step + ES | 0.851 | 0.4977 |

### Seed robustness for winner (`cosine + ES`, seeds `42/123/777`)

| Metric | Mean ± std |
| --- | --- |
| test_acc1 | `0.8717 ± 0.0059` |
| test_acc5 | `0.9830 ± 0.0014` |
| test_loss | `0.4551 ± 0.0160` |

### Batch/freeze ablation (seed `42`)

| Experiment | Change vs `exp02` | Test acc@1 | Test loss |
| --- | --- | --- | --- |
| `exp07` | `batch=16`, `lr=1.5e-4` | `0.877` | `0.4405` |
| `exp08` | `batch=64`, `lr=6e-4` | `0.857` | `0.5044` |
| `exp09` | `freeze_backbone=true`, `freeze_epochs=2` | `0.826` | `0.5968` |

`exp07` is the best single-seed run so far, but `exp02` stays the default showcase because it already has multi-seed robustness results.

### Robustness check for `exp07` (`batch=16`, `lr=1.5e-4`, seeds `42/123/777`)

| Metric | Mean ± std |
| --- | --- |
| test_acc1 | `0.8600 ± 0.0165` |
| test_acc5 | `0.9848 ± 0.0032` |
| test_loss | `0.4641 ± 0.0270` |

Compared to `exp02`, this variant has a better peak single-seed score but worse average accuracy and higher variance across seeds.

![Training curves](assets/training_curves_showcase.png)
![Confusion matrix](assets/confusion_matrix_showcase.png)

### Observations

- Constant LR baseline underfits later training stages compared to cosine schedule.
- With `patience=6`, early stopping avoids stopping too early and captures late improvements.
- Smaller batch (`16`) improved single-seed quality, while larger batch (`64`) degraded quality.
- Freezing backbone for 2 epochs hurts quality in this setup.
- The `batch=16` variant is less stable across seeds than the default `batch=32` cosine setup.

## Predict

```bash
python -m src.predict --ckpt checkpoints/best.pt --image path/to/image.jpg
```

Example output:

```
Top-1: abyssinian (0.9234)
Top-5:
	abyssinian (0.9234)
	bengal (0.0345)
	siamese (0.0121)
	ragdoll (0.0098)
	birman (0.0076)
```

## macOS MPS

The code automatically selects MPS if available via `torch.backends.mps.is_available()`. If MPS is not available, it falls back to CPU.

## Repo hygiene

- Dataset downloads to ./data (not committed)
- Checkpoints saved to ./checkpoints (not committed)

## Roadmap

- Grad-CAM visualization
- AMP training
- Optuna hyperparameter search
- Weights & Biases logging
- ONNX export
