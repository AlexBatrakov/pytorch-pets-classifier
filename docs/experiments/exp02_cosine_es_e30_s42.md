# Experiment: `exp02_cosine_es_e30_s42`

## Goal
Compare against baseline with cosine LR schedule and early stopping enabled.

## Config
- Path: `configs/experiments/exp02_cosine_es_e30_s42.yaml`
- Scheduler: `cosine` (`t_max=30`)
- Early stopping: `enabled` (`monitor=val_acc1`, `mode=max`, `patience=6`, `min_delta=0.001`)

## Commands
```bash
source .venv/bin/activate
./scripts/run_experiment.sh configs/experiments/exp02_cosine_es_e30_s42.yaml runs/exp02_cosine_es_e30_s42
```

If you intentionally rerun into the same folder:

```bash
./scripts/run_experiment.sh --force configs/experiments/exp02_cosine_es_e30_s42.yaml runs/exp02_cosine_es_e30_s42
```

## Outputs
- Checkpoint: `runs/exp02_cosine_es_e30_s42/checkpoints/best.pt`
- Metrics CSV: `runs/exp02_cosine_es_e30_s42/artifacts/metrics.csv`
- Curves: `runs/exp02_cosine_es_e30_s42/assets/training_curves.png`
- Confusion matrix: `runs/exp02_cosine_es_e30_s42/assets/confusion_matrix.png`

## Results
- Best epoch: 23
- Early-stopped epoch: 29
- Val: `loss 0.2890 | acc@1 0.914 | acc@5 0.988`
- Test: `loss 0.4570 | acc@1 0.875 | acc@5 0.984`

## Visuals

![Training curves (exp02)](assets/exp02_training_curves.png)
![Confusion matrix (exp02)](assets/exp02_confusion_matrix.png)

## Notes
- Compare this run with `docs/experiments/exp01_baseline_e15_s42.md`.
- Improvement vs baseline (`exp01`): `+5.16` pp on val acc@1 and `+5.8` pp on test acc@1.
