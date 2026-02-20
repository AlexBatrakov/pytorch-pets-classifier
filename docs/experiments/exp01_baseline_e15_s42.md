# Experiment: `exp01_baseline_e15_s42`

## Goal
Baseline isolated run with fixed seed and 15 epochs.

## Config
- Path: `configs/experiments/exp01_baseline_e15_s42.yaml`
- Scheduler: `none`
- Early stopping: `disabled`

## Commands
```bash
source .venv/bin/activate
./scripts/run_experiment.sh configs/experiments/exp01_baseline_e15_s42.yaml runs/exp01_baseline_e15_s42
```

If you intentionally rerun into the same folder:

```bash
./scripts/run_experiment.sh --force configs/experiments/exp01_baseline_e15_s42.yaml runs/exp01_baseline_e15_s42
```

## Outputs
- Checkpoint: `runs/exp01_baseline_e15_s42/checkpoints/best.pt`
- Metrics CSV: `runs/exp01_baseline_e15_s42/artifacts/metrics.csv`
- Curves: `runs/exp01_baseline_e15_s42/assets/training_curves.png`
- Confusion matrix: `runs/exp01_baseline_e15_s42/assets/confusion_matrix.png`

## Results
- Best epoch: 7
- Val: `loss 0.4513 | acc@1 0.863 | acc@5 0.986`
- Test: `loss 0.5881 | acc@1 0.817 | acc@5 0.974`

## Notes
- 15 epochs were fully trained without early stopping.
- Validation quality peaks around epoch 7, then mostly oscillates/degrades.
