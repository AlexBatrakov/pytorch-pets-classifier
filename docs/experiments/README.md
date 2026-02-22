# Experiments Index

This folder contains the experiment log for the project: individual run pages, seed sweeps, and error analysis for the showcase model.

Use this page as the main navigation hub for all experiment-related materials.

## Start Here

- [Showcase experiment (`exp02`: cosine + early stopping)](exp02_cosine_es_e30_s42.md)
- [Error analysis for showcase model (`exp02`)](error_analysis_exp02.md)
- [Seed sweep for showcase setup (`exp02`, seeds `42/123/777`)](seed_sweep_cosine.md)
- [Seed sweep for best single-seed small-batch variant (`exp07`)](seed_sweep_cosine_bs16.md)

## Experiment Pages

Core runs:
- [exp01: baseline, 15 epochs, seed 42](exp01_baseline_e15_s42.md)
- [exp02: cosine + early stopping, 30 epochs, seed 42 (showcase)](exp02_cosine_es_e30_s42.md)
- [exp03: plateau scheduler + early stopping, seed 42](exp03_plateau_es_e30_s42.md)
- [exp03b: plateau scheduler without early stopping (control), seed 42](exp03b_plateau_noes_e30_s42.md)
- [exp04: step scheduler + early stopping, seed 42](exp04_step_es_e30_s42.md)

Ablations:
- [exp07: cosine + ES, `batch=16`, `lr=1.5e-4`, seed 42](exp07_cosine_es_bs16_lr15e4_s42.md)
- [exp08: cosine + ES, `batch=64`, `lr=6e-4`, seed 42](exp08_cosine_es_bs64_lr6e4_s42.md)
- [exp09: cosine + ES, `freeze_backbone=true`, `freeze_epochs=2`, seed 42](exp09_cosine_es_freeze2_s42.md)

Seed-sweep support runs (used by sweep pages):
- `exp05`, `exp06` for `exp02` cosine robustness
- `exp10`, `exp11` for `exp07` small-batch robustness

## Reproduction Workflow (short)

- Runner script: `scripts/run_experiment.sh`
- It creates isolated artifacts per run (`checkpoints`, `metrics.csv`, plots)
- It refuses to overwrite a run directory unless `--force` is used

Typical pattern:

```bash
./scripts/run_experiment.sh configs/experiments/<experiment>.yaml runs/<run_name>
```

Sweep summary:

```bash
python scripts/seed_sweep_summary.py --runs runs/<run1> runs/<run2> runs/<run3>
```

## Comparison Table

| Experiment | Config | Best epoch | Stopped at epoch | Val (loss / acc@1 / acc@5) | Test (loss / acc@1 / acc@5) |
| --- | --- | --- | --- | --- | --- |
| `exp01_baseline_e15_s42` | `configs/experiments/exp01_baseline_e15_s42.yaml` | 7 | 15 | `0.4513 / 0.863 / 0.986` | `0.5881 / 0.817 / 0.974` |
| `exp02_cosine_es_e30_s42` | `configs/experiments/exp02_cosine_es_e30_s42.yaml` | 23 | 29 (early stop) | `0.2890 / 0.914 / 0.988` | `0.4570 / 0.875 / 0.984` |
| `exp03_plateau_es_e30_s42` | `configs/experiments/exp03_plateau_es_e30_s42.yaml` | 12 | 18 (early stop) | `0.3341 / 0.906 / 0.989` | `0.4906 / 0.852 / 0.982` |
| `exp03b_plateau_noes_e30_s42` | `configs/experiments/exp03b_plateau_noes_e30_s42.yaml` | 21 | 30 | `0.3322 / 0.908 / 0.989` | `0.4864 / 0.859 / 0.981` |
| `exp04_step_es_e30_s42` | `configs/experiments/exp04_step_es_e30_s42.yaml` | 13 | 19 (early stop) | `0.3416 / 0.909 / 0.989` | `0.4977 / 0.851 / 0.982` |
| `exp07_cosine_es_bs16_lr15e4_s42` | `configs/experiments/exp07_cosine_es_bs16_lr15e4_s42.yaml` | 17 | 23 (early stop) | `0.2934 / 0.910 / 0.990` | `0.4405 / 0.877 / 0.983` |
| `exp08_cosine_es_bs64_lr6e4_s42` | `configs/experiments/exp08_cosine_es_bs64_lr6e4_s42.yaml` | 22 | 28 (early stop) | `0.3335 / 0.902 / 0.993` | `0.5044 / 0.857 / 0.978` |
| `exp09_cosine_es_freeze2_s42` | `configs/experiments/exp09_cosine_es_freeze2_s42.yaml` | 11 | 17 (early stop) | `0.4156 / 0.886 / 0.985` | `0.5968 / 0.826 / 0.974` |
| `exp10_cosine_es_bs16_lr15e4_s123` | `configs/experiments/exp10_cosine_es_bs16_lr15e4_s123.yaml` | 9 | 15 (early stop) | `0.3347 / 0.912 / 0.995` | `0.4583 / 0.859 / 0.989` |
| `exp11_cosine_es_bs16_lr15e4_s777` | `configs/experiments/exp11_cosine_es_bs16_lr15e4_s777.yaml` | 5 | 11 (early stop) | `0.3430 / 0.909 / 0.993` | `0.4936 / 0.844 / 0.983` |

## Selection Policy (Project Showcase)

- Run-level checkpoint selection uses `val_acc1` (`best.pt`)
- Candidate comparison uses the table above plus targeted seed sweeps for close contenders
- Final showcase choice prioritizes robustness (`mean ± std` across seeds) over a better single-seed peak

## Current Winner

- Scheduler winner: `cosine` (experiment `exp02_cosine_es_e30_s42`)
- Plateau and step improve over the baseline but do not beat cosine on test metrics
- Seed robustness (`42/123/777`) for cosine winner:
  - `test_acc1 = 0.8717 ± 0.0059`
  - `test_acc5 = 0.9830 ± 0.0014`
  - `test_loss = 0.4551 ± 0.0160`
- Best single-seed run so far: `exp07` (`batch=16`, `lr=1.5e-4`) with `test_acc1=0.877`
- Robustness check for `exp07` (`batch=16`, seeds `42/123/777`):
  - `test_acc1 = 0.8600 ± 0.0165`
  - `test_acc5 = 0.9848 ± 0.0032`
  - `test_loss = 0.4641 ± 0.0270`
- Final decision: keep `exp02` as showcase winner (better mean quality and lower variance)
