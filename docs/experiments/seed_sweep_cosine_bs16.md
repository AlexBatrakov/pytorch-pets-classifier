# Seed sweep: cosine small-batch variant (`exp07`)

## Goal
Check whether the best single-seed ablation (`batch=16`, `lr=1.5e-4`) remains strong across multiple random seeds.

## Runs
- `runs/exp07_cosine_es_bs16_lr15e4_s42` (seed 42)
- `runs/exp10_cosine_es_bs16_lr15e4_s123` (seed 123)
- `runs/exp11_cosine_es_bs16_lr15e4_s777` (seed 777)

## Commands
```bash
source .venv/bin/activate
./scripts/run_experiment.sh configs/experiments/exp10_cosine_es_bs16_lr15e4_s123.yaml runs/exp10_cosine_es_bs16_lr15e4_s123
./scripts/run_experiment.sh configs/experiments/exp11_cosine_es_bs16_lr15e4_s777.yaml runs/exp11_cosine_es_bs16_lr15e4_s777
python scripts/seed_sweep_summary.py --runs runs/exp07_cosine_es_bs16_lr15e4_s42 runs/exp10_cosine_es_bs16_lr15e4_s123 runs/exp11_cosine_es_bs16_lr15e4_s777
```

## Results
- test_acc1 mean ± std: `0.8600 ± 0.0165`
- test_acc5 mean ± std: `0.9848 ± 0.0032`
- test_loss mean ± std: `0.4641 ± 0.0270`

## Notes
- Per-run:
  - `exp07 (seed 42)`: `test_acc1=0.8771`, `test_loss=0.4405`
  - `exp10 (seed 123)`: `test_acc1=0.8588`, `test_loss=0.4583`
  - `exp11 (seed 777)`: `test_acc1=0.8441`, `test_loss=0.4936`
- Compared to the default cosine winner (`exp02` seed sweep):
  - better peak single-seed score (`0.8771` vs `0.8752`)
  - worse mean `test_acc1` and worse stability (higher std)
- Decision: keep `exp02` as the main showcase configuration.
