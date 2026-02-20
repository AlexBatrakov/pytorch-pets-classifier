# Seed sweep: cosine winner

## Goal
Measure run-to-run variance of the best scheduler setup (`cosine + early stopping`).

## Runs
- `runs/exp02_cosine_es_e30_s42` (seed 42)
- `runs/exp05_cosine_es_e30_s123` (seed 123)
- `runs/exp06_cosine_es_e30_s777` (seed 777)

## Commands
```bash
source .venv/bin/activate
./scripts/run_experiment.sh configs/experiments/exp05_cosine_es_e30_s123.yaml runs/exp05_cosine_es_e30_s123
./scripts/run_experiment.sh configs/experiments/exp06_cosine_es_e30_s777.yaml runs/exp06_cosine_es_e30_s777
python scripts/seed_sweep_summary.py --runs runs/exp02_cosine_es_e30_s42 runs/exp05_cosine_es_e30_s123 runs/exp06_cosine_es_e30_s777
```

## Results
- test_acc1 mean ± std: `0.8717 ± 0.0059`
- test_acc5 mean ± std: `0.9830 ± 0.0014`
- test_loss mean ± std: `0.4551 ± 0.0160`

## Notes
- Per-run:
  - `exp02 (seed 42)`: `test_acc1=0.8750`, `test_loss=0.4570`
  - `exp05 (seed 123)`: `test_acc1=0.8752`, `test_loss=0.4383`
  - `exp06 (seed 777)`: `test_acc1=0.8648`, `test_loss=0.4701`
