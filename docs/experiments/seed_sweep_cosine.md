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
- test_acc1 mean ± std: TODO
- test_acc5 mean ± std: TODO
- test_loss mean ± std: TODO

## Notes
- Fill values from `scripts/seed_sweep_summary.py` output.
