# Seed sweep: Group A-short candidate (`img256`, `wd=1e-3`)

## Goal
Validate whether the best Group A-short screening recipe remains strong across multiple random seeds.

Candidate recipe:
- `image_size=256`
- `eval_resize_size=292`
- `weight_decay=1e-3`
- cosine scheduler + early stopping

## Runs
- `runs/exp17_cosine_es_img256_wd1e3_s42` (seed 42)
- `runs/exp19_cosine_es_img256_wd1e3_s123` (seed 123)
- `runs/exp20_cosine_es_img256_wd1e3_s777` (seed 777)

## Commands
```bash
source .venv/bin/activate
./scripts/run_experiment.sh configs/experiments/exp19_cosine_es_img256_wd1e3_s123.yaml runs/exp19_cosine_es_img256_wd1e3_s123
./scripts/run_experiment.sh configs/experiments/exp20_cosine_es_img256_wd1e3_s777.yaml runs/exp20_cosine_es_img256_wd1e3_s777
python scripts/seed_sweep_summary.py --runs runs/exp17_cosine_es_img256_wd1e3_s42 runs/exp19_cosine_es_img256_wd1e3_s123 runs/exp20_cosine_es_img256_wd1e3_s777
```

## Results
- test_acc1 mean ± std: `0.8829 ± 0.0015`
- test_acc5 mean ± std: `0.9854 ± 0.0014`
- test_loss mean ± std: `0.4159 ± 0.0158`

## Notes
- Per-run:
  - `exp17 (seed 42)`: `test_acc1=0.8812`, `test_loss=0.4292`
  - `exp19 (seed 123)`: `test_acc1=0.8842`, `test_loss=0.3985`
  - `exp20 (seed 777)`: `test_acc1=0.8833`, `test_loss=0.4201`
- Compared to the previous cosine showcase seed sweep (`exp02` recipe):
  - `test_acc1`: `0.8829 ± 0.0015` vs `0.8717 ± 0.0059`
  - better mean accuracy and lower variance
  - lower mean test loss (`0.4159` vs `0.4551`)

## Decision
- Promote this recipe as the new showcase configuration for the project.
- Keep `exp02` documentation and error analysis as historical reference; refresh error analysis later for the new showcase recipe.
