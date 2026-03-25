# Roadmap

This file tracks repo-level milestones and next steps for the showcase project.

Detailed experiment evidence lives in `docs/experiments/`. This roadmap is intentionally higher-level: it summarizes what is already shipped in the repo and what is most valuable to build next for a portfolio targeting Data Science / Data Analytics roles, with Software Development as a secondary signal.

Last updated: 2026-03-25

## Current Showcase State

- Current promoted recipe: `exp17_cosine_es_img256_wd1e3_s42`
- Selection rule: prefer robustness (`mean +/- std` across seeds), not only the best single run
- Current delivery shape: local training/eval/inference, cross-run experiment analytics, local MLflow tracking, FastAPI service, Docker image, Azure Container Apps deployment, CI, and release-oriented CD
- Main evidence trail:
  - `README.md`
  - `docs/experiments/README.md`
  - `docs/experiments/cross_run_analytics.md`
  - `docs/experiments/error_analysis_exp17.md`
  - `docs/experiments/calibration_exp17.md`
  - `deploy/showcase_model.json`

## Completed Milestones

- [x] Build a reproducible PyTorch training and evaluation pipeline with YAML configs and CLI overrides
- [x] Add isolated per-run artifacts (`checkpoints`, `metrics.csv`, evaluation JSON, plots) and a reproducible runner script
- [x] Create a structured experiment catalog in `docs/experiments/` with a comparison table and documented selection logic
- [x] Use robustness-first model selection with targeted seed sweeps instead of relying on a single best run
- [x] Add structured error analysis exports and plots: per-class metrics, top confusion pairs, confidence histograms, and overconfident error galleries
- [x] Promote the `exp17` recipe as the current showcase model after a controlled screening cycle (`image_size`, `weight_decay`, one augmentation test)
- [x] Preserve negative results in the project story, including the rejected mild `ColorJitter` recipe
- [x] Add a shared inference layer plus CLI prediction path from trained checkpoints
- [x] Add a FastAPI prediction service with health checks, upload validation, top-k responses, and model metadata
- [x] Add Docker packaging for local reproducibility
- [x] Add cloud-minimum deployment to Azure Container Apps for the showcase model
- [x] Add release-oriented Continuous Delivery via GitHub Actions, using a pinned release artifact manifest and post-deploy smoke checks
- [x] Add Makefile shortcuts for setup, testing, training, evaluation, prediction, serving, Docker, and live endpoint smoke checks
- [x] Add automated tests and GitHub Actions CI for core training, transforms, inference, plotting, error analysis, and API behavior
- [x] Add optional local MLflow tracking for params, metrics, small artifacts, and run comparison, with SQLite-backed local metadata storage
- [x] Add post-hoc calibration analysis for `exp17` with validation-fit temperature scaling, reliability diagrams, and confidence-threshold reporting
- [x] Add a compact cross-run experiment analytics layer from configs + run artifacts, with a repo-level summary and public synthesis note

## Ranked Next Steps

### 1. Error-Delta Report For `exp17` vs `exp02`

Why this is first:
- the repo already has strong per-model error analysis
- a direct before/after comparison would make the improvement story much easier to communicate

Portfolio payoff:
- strong evidence-based iteration story
- useful talking point for interviews: "where exactly did the new recipe improve?"

Likely deliverables:
- per-class accuracy deltas
- confusion-pair deltas
- confidence bucket deltas
- top-5 recovery deltas

### 2. Interpretability Mini-Study (Grad-CAM)

Why this is second:
- adds explanatory depth without requiring a full new training cycle
- fits naturally with the repo's existing error-analysis workflow
- creates strong interview talking points around failure modes and model attention

Portfolio payoff:
- stronger DS storytelling
- helps explain correct vs incorrect predictions visually
- complements calibration by adding qualitative, not only quantitative, trust analysis

Likely deliverables:
- Grad-CAM examples on hard confusion pairs
- correct vs incorrect comparison cases
- a short note on what the model appears to attend to

### 3. One Controlled Backbone Upgrade

Why this is third:
- after recipe tuning on `ResNet18`, the next clean modeling question is whether capacity is now the main bottleneck

Portfolio payoff:
- shows disciplined model comparison instead of random trial-and-error
- can improve the modeling side of the project without changing its evaluation philosophy

Likely deliverables:
- `ResNet18` vs `ResNet34` or `ResNet50` under the same protocol
- updated comparison table and a short decision note

### 4. Data-Centric Audit For Hard Classes

Why this is fourth:
- hard breed clusters may still hide label noise, ambiguous samples, or near-duplicates
- this is a good way to deepen the analytical side of the project

Portfolio payoff:
- shows data quality thinking, not only model tuning
- especially relevant for DS / DA interview discussions

Likely deliverables:
- hard-class sample review
- nearest-neighbor / embedding inspection
- notes on ambiguous or suspicious examples

### 5. Lightweight Demo / Landing Page

Why this is fifth:
- improves polish and reviewability
- but adds less DS/DA value than the items above

Portfolio payoff:
- nice recruiter-facing polish
- slightly stronger software/product presentation

Likely deliverables:
- a small page on top of the existing FastAPI app
- example requests, prediction display, and project links

## Guiding Principles

- Prefer additions that increase Data Science / Data Analytics signal more than infrastructure sprawl.
- Keep detailed run evidence in `docs/experiments/` and `runs/`; use this roadmap only for repo-level milestones.
- Prefer controlled comparisons that change one major axis at a time.
- Favor additions that create explainable project storylines, not only new keywords.
- When effort is comparable, prioritize features that improve both resume value and interview discussion quality.
