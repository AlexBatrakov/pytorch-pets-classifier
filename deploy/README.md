# Deploy Metadata

This directory defines the tracked source of truth for the serving artifact used by release-oriented delivery.

## Current contract

- serving model version:
  - `exp17_cosine_es_img256_wd1e3_s42`
- canonical local checkpoint:
  - `runs/exp17_cosine_es_img256_wd1e3_s42/checkpoints/best.pt`
- pinned public artifact:
  - GitHub Release asset on `v1.1.0`
- tracked manifest:
  - `deploy/showcase_model.json`

## Why this exists

The showcase checkpoint is intentionally not tracked in git.

That is good repo hygiene, but it means GitHub-hosted delivery cannot depend on a local `runs/` directory on the author's machine.

The manifest solves that problem by recording:

- which model version should be served
- which GitHub Release asset should be downloaded
- which checksum the workflow should verify
- which local run originally produced the pinned artifact

## Delivery expectation

Future deployment workflows should treat `deploy/showcase_model.json` as the source of truth for:

- the checkpoint asset to fetch
- the checksum to verify
- the model version exposed by the API

The workflow should not guess the serving checkpoint from:

- `checkpoints/best.pt`
- the latest run directory
- ad hoc local files on the author's machine
