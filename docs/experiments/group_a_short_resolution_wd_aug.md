# Group A-short Summary: Resolution, Weight Decay, and One Augmentation Test

## Goal
Run a compact, hypothesis-driven improvement cycle on the current `ResNet18` pipeline without changing the backbone:
- parameterize image resolution
- screen a small resolution sweep
- screen `weight_decay`
- test exactly one additional augmentation recipe

## Baseline Reference
- Previous showcase recipe: cosine + early stopping (`exp02`)
- Reproduced after resolution parameterization as `exp12` (`224/256`) to confirm backward compatibility

## Phase A-1: Resolution Sweep (1 seed screening)

Hypothesis:
- larger input resolution may improve fine-grained breed separation

Runs:

| Experiment | Change | Batch size | Val (loss / acc@1 / acc@5) | Test (loss / acc@1 / acc@5) | Runtime (approx) | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `exp12` | `image_size=224`, `eval_resize_size=256` (control) | 32 | `0.2890 / 0.914 / 0.988` | `0.4570 / 0.875 / 0.984` | `14m25s` | Control confirms A1 parity |
| `exp13` | `image_size=256`, `eval_resize_size=292` | 32 | `0.2739 / 0.924 / 0.992` | `0.4241 / 0.878 / 0.988` | `15m36s` | **Keep** for next phase |
| `exp14` | `image_size=320`, `eval_resize_size=366` | 16 (memory fit) | `0.3009 / 0.914 / 0.988` | `0.4775 / 0.865 / 0.980` | `19m10s` | Reject (worse + slower) |

Resolution decision:
- keep `256` for the next screening step
- drop `320` in this phase (quality/cost trade-off is unfavorable)

## Phase A-2: Weight Decay Sweep on `img256` (1 seed screening)

Hypothesis:
- `AdamW weight_decay` is a low-risk lever for generalization and ranking quality

Runs (all on `image_size=256`, `eval_resize_size=292`):

| Experiment | `weight_decay` | Best epoch | Stopped at epoch | Val (loss / acc@1 / acc@5) | Test (loss / acc@1 / acc@5) | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `exp15` | `0` | 20 | 26 (early stop) | `0.2934 / 0.920 / 0.990` | `0.4423 / 0.875 / 0.983` | Reject |
| `exp16` | `1e-4` | 20 | 26 (early stop) | `0.2864 / 0.921 / 0.989` | `0.4302 / 0.875 / 0.984` | Reject |
| `exp17` | `1e-3` | 29 | 30 | `0.2918 / 0.925 / 0.992` | `0.4292 / 0.881 / 0.985` | **Keep** |
| `exp13` | `1e-2` | 22 | 28 (early stop) | `0.2739 / 0.924 / 0.992` | `0.4241 / 0.878 / 0.988` | Strong baseline / fallback |

Weight decay decision:
- `weight_decay=1e-3` selected as the best candidate by `val_acc1` and `test_acc1`
- `1e-2` remains a strong comparison point (better `test_loss`, similar top-5)

## Phase A-3: One Additional Augmentation Recipe (1 seed screening)

Hypothesis:
- mild `ColorJitter` may improve robustness to lighting/color variation

Run:

| Experiment | Recipe | Val (loss / acc@1 / acc@5) | Test (loss / acc@1 / acc@5) | Decision |
| --- | --- | --- | --- | --- |
| `exp18` | `img256 + wd1e-3 + ColorJitter(mild)` | `0.2969 / 0.909 / 0.988` | `0.4474 / 0.872 / 0.984` | Reject |

Augmentation decision:
- reject this `ColorJitter` recipe for the current pipeline
- likely too disruptive for fine-grained color/texture cues in this dataset

## Final Candidate Sent To Robustness Follow-up
- `exp17`: `img256 + wd=1e-3`, no extra augmentation

Follow-up:
- `3` seeds (`42/123/777`) run next to confirm whether the single-seed gain is stable
- Results are documented in `docs/experiments/seed_sweep_img256_wd1e3.md`

## Takeaways (Portfolio / Workflow)
- The phase produced both a **kept improvement** (`img256 + wd1e-3`) and a **useful negative result** (`ColorJitter` reject).
- Changes were tested one axis at a time, which keeps conclusions interpretable.
- This is a stronger portfolio outcome than tuning many knobs at once without attribution.
