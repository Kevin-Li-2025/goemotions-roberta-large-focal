# GoEmotions RoBERTa-Large Focal Loss Classifier

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-public%20model-yellow)](https://huggingface.co/AliceYin/goemotions-roberta-large-focal-sota)
[![Kaggle](https://img.shields.io/badge/Kaggle-public%20artifact-blue)](https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42)
[![CI](https://github.com/Kevin-Li-2025/goemotions-roberta-large-focal/actions/workflows/ci.yml/badge.svg)](https://github.com/Kevin-Li-2025/goemotions-roberta-large-focal/actions/workflows/ci.yml)
[![Manual Smoke](https://github.com/Kevin-Li-2025/goemotions-roberta-large-focal/actions/workflows/manual-smoke.yml/badge.svg)](https://github.com/Kevin-Li-2025/goemotions-roberta-large-focal/actions/workflows/manual-smoke.yml)
[![Release Metadata](https://github.com/Kevin-Li-2025/goemotions-roberta-large-focal/actions/workflows/release.yml/badge.svg)](https://github.com/Kevin-Li-2025/goemotions-roberta-large-focal/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

RoBERTa-large multi-label classifier for Google's GoEmotions benchmark. The
released model predicts 27 fine-grained emotion labels plus `neutral` and uses
focal loss with validation-tuned coordinate thresholds. The Hugging Face repo is
the easiest inference endpoint; the Kaggle Models page is the public artifact
mirror with weights, metrics, labels, thresholds, and run logs.

## Released Seed-42 Result

| Split | Macro-F1 | Micro-F1 | Samples-F1 |
| --- | ---: | ---: | ---: |
| Validation | 0.5659 | 0.5966 | 0.6051 |
| Test | 0.5330 | 0.5767 | 0.5859 |

The headline result uses the validation-selected coordinate threshold policy to
avoid test-set overfitting. A per-label threshold candidate reached test
macro-F1 0.5350, but it was not selected by the validation policy and is
therefore reported only as a candidate.

Validation threshold selection:

| Threshold Policy | Validation Macro-F1 | Test Macro-F1 |
| --- | ---: | ---: |
| Fixed 0.5 | 0.5147 | 0.5184 |
| Global validation-tuned threshold | 0.5383 | 0.5320 |
| Per-label thresholds | 0.5634 | 0.5350 |
| Coordinate thresholds | 0.5659 | 0.5330 |

The selected policy is based on validation macro-F1, not test-set tuning.

## Seed Sweep Robustness

The same RoBERTa-large focal-loss recipe was rerun as a metrics-only Kaggle
seed sweep with seeds 43 and 44. Both repeat seeds beat the released seed-42
test macro-F1 and the tracked public model-card reference point of 0.519.

| Seed | Validation Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test Samples-F1 | Test Macro-F1 95% CI |
| ---: | ---: | ---: | ---: | ---: | --- |
| 43 | 0.5588 | 0.5365 | 0.5909 | 0.5974 | [0.5139, 0.5565] |
| 44 | 0.5679 | 0.5380 | 0.5938 | 0.5997 | [0.5163, 0.5571] |
| Mean | 0.5633 | 0.5373 | 0.5923 | 0.5986 | - |

This strengthens the recipe-level claim, but the public wording remains
conservative: competitive public-reference result, not formal leaderboard SOTA.

## Start Here

- [Project README](emotion-model/README.md): setup, inference, training, and reproducibility.
- [Model card](emotion-model/MODEL_CARD.md): public Hugging Face model card source.
- [Kaggle inference notebook](emotion-model/examples/kaggle_inference_demo.ipynb): runnable usage demo.
- [Seed sweep record](emotion-model/experiments/2026-06-01-roberta-large-focal-seed-sweep.json): repeat-seed metrics and bootstrap CIs.
- [Research notes](emotion-model/RESEARCH.md): reference points, failed runs, and iteration history.
- [Promotion kit](emotion-model/PROMOTION.md): concise public launch copy and links.
- [CI/CD](emotion-model/README.md#cicd): validation, smoke testing, and release metadata automation.

## Public Artifacts

- Hugging Face: https://huggingface.co/AliceYin/goemotions-roberta-large-focal-sota
- Kaggle Models: https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42
- Kaggle inference notebook: https://www.kaggle.com/code/kevin250304/goemotions-roberta-large-focal-model-demo
- Training source: [`emotion-model/train_goemotions.py`](emotion-model/train_goemotions.py)

## Repository Scope

This repository tracks code, configs, metrics summaries, model-card text, CI/CD,
and research notes for the GoEmotions release. It intentionally does not track
raw datasets, Kaggle outputs, model checkpoints, local caches, credentials, or
unrelated Kaggle projects.

## Naming

The repository name uses the concrete model family and training recipe:
`goemotions-roberta-large-focal`. The public claim is intentionally narrow:
competitive public-reference GoEmotions result, not formal leaderboard SOTA.
