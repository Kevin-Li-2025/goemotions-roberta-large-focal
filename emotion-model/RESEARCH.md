# GoEmotions Public-Reference Research Notes

Last updated: 2026-06-01

## Dataset And Metric Target

GoEmotions is the primary benchmark for this project: 58k curated Reddit
comments labeled with 27 fine-grained emotions plus `neutral`.

Primary target metric: validation/test `macro_f1`, because rare emotions are
the hard part of this benchmark. Track `micro_f1`, `samples_f1`,
`hamming_loss`, and subset accuracy as secondary metrics.

## Current Public Reference Points

- Google Research / ACL 2020 introduced the dataset and official benchmark:
  https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/
  and https://aclanthology.org/2020.acl-main.372/
- Hugging Face hosts the public benchmark data:
  https://huggingface.co/datasets/google-research-datasets/go_emotions
- A public DeBERTa-v3-large GoEmotions model card reports macro-F1 0.518 and
  micro-F1 0.598:
  https://huggingface.co/duelker/samo-goemotions-deberta-v3-large
- A public RoBERTa-large GoEmotions model card reports test macro-F1 around
  0.519 and attributes the gain to focal loss, per-label thresholds, mean
  pooling, targeted augmentation, and gradual unfreezing. This is now the
  default direction because the Kaggle DeBERTa-v3 runs are failing before
  producing usable logits:
  https://huggingface.co/Lakssssshya/roberta-large-goemotions
- A public DeBERTa-v3-large model card emphasizes per-label threshold tuning:
  https://huggingface.co/FurqonAryadana/deberta-emotion-multilabel-0.5007
- Recent literature and benchmark-style papers point to label imbalance,
  thresholding, label dependency modeling, and data augmentation as the main
  levers:
  https://arxiv.org/abs/2403.06108
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12528697/

## Iteration Roadmap

1. Establish a stable RoBERTa-large run on Kaggle T4 with focal loss,
   per-label/coordinate threshold tuning, and hard non-finite loss guards.
2. Return to DeBERTa-v3-large only after its Kaggle non-finite-logit failure is
   isolated; use RoBERTa metrics as the active regression check.
3. Add loss variants:
   - asymmetric loss for multi-label imbalance: implemented as the next default
     experiment after the weighted BCE baseline
   - focal loss: implemented as a selectable experiment
   - class-balanced focal loss: available by combining `--loss_type focal` with
     `--use_pos_weight`
4. Run seed sweeps for the best loss and threshold policy.
5. Train complementary models for ensembling:
   - `microsoft/deberta-v3-large`
   - `roberta-large`
   - `microsoft/deberta-v3-base` for faster sweep feedback
6. Add probability calibration and threshold search variants:
   - optimize per-label F1
   - optimize macro-F1 directly with validation coordinate descent
   - neutral exclusivity on/off
7. Try label-dependency inference:
   - classifier chain style post-processing
   - co-occurrence prior correction
8. Only after a strong clean baseline, evaluate augmentation or distillation.

## Failed Runs

- 2026-05-31 weighted BCE DeBERTa-v3-large: completed, but diverged with
  `grad_norm=nan`, `eval_loss=nan`, validation macro-F1 0.005894, and test
  macro-F1 0.006070.
- 2026-05-31 asymmetric-loss DeBERTa-v3-large: completed, but diverged with
  the same NaN pattern, validation macro-F1 0.005894, and test macro-F1
  0.006070. The next run is a stability reset, not a SOTA attempt.
- 2026-05-31 stable DeBERTa-v3-base: failed fast before metrics. The first
  guarded Kaggle run stopped with non-finite logits on Tesla T4, so the active
  path is now RoBERTa-large rather than more DeBERTa sweeps.

## Strong Runs

- 2026-05-31 RoBERTa-large focal seed 42: completed cleanly. Coordinate
  thresholds selected by validation macro-F1 reached validation macro-F1
  0.565864 and test macro-F1 0.533020. This exceeds the strongest public
  model-card reference found during this iteration, which reports test macro-F1
  0.519 for a RoBERTa-large focal/per-label-threshold model.
- 2026-06-01 RoBERTa-large focal seed sweep: seeds 43 and 44 completed as a
  metrics-only Kaggle run with checkpoint/model saving disabled. Both repeat
  seeds selected coordinate thresholds by validation macro-F1 and exceeded the
  seed-42 point estimate: seed 43 reached test macro-F1 0.536506 and seed 44
  reached test macro-F1 0.538002. The two-seed mean test macro-F1 is 0.537254.
  Test macro-F1 bootstrap intervals were [0.513942, 0.556526] for seed 43 and
  [0.516281, 0.557067] for seed 44, so the project should keep the
  conservative "competitive public-reference" wording rather than formal SOTA.

## Commit And Data Policy

Commit code, configs, metrics summaries, and small research notes. Do not commit
raw datasets, model checkpoints, Kaggle outputs, credentials, caches, or
unrelated Kaggle projects.
