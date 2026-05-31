# GoEmotions SOTA Research Notes

Last updated: 2026-05-31

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
- A public DeBERTa-v3-large model card emphasizes per-label threshold tuning:
  https://huggingface.co/FurqonAryadana/deberta-emotion-multilabel-0.5007
- Recent literature and benchmark-style papers point to label imbalance,
  thresholding, label dependency modeling, and data augmentation as the main
  levers:
  https://arxiv.org/abs/2403.06108
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12528697/

## Iteration Roadmap

1. Establish a stable DeBERTa-v3-large fp32 baseline on Kaggle T4 with
   class-balanced BCE and per-label threshold tuning.
2. Add loss variants:
   - focal loss
   - asymmetric loss for multi-label imbalance
   - class-balanced focal loss
3. Run seed sweeps for the best loss and threshold policy.
4. Train complementary models for ensembling:
   - `microsoft/deberta-v3-large`
   - `roberta-large`
   - `microsoft/deberta-v3-base` for faster sweep feedback
5. Add probability calibration and threshold search variants:
   - optimize per-label F1
   - optimize macro-F1 directly with validation coordinate descent
   - neutral exclusivity on/off
6. Try label-dependency inference:
   - classifier chain style post-processing
   - co-occurrence prior correction
7. Only after a strong clean baseline, evaluate augmentation or distillation.

## Commit And Data Policy

Commit code, configs, metrics summaries, and small research notes. Do not commit
raw datasets, model checkpoints, Kaggle outputs, credentials, caches, or
unrelated Kaggle projects.
