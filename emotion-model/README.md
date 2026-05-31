# GoEmotions Emotion Model

This project trains a strong multi-label emotion classifier on Google's
GoEmotions dataset: 58k curated Reddit comments labeled with 27 fine-grained
emotion classes plus `neutral`.

The default full run currently uses `microsoft/deberta-v3-base`, plain
multi-label BCE, validation threshold tuning, and test-set reporting. This is a
stability reset after two DeBERTa-v3-large Kaggle runs diverged with
`grad_norm=nan` and `eval_loss=nan`. The next large-model sweep should only run
after this base configuration produces finite losses and non-collapsed
predictions.

References:

- Google Research: https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/
- ACL Anthology: https://aclanthology.org/2020.acl-main.372/
- Hugging Face dataset: https://huggingface.co/datasets/google-research-datasets/go_emotions

## Local Smoke Test

Run a tiny model on a small subset to verify the code path:

```bash
cd /Users/yinxiaogou/Documents/Kaggle
.venv/bin/python emotion-model/train_goemotions.py \
  --model_name hf-internal-testing/tiny-random-bert \
  --output_dir emotion-model/outputs/smoke-tiny-random-bert \
  --max_train_samples 96 \
  --max_eval_samples 64 \
  --max_test_samples 64 \
  --max_steps 3 \
  --eval_steps 1 \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --max_length 96 \
  --loss_type bce \
  --mixed_precision none
```

## Full Kaggle GPU Run

The folder contains a valid Kaggle script kernel configuration. From the repo
root:

```bash
.venv/bin/kaggle kernels push -p emotion-model
```

The kernel runs with internet enabled so Hugging Face can provide the public
dataset and pretrained model. `kernel-metadata.json` requests `NvidiaTeslaT4`;
do not pass `--accelerator GPU`, because that can override the specific machine
shape and assign an incompatible P100. Outputs are written under:

```text
/kaggle/working/goemotions-deberta-v3-base-stable
```

Key artifacts:

- `model/`: fine-tuned model and tokenizer
- `metrics.json`: validation/test metrics for fixed, global, and per-label thresholds
- `thresholds.json`: selected validation-tuned decision thresholds, including a
  coordinate-search candidate that directly optimizes validation macro-F1 after
  post-processing
- `validation_predictions.csv` and `test_predictions.csv`: decoded labels and probabilities

## Local Full Run

If local GPU memory is enough:

```bash
.venv/bin/python emotion-model/train_goemotions.py \
  --model_name microsoft/deberta-v3-base \
  --output_dir emotion-model/outputs/deberta-v3-base-goemotions \
  --epochs 4 \
  --train_batch_size 4 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing \
  --mixed_precision none
```

Use `microsoft/deberta-v3-large` for the strongest run after the base sanity
run is stable; on Apple MPS it is usually slower and more memory-sensitive than
on a Kaggle/NVIDIA GPU.

To reproduce the original weighted BCE baseline, pass:

```bash
--loss_type bce --use_pos_weight
```

To reproduce the failed asymmetric large run, pass:

```bash
--model_name microsoft/deberta-v3-large --loss_type asymmetric --no_pos_weight
```
