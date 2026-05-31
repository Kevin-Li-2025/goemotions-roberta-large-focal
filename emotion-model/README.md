# GoEmotions Emotion Model

This project trains a strong multi-label emotion classifier on Google's
GoEmotions dataset: 58k curated Reddit comments labeled with 27 fine-grained
emotion classes plus `neutral`.

The default full run uses `microsoft/deberta-v3-large`, class-balanced
multi-label loss, validation threshold tuning, and test-set reporting. The
Kaggle default uses fp32 because the current Kaggle T4 image has an fp16
gradient-scaling incompatibility with this stack. It is set up for Kaggle GPU
execution, with a local smoke-test path for fast checks on Apple Silicon MPS.

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
/kaggle/working/goemotions-deberta-v3-large
```

Key artifacts:

- `model/`: fine-tuned model and tokenizer
- `metrics.json`: validation/test metrics for fixed, global, and per-label thresholds
- `thresholds.json`: selected validation-tuned decision thresholds
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

Use `microsoft/deberta-v3-large` for the strongest run; on Apple MPS it is
usually slower and more memory-sensitive than on a Kaggle/NVIDIA GPU.
