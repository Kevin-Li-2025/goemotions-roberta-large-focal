# GoEmotions RoBERTa-Large Focal Classifier

RoBERTa-large multi-label emotion classifier for the public GoEmotions
benchmark. The released checkpoint predicts 27 fine-grained emotion labels plus
`neutral` from English short-form text, using focal loss and validation-tuned
coordinate thresholds.

Public artifacts:

- Hugging Face model: https://huggingface.co/AliceYin/goemotions-roberta-large-focal-sota
- Kaggle model artifact: https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42

## Status

The current release is a completed seed-42 RoBERTa-large focal-loss run on
`google-research-datasets/go_emotions`, simplified configuration. The
validation-selected threshold policy is the reported policy for the headline
test metrics.

This repository tracks source code, configuration, metrics summaries, research
notes, and release documentation. It does not track raw datasets, Kaggle
outputs, model checkpoints, local caches, credentials, or unrelated Kaggle
projects.

## Results

Primary metric: macro-F1, because rare emotion labels are the main challenge in
GoEmotions.

| Split | Macro-F1 | Micro-F1 | Samples-F1 | Subset Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.5659 | 0.5966 | 0.6051 | 0.4784 |
| Test | 0.5330 | 0.5767 | 0.5859 | 0.4695 |

Threshold comparison on the test split:

| Threshold Policy | Test Macro-F1 |
| --- | ---: |
| Fixed 0.5 | 0.5184 |
| Global validation-tuned threshold | 0.5320 |
| Validation-selected coordinate thresholds | 0.5330 |
| Per-label thresholds | 0.5350 |

The per-label threshold candidate reached the highest test macro-F1, but the
coordinate threshold policy was selected by validation macro-F1 and is the
strict reported policy.

## Quick Inference

Install runtime dependencies:

```bash
python -m pip install torch transformers huggingface_hub safetensors
```

Run prediction with the released model and saved validation-selected thresholds:

```python
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo_id = "AliceYin/goemotions-roberta-large-focal-sota"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)

threshold_data = json.load(open(hf_hub_download(repo_id, "thresholds.json")))
labels = json.load(open(hf_hub_download(repo_id, "labels.json")))["label_names"]
threshold_map = threshold_data[threshold_data["selected"]]
thresholds = [threshold_map[label] for label in labels]

text = "I finally got this working and I am so relieved."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=160)

with torch.no_grad():
    probs = torch.sigmoid(model(**inputs).logits)[0]

predictions = [
    {"label": label, "score": float(prob)}
    for label, prob, threshold in zip(labels, probs, thresholds)
    if prob >= threshold
]

print(predictions)
```

## Reproduce Locally

Create an environment from the repository root:

```bash
git clone https://github.com/Kevin-Li-2025/goemotions-sota-emotion-model.git
cd goemotions-sota-emotion-model
python -m venv .venv
.venv/bin/pip install -r emotion-model/requirements.txt
```

Run a tiny smoke test to validate the data, model, loss, thresholding, and
metrics code paths without a full GPU run:

```bash
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
  --loss_type focal \
  --mixed_precision none
```

Run the release configuration explicitly:

```bash
.venv/bin/python emotion-model/train_goemotions.py \
  --model_name FacebookAI/roberta-large \
  --output_dir emotion-model/outputs/goemotions-roberta-large-focal-seed42 \
  --seed 42 \
  --epochs 4 \
  --learning_rate 1e-5 \
  --train_batch_size 2 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 16 \
  --loss_type focal \
  --focal_alpha 0.38 \
  --focal_gamma 2.8 \
  --threshold_metric macro_f1 \
  --threshold_coordinate_passes 2 \
  --mixed_precision none
```

For Kaggle GPU execution, push the script kernel from the repository root:

```bash
.venv/bin/kaggle kernels push -p emotion-model
```

`kernel-metadata.json` requests `NvidiaTeslaT4` and enables internet access so
the public Hugging Face dataset and base model can be downloaded during the
run.

## Outputs

A completed run writes:

- `model/`: Transformers checkpoint and tokenizer files
- `metrics.json`: validation and test metrics for all threshold policies
- `thresholds.json`: fixed, global, per-label, and selected coordinate thresholds
- `labels.json`: GoEmotions label names and neutral index
- `validation_predictions.csv` and `test_predictions.csv`: decoded labels and probabilities

Generated outputs are intentionally ignored by Git. Use the public Hugging Face
or Kaggle artifacts for the released weights and metrics bundle.

## CI/CD

GitHub Actions is configured with three workflows:

- `CI`: runs on `main`, pull requests, and manual dispatch. It checks tracked
  file boundaries, rejects obvious credential leaks and local absolute paths,
  validates JSON metadata, compiles the training script, and runs offline unit
  smoke tests for thresholding, metric support code, JSON serialization, and
  custom loss functions.
- `Manual End-to-End Smoke`: runs only on manual dispatch. It installs the full
  runtime dependencies and runs a tiny Hugging Face based training pass with
  `hf-internal-testing/tiny-random-bert`.
- `Release Metadata`: runs manually and on `v*` tags. It packages README,
  model-card, research notes, experiment JSON, kernel metadata, and dependency
  metadata into `goemotions-release-metadata.tgz`. On version tags, it attaches
  that bundle to a GitHub Release. It does not package model weights.

The workflows follow GitHub Actions' least-privilege pattern: CI uses
read-only repository permissions, while release packaging requests write
permissions only for creating or updating GitHub Releases.

## Repository Layout

```text
emotion-model/
  train_goemotions.py                         # training, evaluation, threshold tuning
  kernel-metadata.json                        # Kaggle script-kernel config
  requirements.txt                            # runtime dependencies
  MODEL_CARD.md                               # Hugging Face model card source
  PROMOTION.md                                # launch copy and public links
  RESEARCH.md                                 # references, run history, next experiments
  experiments/
    2026-05-31-roberta-large-focal-seed42.json
    2026-05-31-weighted-bce-baseline.json
    2026-05-31-asymmetric-large-failed.json
    2026-05-31-deberta-base-nonfinite-logits.json
```

## Model Details

- Base model: `FacebookAI/roberta-large`
- Dataset: `google-research-datasets/go_emotions`, simplified split
- Task: multi-label classification over 28 labels
- Loss: focal loss, alpha `0.38`, gamma `2.8`
- Epochs: `4`
- Learning rate: `1e-5`
- Effective batch size: `2 x 16` gradient accumulation
- Mixed precision: disabled for stability on Kaggle T4
- Threshold policy: coordinate search selected by validation macro-F1
- Seed: `42`

## Limitations

- GoEmotions labels are subjective and context-dependent.
- The model is best suited to English short-form text close to the dataset
  distribution.
- It should not be used as the sole input for high-stakes decisions.
- Public-reference SOTA-level is a defensible claim for the references tracked
  in `RESEARCH.md`; it is not a claim of formal leaderboard dominance.

## References

- GoEmotions paper: https://aclanthology.org/2020.acl-main.372/
- Google Research summary: https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/
- Hugging Face dataset: https://huggingface.co/datasets/google-research-datasets/go_emotions
- Public reference notes: [RESEARCH.md](RESEARCH.md)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for scope, validation commands, and
data-handling rules. Keep raw datasets, checkpoints, credentials, and generated
outputs out of Git.
