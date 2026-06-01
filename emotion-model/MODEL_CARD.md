---
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: FacebookAI/roberta-large
datasets:
- google-research-datasets/go_emotions
metrics:
- f1
language:
- en
tags:
- goemotions
- emotion-classification
- multi-label-classification
- roberta
- roberta-large
- focal-loss
- threshold-optimization
- nlp
model-index:
- name: GoEmotions RoBERTa-large Focal Loss Classifier
  results:
  - task:
      type: text-classification
      name: Multi-label emotion classification
    dataset:
      name: GoEmotions simplified
      type: google-research-datasets/go_emotions
      config: simplified
      split: test
    metrics:
    - type: f1
      value: 0.5330202487288448
      name: Macro-F1
    - type: f1
      value: 0.5766508516761297
      name: Micro-F1
    - type: f1
      value: 0.5859415444821746
      name: Samples-F1
---

# GoEmotions RoBERTa-large Focal Loss Classifier

This model is a RoBERTa-large multi-label emotion classifier trained on the
public GoEmotions simplified split. It predicts 27 fine-grained emotions plus
`neutral` from English Reddit-style text.

The run uses focal loss for label imbalance and validation-tuned coordinate
thresholds for multi-label decisions. It is a competitive public-reference
result: the validation-selected policy reached test macro-F1 0.5330, while the
strongest public model card found during this iteration reported test macro-F1
0.519. This is not presented as formal SOTA because there is no official
GoEmotions leaderboard comparison here.

A follow-up metrics-only seed sweep reran the same recipe with seeds 43 and 44.
Both repeat seeds exceeded the released seed-42 point estimate, reaching test
macro-F1 0.5365 and 0.5380 respectively. Their bootstrap intervals still overlap
the tracked public reference, so the public claim remains intentionally
conservative.

## Links

- Kaggle model artifact: https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42
- Kaggle inference notebook: https://www.kaggle.com/code/kevin250304/goemotions-roberta-large-focal-model-demo
- Training source: `emotion-model/train_goemotions.py` in the release repository
- Dataset: https://huggingface.co/datasets/google-research-datasets/go_emotions
- GoEmotions paper: https://aclanthology.org/2020.acl-main.372/

## Maintainer

- GitHub: `Kevin-Li-2025`
- Kaggle: `kevin250304`
- Hugging Face: `AliceYin`

## Results

| Split | Macro-F1 | Micro-F1 | Samples-F1 | Subset accuracy |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.5659 | 0.5966 | 0.6051 | 0.4784 |
| Test | 0.5330 | 0.5767 | 0.5859 | 0.4695 |

Threshold selection on validation:

| Threshold policy | Validation macro-F1 | Validation micro-F1 | Validation samples-F1 |
| --- | ---: | ---: | ---: |
| Fixed 0.5 | 0.5147 | 0.6021 | 0.6086 |
| Global validation-tuned threshold | 0.5383 | 0.5676 | 0.5783 |
| Per-label thresholds | 0.5634 | 0.5925 | 0.6007 |
| Coordinate thresholds | 0.5659 | 0.5966 | 0.6051 |

Additional threshold candidates on test:

| Threshold policy | Test macro-F1 |
| --- | ---: |
| Fixed 0.5 | 0.5184 |
| Global threshold | 0.5320 |
| Validation coordinate search | 0.5330 |
| Per-label thresholds | 0.5350 |

The headline result uses the validation-selected coordinate threshold policy to
avoid test-set overfitting. The per-label threshold candidate reached the
highest test macro-F1, but it was not selected by validation macro-F1 and is
therefore not the headline policy. The exported `thresholds.json` stores all
threshold policies plus `selected: "coordinate"`.

Additional repeat-seed robustness check:

| Seed | Validation macro-F1 | Test macro-F1 | Test micro-F1 | Test samples-F1 | Test macro-F1 95% CI |
| ---: | ---: | ---: | ---: | ---: | --- |
| 43 | 0.5588 | 0.5365 | 0.5909 | 0.5974 | [0.5139, 0.5565] |
| 44 | 0.5679 | 0.5380 | 0.5938 | 0.5997 | [0.5163, 0.5571] |
| Mean | 0.5633 | 0.5373 | 0.5923 | 0.5986 | - |

## Intended Use

Use this model for research, benchmarking, exploratory emotion analysis, and
building GoEmotions-compatible classifiers. It is best suited to English
short-form text that resembles the public GoEmotions data distribution.

This model should not be used as the sole basis for decisions that affect
people in high-stakes settings. Emotion labels are subjective, culturally
dependent, and sensitive to context that may not be present in a single comment.

## Quick Start

```python
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF_MODEL_ID = "AliceYin/goemotions-roberta-large-focal-sota"
KAGGLE_MODEL_URL = (
    "https://www.kaggle.com/models/kevin250304/"
    "goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42"
)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)

with open(hf_hub_download(HF_MODEL_ID, "thresholds.json"), encoding="utf-8") as f:
    threshold_data = json.load(f)
with open(hf_hub_download(HF_MODEL_ID, "labels.json"), encoding="utf-8") as f:
    labels = json.load(f)["label_names"]

selected_policy = threshold_data["selected"]
selected_thresholds = threshold_data[selected_policy]
threshold_map = (
    selected_thresholds["per_label"]
    if selected_policy == "global"
    else selected_thresholds
)
thresholds = [threshold_map[label] for label in labels]

text = "I finally got this working and I am so relieved."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=160)

with torch.no_grad():
    probs = torch.sigmoid(model(**inputs).logits)[0]

predicted = [
    {"label": label, "score": float(prob)}
    for label, prob, threshold in zip(labels, probs, thresholds)
    if prob >= threshold
]
print(predicted)
```

## Training Details

- Base model: `FacebookAI/roberta-large`
- Dataset: `google-research-datasets/go_emotions`, simplified configuration
- Loss: focal loss, alpha 0.38, gamma 2.8
- Epochs: 4
- Learning rate: 1e-5
- Batch size: 2 with gradient accumulation 16
- Mixed precision: disabled for stability
- Threshold selection: validation macro-F1 coordinate search
- Seed: 42

## Citation

```bibtex
@inproceedings{demszky-etal-2020-goemotions,
  title = "{G}o{E}motions: A Dataset of Fine-Grained Emotions",
  author = "Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith",
  booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
  year = "2020",
  doi = "10.18653/v1/2020.acl-main.372",
  pages = "4040--4054"
}
```

## Reproducibility

The Kaggle artifact includes `metrics.json`, `thresholds.json`, `labels.json`,
the tokenizer, the model weights, and the Kaggle run log. The training script
and experiment notes record the exact settings used for the reported metrics.
