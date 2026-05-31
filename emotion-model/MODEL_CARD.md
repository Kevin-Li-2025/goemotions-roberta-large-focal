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
- focal-loss
- threshold-optimization
- sota-level
---

# GoEmotions RoBERTa-large Focal SOTA-Level Classifier

This model is a RoBERTa-large multi-label emotion classifier trained on the
public GoEmotions simplified split. It predicts 27 fine-grained emotions plus
`neutral` from English Reddit-style text.

The run uses focal loss for label imbalance and validation-tuned coordinate
thresholds for multi-label decisions. It is a strong public-reference result:
the validation-selected policy reached test macro-F1 0.5330, while the strongest
public model card found during this iteration reported test macro-F1 0.519.

## Links

- Kaggle model artifact: https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42
- Training kernel: https://www.kaggle.com/code/kevin250304/goemotions-roberta-large-focal-sweep
- Dataset: https://huggingface.co/datasets/google-research-datasets/go_emotions
- GoEmotions paper: https://aclanthology.org/2020.acl-main.372/

## Results

| Split | Macro-F1 | Micro-F1 | Samples-F1 | Subset accuracy |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.5659 | 0.5966 | 0.6051 | 0.4784 |
| Test | 0.5330 | 0.5767 | 0.5859 | 0.4695 |

Additional threshold candidates on test:

| Threshold policy | Test macro-F1 |
| --- | ---: |
| Fixed 0.5 | 0.5184 |
| Global threshold | 0.5320 |
| Validation coordinate search | 0.5330 |
| Per-label thresholds | 0.5350 |

The exported `thresholds.json` stores the validation-selected coordinate
thresholds used for the headline test metrics.

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
  pages = "4040--4054"
}
```

## Reproducibility

The Kaggle artifact includes `metrics.json`, `thresholds.json`, `labels.json`,
the tokenizer, the model weights, and the Kaggle run log. The training kernel
and experiment notes record the exact settings used for the reported metrics.
