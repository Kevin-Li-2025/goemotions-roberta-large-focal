# Promotion Kit

Use the public Hugging Face page as the primary link because it supports direct
model loading. Use Kaggle as the reproducibility link because it hosts the run
artifact and kernel context.

Primary links:

- Hugging Face: https://huggingface.co/AliceYin/goemotions-roberta-large-focal-sota
- Kaggle: https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42

## One-Line Pitch

RoBERTa-large GoEmotions classifier with focal loss and validation-tuned
thresholds, reaching test macro-F1 0.5330 on the 28-label GoEmotions benchmark.

## Short Launch Post

Released a public GoEmotions emotion-classification model:
RoBERTa-large + focal loss + validation-tuned coordinate thresholds.

Results on the public GoEmotions simplified split:

- Validation macro-F1: 0.5659
- Test macro-F1: 0.5330
- Test micro-F1: 0.5767
- Test samples-F1: 0.5859

The model, tokenizer, thresholds, labels, metrics, and run log are public:
https://huggingface.co/AliceYin/goemotions-roberta-large-focal-sota

Kaggle artifact:
https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42

## X / Twitter

Released a public GoEmotions model: RoBERTa-large + focal loss + tuned
multi-label thresholds.

Test macro-F1: 0.5330 on 28-label GoEmotions.
Includes weights, tokenizer, thresholds, labels, metrics, and run log.

HF: https://huggingface.co/AliceYin/goemotions-roberta-large-focal-sota

## LinkedIn

I released a public GoEmotions emotion-classification model on Hugging Face.

The model is based on RoBERTa-large and uses focal loss plus validation-tuned
coordinate thresholds for multi-label prediction across 27 emotion labels plus
neutral. On the public GoEmotions simplified split, it reaches test macro-F1
0.5330, test micro-F1 0.5767, and test samples-F1 0.5859.

Artifacts include the model weights, tokenizer, thresholds, labels, metrics,
and Kaggle run log so the result can be inspected and reused.

Hugging Face:
https://huggingface.co/AliceYin/goemotions-roberta-large-focal-sota

Kaggle:
https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42

## Reddit / Hacker News

I trained and released a public RoBERTa-large model for GoEmotions, the
28-label multi-label emotion classification dataset from Google Research.

The model uses focal loss for label imbalance and validation-tuned coordinate
thresholds for prediction. On the public simplified split it reaches test
macro-F1 0.5330, micro-F1 0.5767, and samples-F1 0.5859.

The Hugging Face repo includes model weights, tokenizer, thresholds, labels,
metrics, and run logs:
https://huggingface.co/AliceYin/goemotions-roberta-large-focal-sota

The Kaggle model artifact is also public:
https://www.kaggle.com/models/kevin250304/goemotions-roberta-large-focal-sota/Transformers/roberta-large-focal-seed42

I would appreciate feedback on evaluation, threshold calibration, and useful
next experiments for emotion classification.

## Suggested Targets

- Hugging Face model community discussion
- Kaggle model discussion
- r/MachineLearning, using a results-focused title and avoiding inflated claims
- r/LanguageTechnology
- LinkedIn AI/NLP post
- X / Twitter thread with metric table and links
- Papers with Code contribution if the benchmark entry can be represented
  accurately with the public split and metric definition

## Careful Claiming

Use "competitive public GoEmotions result" or "strong public-reference result"
unless a formal leaderboard or paper survey confirms absolute SOTA. The
defensible claim is that this run beats the strongest public model-card
reference found during the experiment by about 0.014 test macro-F1.
