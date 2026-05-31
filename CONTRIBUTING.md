# Contributing

This repository is focused on a reproducible GoEmotions model release. Keep
changes small, inspectable, and tied to code, documentation, metrics, or public
artifact metadata.

## What Belongs Here

- Training and evaluation code under `emotion-model/`
- Kaggle kernel configuration
- Metrics summaries and experiment records
- README, model card, research notes, and promotion copy
- Small metadata files needed to explain or reproduce the release

## What Does Not Belong Here

- Raw datasets
- Model checkpoints or exported weights
- Kaggle output folders
- Credentials, API tokens, or local config files
- Python caches, notebook checkpoints, and virtual environments
- Unrelated Kaggle projects from the parent workspace

## Validation

Before committing, run the fastest checks that match the change:

```bash
.venv/bin/python -m py_compile emotion-model/train_goemotions.py
.venv/bin/python -m json.tool emotion-model/kernel-metadata.json >/dev/null
.venv/bin/python -m json.tool emotion-model/experiments/2026-05-31-roberta-large-focal-seed42.json >/dev/null
git diff --check -- emotion-model .github CONTRIBUTING.md LICENSE .gitignore
```

For code-path changes, also run the tiny local smoke test documented in
`emotion-model/README.md`.
