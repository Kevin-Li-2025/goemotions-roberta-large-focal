from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def ensure_runtime_deps() -> None:
    modules_to_packages = {
        "accelerate": "accelerate",
        "datasets": "datasets",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "sentencepiece": "sentencepiece",
        "torch": "torch",
        "transformers": "transformers",
    }
    missing = [pkg for module, pkg in modules_to_packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        print(f"Installing missing runtime dependencies: {', '.join(missing)}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_runtime_deps()

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GoEmotions multi-label emotion classifier.")
    parser.add_argument("--dataset_name", default="google-research-datasets/go_emotions")
    parser.add_argument("--dataset_config", default="simplified")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large")
    parser.add_argument("--output_dir", default="/kaggle/working/goemotions-deberta-v3-large")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.08)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["auto", "fp16", "bf16", "none"], default="none")
    parser.add_argument("--use_pos_weight", action="store_true", default=True)
    parser.add_argument("--no_pos_weight", dest="use_pos_weight", action="store_false")
    parser.add_argument("--max_pos_weight", type=float, default=20.0)
    parser.add_argument("--threshold_start", type=float, default=0.05)
    parser.add_argument("--threshold_stop", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.02)
    parser.add_argument("--threshold_metric", choices=["macro_f1", "micro_f1", "samples_f1"], default="macro_f1")
    parser.add_argument("--neutral_exclusive", action="store_true", default=True)
    parser.add_argument("--allow_neutral_with_other_labels", dest="neutral_exclusive", action="store_false")
    parser.add_argument("--force_at_least_one_label", action="store_true", default=True)
    parser.add_argument("--allow_empty_predictions", dest="force_at_least_one_label", action="store_false")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--save_predictions", action="store_true", default=True)
    parser.add_argument("--no_save_predictions", dest="save_predictions", action="store_false")
    return parser.parse_args()


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return 1.0 / (1.0 + np.exp(-logits))


def labels_to_multihot(label_lists: list[list[int]], num_labels: int) -> np.ndarray:
    labels = np.zeros((len(label_lists), num_labels), dtype=np.float32)
    for row, label_ids in enumerate(label_lists):
        labels[row, label_ids] = 1.0
    return labels


def limit_split(ds: DatasetDict, split: str, max_samples: int | None) -> DatasetDict:
    if max_samples is None:
        return ds
    n = min(max_samples, len(ds[split]))
    ds[split] = ds[split].select(range(n))
    return ds


def prepare_dataset(raw: DatasetDict, tokenizer: AutoTokenizer, num_labels: int, max_length: int) -> DatasetDict:
    def encode_batch(batch: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length)
        encoded["labels"] = labels_to_multihot(batch["labels"], num_labels).tolist()
        return encoded

    remove_columns = raw["train"].column_names
    return raw.map(encode_batch, batched=True, remove_columns=remove_columns, desc="Tokenizing GoEmotions")


def get_label_names(raw: DatasetDict) -> list[str]:
    return list(raw["train"].features["labels"].feature.names)


def compute_pos_weight(raw_train_labels: list[list[int]], num_labels: int, max_pos_weight: float) -> torch.Tensor:
    y = labels_to_multihot(raw_train_labels, num_labels)
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, max_pos_weight)
    return torch.tensor(pos_weight, dtype=torch.float32)


def apply_thresholds(
    probs: np.ndarray,
    thresholds: np.ndarray | float,
    *,
    force_one: bool,
    neutral_index: int | None,
    neutral_exclusive: bool,
) -> np.ndarray:
    thresholds_arr = np.asarray(thresholds, dtype=np.float32)
    if thresholds_arr.ndim == 0:
        thresholds_arr = np.full(probs.shape[1], float(thresholds_arr), dtype=np.float32)
    preds = probs >= thresholds_arr.reshape(1, -1)

    if neutral_exclusive and neutral_index is not None:
        non_neutral = preds.copy()
        non_neutral[:, neutral_index] = False
        rows_with_emotion = non_neutral.any(axis=1)
        preds[rows_with_emotion, neutral_index] = False

    if force_one:
        empty_rows = ~preds.any(axis=1)
        if empty_rows.any():
            best_labels = probs[empty_rows].argmax(axis=1)
            preds[empty_rows, best_labels] = True

    return preds.astype(np.int32)


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "label_density": float(y_pred.mean()),
    }


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for index, name in enumerate(label_names):
        rows[name] = {
            "f1": float(f1_score(y_true[:, index], y_pred[:, index], zero_division=0)),
            "precision": float(precision_score(y_true[:, index], y_pred[:, index], zero_division=0)),
            "recall": float(recall_score(y_true[:, index], y_pred[:, index], zero_division=0)),
            "support": float(y_true[:, index].sum()),
            "predicted": float(y_pred[:, index].sum()),
        }
    return rows


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(math.floor((stop - start) / step)) + 1
    grid = start + np.arange(count + 1) * step
    return np.round(grid[grid <= stop + 1e-9], 6)


def tune_global_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray,
    *,
    metric: str,
    force_one: bool,
    neutral_index: int | None,
    neutral_exclusive: bool,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    best_score = -1.0
    for threshold in grid:
        preds = apply_thresholds(
            probs,
            float(threshold),
            force_one=force_one,
            neutral_index=neutral_index,
            neutral_exclusive=neutral_exclusive,
        )
        metrics = summarize_metrics(y_true, preds)
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def tune_per_label_thresholds(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for label_index in range(y_true.shape[1]):
        best_score = -1.0
        best_threshold = 0.5
        for threshold in grid:
            label_pred = probs[:, label_index] >= threshold
            score = f1_score(y_true[:, label_index], label_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[label_index] = best_threshold
    return thresholds


class WeightedMultilabelTrainer(Trainer):
    def __init__(self, *args: Any, pos_weight: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fct(logits, labels.to(dtype=torch.float32))
        return (loss, outputs) if return_outputs else loss


def build_training_args(args: argparse.Namespace, use_cuda: bool) -> TrainingArguments:
    use_fp16 = args.mixed_precision == "fp16" and use_cuda
    use_bf16 = args.mixed_precision == "bf16" and use_cuda and torch.cuda.is_bf16_supported()
    strategy = "steps" if args.max_steps > 0 else "epoch"

    kwargs: dict[str, Any] = {
        "output_dir": str(Path(args.output_dir) / "checkpoints"),
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": strategy,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "adamw_torch",
        "report_to": "none",
        "dataloader_pin_memory": use_cuda,
        "seed": args.seed,
        "data_seed": args.seed,
        "do_train": True,
        "do_eval": True,
    }
    if strategy == "steps":
        kwargs["eval_steps"] = args.eval_steps
        kwargs["save_steps"] = args.eval_steps

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = strategy
    else:
        kwargs["evaluation_strategy"] = strategy

    return TrainingArguments(**kwargs)


def compute_metrics_factory(args: argparse.Namespace, neutral_index: int | None) -> Any:
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        probs = sigmoid(logits)
        preds = apply_thresholds(
            probs,
            0.5,
            force_one=args.force_at_least_one_label,
            neutral_index=neutral_index,
            neutral_exclusive=args.neutral_exclusive,
        )
        return summarize_metrics(labels.astype(np.int32), preds)

    return compute_metrics


def save_json(path: Path, value: Any) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n")


def decode_labels(row: np.ndarray, label_names: list[str]) -> str:
    return "|".join(label_names[i] for i, active in enumerate(row) if active)


def write_predictions(
    path: Path,
    raw_split: Any,
    probs: np.ndarray,
    preds: np.ndarray,
    y_true: np.ndarray,
    label_names: list[str],
) -> None:
    data = {
        "id": list(raw_split["id"]) if "id" in raw_split.column_names else list(range(len(preds))),
        "text": list(raw_split["text"]),
        "predicted_labels": [decode_labels(row, label_names) for row in preds],
        "true_labels": [decode_labels(row, label_names) for row in y_true],
    }
    for index, name in enumerate(label_names):
        data[f"prob_{name}"] = probs[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def evaluate_split(
    trainer: Trainer,
    tokenized_split: Any,
    raw_split: Any,
    label_names: list[str],
    thresholds: np.ndarray | float,
    args: argparse.Namespace,
    neutral_index: int | None,
    split_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    prediction = trainer.predict(tokenized_split)
    probs = sigmoid(prediction.predictions)
    y_true = prediction.label_ids.astype(np.int32)
    y_pred = apply_thresholds(
        probs,
        thresholds,
        force_one=args.force_at_least_one_label,
        neutral_index=neutral_index,
        neutral_exclusive=args.neutral_exclusive,
    )
    metrics = summarize_metrics(y_true, y_pred)
    metrics["per_label"] = per_label_metrics(y_true, y_pred, label_names)
    if args.save_predictions:
        write_predictions(output_dir / f"{split_name}_predictions.csv", raw_split, probs, y_pred, y_true, label_names)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_summary = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        device_summary["cuda_device"] = torch.cuda.get_device_name(0)
    print(f"Device summary: {device_summary}", flush=True)

    raw = load_dataset(args.dataset_name, args.dataset_config)
    raw = limit_split(raw, "train", args.max_train_samples)
    raw = limit_split(raw, "validation", args.max_eval_samples)
    raw = limit_split(raw, "test", args.max_test_samples)

    label_names = get_label_names(raw)
    num_labels = len(label_names)
    neutral_index = label_names.index("neutral") if "neutral" in label_names else None
    save_json(output_dir / "labels.json", {"label_names": label_names, "neutral_index": neutral_index})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenized = prepare_dataset(raw, tokenizer, num_labels, args.max_length)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    pos_weight = None
    if args.use_pos_weight:
        pos_weight = compute_pos_weight(list(raw["train"]["labels"]), num_labels, args.max_pos_weight)
        save_json(output_dir / "pos_weight.json", dict(zip(label_names, pos_weight.numpy().tolist(), strict=True)))

    train_args = build_training_args(args, torch.cuda.is_available())
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics_factory(args, neutral_index),
        "pos_weight": pos_weight,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedMultilabelTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    grid = threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    val_prediction = trainer.predict(tokenized["validation"])
    val_probs = sigmoid(val_prediction.predictions)
    val_true = val_prediction.label_ids.astype(np.int32)

    fixed_thresholds = np.full(num_labels, 0.5, dtype=np.float32)
    global_threshold, global_val_metrics = tune_global_threshold(
        val_true,
        val_probs,
        grid,
        metric=args.threshold_metric,
        force_one=args.force_at_least_one_label,
        neutral_index=neutral_index,
        neutral_exclusive=args.neutral_exclusive,
    )
    per_label_thresholds = tune_per_label_thresholds(val_true, val_probs, grid)

    fixed_val_pred = apply_thresholds(
        val_probs,
        fixed_thresholds,
        force_one=args.force_at_least_one_label,
        neutral_index=neutral_index,
        neutral_exclusive=args.neutral_exclusive,
    )
    per_label_val_pred = apply_thresholds(
        val_probs,
        per_label_thresholds,
        force_one=args.force_at_least_one_label,
        neutral_index=neutral_index,
        neutral_exclusive=args.neutral_exclusive,
    )
    fixed_val_metrics = summarize_metrics(val_true, fixed_val_pred)
    per_label_val_metrics = summarize_metrics(val_true, per_label_val_pred)

    threshold_candidates = {
        "fixed_0_5": {"thresholds": fixed_thresholds, "validation": fixed_val_metrics},
        "global": {
            "thresholds": np.full(num_labels, global_threshold, dtype=np.float32),
            "global_threshold": global_threshold,
            "validation": global_val_metrics,
        },
        "per_label": {"thresholds": per_label_thresholds, "validation": per_label_val_metrics},
    }
    selected_name = max(
        threshold_candidates,
        key=lambda name: threshold_candidates[name]["validation"][args.threshold_metric],
    )
    selected_thresholds = threshold_candidates[selected_name]["thresholds"]

    thresholds_payload = {
        "selected": selected_name,
        "selection_metric": args.threshold_metric,
        "fixed_0_5": dict(zip(label_names, fixed_thresholds.tolist(), strict=True)),
        "global": {
            "value": global_threshold,
            "per_label": dict(zip(label_names, threshold_candidates["global"]["thresholds"].tolist(), strict=True)),
        },
        "per_label": dict(zip(label_names, per_label_thresholds.tolist(), strict=True)),
    }
    save_json(output_dir / "thresholds.json", thresholds_payload)

    metrics: dict[str, Any] = {
        "device": device_summary,
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "train_rows": len(raw["train"]),
            "validation_rows": len(raw["validation"]),
            "test_rows": len(raw["test"]),
        },
        "model_name": args.model_name,
        "threshold_selection": {
            "selected": selected_name,
            "metric": args.threshold_metric,
            "validation_candidates": {
                name: payload["validation"] for name, payload in threshold_candidates.items()
            },
        },
    }

    for name, payload in threshold_candidates.items():
        metrics[f"validation_{name}"] = evaluate_split(
            trainer,
            tokenized["validation"],
            raw["validation"],
            label_names,
            payload["thresholds"],
            args,
            neutral_index,
            f"validation_{name}",
            output_dir,
        )
        metrics[f"test_{name}"] = evaluate_split(
            trainer,
            tokenized["test"],
            raw["test"],
            label_names,
            payload["thresholds"],
            args,
            neutral_index,
            f"test_{name}",
            output_dir,
        )

    metrics["selected_validation"] = metrics[f"validation_{selected_name}"]
    metrics["selected_test"] = metrics[f"test_{selected_name}"]
    save_json(output_dir / "metrics.json", metrics)

    selected_test = metrics["selected_test"]
    print(
        "Selected thresholds: "
        f"{selected_name}; validation {args.threshold_metric}="
        f"{metrics['selected_validation'][args.threshold_metric]:.5f}; "
        f"test micro_f1={selected_test['micro_f1']:.5f}; "
        f"test macro_f1={selected_test['macro_f1']:.5f}; "
        f"test samples_f1={selected_test['samples_f1']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
