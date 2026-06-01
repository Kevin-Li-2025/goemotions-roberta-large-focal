from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_SOURCE_URL = (
    "https://raw.githubusercontent.com/Kevin-Li-2025/"
    "goemotions-roberta-large-focal/main/emotion-model/train_goemotions.py"
)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in value.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def metric_row(seed: int, output_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    selected = metrics["threshold_selection"]["selected"]
    selected_validation = metrics["selected_validation"]
    selected_test = metrics["selected_test"]
    row = {
        "seed": seed,
        "output_dir": str(output_dir),
        "selected_thresholds": selected,
        "validation_macro_f1": selected_validation["macro_f1"],
        "validation_micro_f1": selected_validation["micro_f1"],
        "validation_samples_f1": selected_validation["samples_f1"],
        "test_macro_f1": selected_test["macro_f1"],
        "test_micro_f1": selected_test["micro_f1"],
        "test_samples_f1": selected_test["samples_f1"],
    }
    if "bootstrap_ci" in selected_test:
        row["test_bootstrap_ci"] = selected_test["bootstrap_ci"]
    if "bootstrap_ci" in selected_validation:
        row["validation_bootstrap_ci"] = selected_validation["bootstrap_ci"]
    return row


def write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        macro_values = [row["test_macro_f1"] for row in rows]
        aggregate = {
            "seeds_completed": [row["seed"] for row in rows],
            "test_macro_f1_mean": sum(macro_values) / len(macro_values),
            "test_macro_f1_min": min(macro_values),
            "test_macro_f1_max": max(macro_values),
        }
    else:
        aggregate = {"seeds_completed": []}
    summary_path.write_text(
        json.dumps({"runs": rows, "aggregate": aggregate}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_training_script() -> Path:
    local_script = Path(__file__).with_name("train_goemotions.py")
    if local_script.exists():
        return local_script

    source_url = os.environ.get("TRAIN_GOEMOTIONS_URL", DEFAULT_TRAIN_SOURCE_URL)
    target = Path(os.environ.get("TRAIN_GOEMOTIONS_SCRIPT", "/kaggle/working/train_goemotions.py"))
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading training script from {source_url} to {target}", flush=True)
    urllib.request.urlretrieve(source_url, target)
    return target


def main() -> None:
    seeds = parse_seeds(os.environ.get("SEED_SWEEP_SEEDS", "43,44"))
    base_output_dir = Path(os.environ.get("SEED_SWEEP_OUTPUT_DIR", "/kaggle/working/goemotions-seed-sweep"))
    bootstrap_samples = os.environ.get("BOOTSTRAP_SAMPLES", "1000")
    script = resolve_training_script()
    summary_path = base_output_dir / "seed_sweep_summary.json"
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        output_dir = base_output_dir / f"seed-{seed}"
        cmd = [
            sys.executable,
            str(script),
            "--model_name",
            "FacebookAI/roberta-large",
            "--output_dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--epochs",
            "4",
            "--learning_rate",
            "1e-5",
            "--train_batch_size",
            "2",
            "--eval_batch_size",
            "16",
            "--gradient_accumulation_steps",
            "16",
            "--loss_type",
            "focal",
            "--focal_alpha",
            "0.38",
            "--focal_gamma",
            "2.8",
            "--threshold_metric",
            "macro_f1",
            "--threshold_coordinate_passes",
            "2",
            "--checkpoint_strategy",
            "no",
            "--no_save_model",
            "--no_save_predictions",
            "--mixed_precision",
            "none",
            "--bootstrap_samples",
            bootstrap_samples,
        ]
        print(f"Running seed {seed}: {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd)
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(metric_row(seed, output_dir, metrics))
        write_summary(summary_path, rows)
        print(f"Wrote partial seed sweep summary to {summary_path}", flush=True)

    print(summary_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
