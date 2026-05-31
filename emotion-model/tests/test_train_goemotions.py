from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_goemotions.py"
SPEC = importlib.util.spec_from_file_location("train_goemotions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
train_goemotions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_goemotions)


class GoEmotionsTrainingUtilitiesTest(unittest.TestCase):
    def test_labels_to_multihot(self) -> None:
        y = train_goemotions.labels_to_multihot([[0, 2], [1]], 4)
        np.testing.assert_array_equal(
            y,
            np.array(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_apply_thresholds_forces_one_label_and_neutral_exclusion(self) -> None:
        probs = np.array(
            [
                [0.91, 0.90, 0.10],
                [0.20, 0.30, 0.40],
            ],
            dtype=np.float32,
        )
        preds = train_goemotions.apply_thresholds(
            probs,
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
            force_one=True,
            neutral_index=0,
            neutral_exclusive=True,
        )
        np.testing.assert_array_equal(
            preds,
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.int32,
            ),
        )

    def test_threshold_grid_and_tuning(self) -> None:
        grid = train_goemotions.threshold_grid(0.05, 0.95, 0.45)
        np.testing.assert_allclose(grid, np.array([0.05, 0.50, 0.95]))

        y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int32)
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32)
        thresholds, metrics = train_goemotions.tune_coordinate_thresholds(
            y_true,
            probs,
            grid,
            initial_thresholds=np.array([0.5, 0.5], dtype=np.float32),
            metric="macro_f1",
            passes=1,
            force_one=True,
            neutral_index=None,
            neutral_exclusive=False,
        )
        self.assertEqual(thresholds.shape, (2,))
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)

    def test_loss_functions_are_finite(self) -> None:
        trainer = train_goemotions.WeightedMultilabelTrainer.__new__(
            train_goemotions.WeightedMultilabelTrainer
        )
        trainer.pos_weight = None
        trainer.focal_gamma = 2.0
        trainer.focal_alpha = 0.38
        trainer.asl_gamma_pos = 0.0
        trainer.asl_gamma_neg = 4.0
        trainer.asl_clip = 0.05
        trainer.loss_eps = 1e-8

        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        for loss in (
            trainer._bce_loss(logits, labels),
            trainer._focal_loss(logits, labels),
            trainer._asymmetric_loss(logits, labels),
        ):
            self.assertTrue(torch.isfinite(loss).item())
            self.assertGreaterEqual(loss.item(), 0.0)

    def test_save_json_handles_numpy_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "payload.json"
            train_goemotions.save_json(path, {"value": np.float32(0.5), "items": np.array([1, 2])})
            text = path.read_text()
            self.assertIn('"value": 0.5', text)
            self.assertIn('"items"', text)
            self.assertTrue(text.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
