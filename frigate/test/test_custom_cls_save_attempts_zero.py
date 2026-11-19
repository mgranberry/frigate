"""Verify save_attempts <= 0 disables disk writes, deletions, and BGR conversion."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

for mod in (
    "tflite_runtime",
    "tflite_runtime.interpreter",
    "ai_edge_litert",
    "ai_edge_litert.interpreter",
):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from frigate.data_processing.real_time.custom_classification import (  # noqa: E402
    CustomObjectClassificationProcessor,
    write_classification_attempt,
)


class TestCustomClsSaveAttemptsZero(unittest.TestCase):
    def test_write_classification_attempt_zero_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            existing = Path(td) / "keep.webp"
            existing.write_bytes(b"keep")
            write_classification_attempt(
                td,
                np.zeros((16, 16, 3), dtype=np.uint8),
                "obj1",
                1.0,
                "fox",
                0.9,
                max_files=0,
            )
            self.assertEqual(os.listdir(td), ["keep.webp"])

            nonexistent = Path(td) / "missing_dir"
            write_classification_attempt(
                str(nonexistent),
                np.zeros((16, 16, 3), dtype=np.uint8),
                "obj1",
                1.0,
                "fox",
                0.9,
                max_files=0,
            )
            self.assertFalse(nonexistent.exists())

    def test_process_frame_skips_bgr_when_save_attempts_zero(self):
        proc = CustomObjectClassificationProcessor.__new__(
            CustomObjectClassificationProcessor
        )
        proc.model_config = MagicMock()
        proc.model_config.name = "test"
        proc.model_config.save_attempts = 0
        proc.model_config.object_config = MagicMock()
        proc.model_config.object_config.objects = ["animal"]
        proc.metrics = None
        proc.classification_history = {}
        proc.classifications_per_second = MagicMock()
        proc._enqueue_task = MagicMock()

        yuv = np.zeros((240 * 3 // 2, 320), dtype=np.uint8)
        proc.process_frame(
            {
                "id": "o1",
                "camera": "cam",
                "label": "animal",
                "false_positive": False,
                "box": [20, 20, 100, 100],
            },
            yuv,
        )
        crop_bgr = proc._enqueue_task.call_args[0][0][5]
        self.assertIsNone(crop_bgr)
