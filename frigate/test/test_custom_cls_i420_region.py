"""Verify CustomObjectClassificationProcessor passes RGB dimensions to calculate_region."""

import sys
import unittest
from unittest.mock import MagicMock, patch

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
)


class TestCustomClsI420Region(unittest.TestCase):
    def test_bottom_edge_crop_clamped_to_rgb_height(self):
        proc = CustomObjectClassificationProcessor.__new__(
            CustomObjectClassificationProcessor
        )
        proc.model_config = MagicMock()
        proc.model_config.name = "test"
        proc.model_config.object_config = MagicMock()
        proc.model_config.object_config.objects = ["animal"]
        proc.metrics = None
        proc.classification_history = {}
        proc.classifications_per_second = MagicMock()
        proc._enqueue_task = MagicMock()

        yuv = np.zeros((960 * 3 // 2, 1280), dtype=np.uint8)
        obj_data = {
            "id": "obj1",
            "camera": "backyard",
            "label": "animal",
            "false_positive": False,
            "box": [400, 860, 500, 950],
        }

        with patch(
            "frigate.data_processing.real_time.custom_classification.calculate_region",
            wraps=__import__(
                "frigate.util.image", fromlist=["calculate_region"]
            ).calculate_region,
        ) as mock_calc:
            proc.process_frame(obj_data, yuv)
            self.assertEqual(mock_calc.call_args[0][0], (960, 1280, 3))

        resized_crop = proc._enqueue_task.call_args[0][0][4]
        self.assertEqual(resized_crop.shape, (224, 224, 3))
