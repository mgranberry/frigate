"""Verify BirdRealTimeProcessor computes box size from width and height."""

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

from frigate.data_processing.real_time.bird import BirdRealTimeProcessor  # noqa: E402


class TestBirdBoxDimensions(unittest.TestCase):
    def test_bird_region_uses_width_and_height(self):
        proc = BirdRealTimeProcessor.__new__(BirdRealTimeProcessor)
        proc.interpreter = MagicMock()
        proc.tensor_input_details = [{"index": 0}]
        proc.tensor_output_details = [{"index": 0}]
        proc.interpreter.get_tensor.return_value = [
            np.array([0.1, 0.9], dtype=np.float32)
        ]
        proc.labelmap = {0: "sparrow", 1: "robin"}
        proc.detected_birds = {}
        proc.sub_label_publisher = MagicMock()
        proc.config = MagicMock()
        proc.config.classification.bird.threshold = 0.5

        yuv = np.zeros((960 * 3 // 2, 1280), dtype=np.uint8)
        # Box 100x80 on right side: xmin=900, ymin=100, xmax=1000, ymax=180
        # Buggy formula gave max(-800, -820) = -800 instead of max(100, 80) = 100
        obj_data = {"id": "b1", "label": "bird", "box": [900, 100, 1000, 180]}

        with patch(
            "frigate.data_processing.real_time.bird.calculate_region",
            wraps=__import__(
                "frigate.util.image", fromlist=["calculate_region"]
            ).calculate_region,
        ) as mock_calc:
            proc.process_frame(obj_data, yuv)
            self.assertEqual(mock_calc.call_args[0][5], 110)
