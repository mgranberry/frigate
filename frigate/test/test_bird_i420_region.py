"""Verify BirdRealTimeProcessor passes RGB dimensions to calculate_region."""

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


class TestBirdI420Region(unittest.TestCase):
    def test_bird_uses_rgb_shape_not_i420_shape(self):
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
        obj_data = {"id": "b1", "label": "bird", "box": [400, 860, 500, 950]}

        with patch(
            "frigate.data_processing.real_time.bird.calculate_region",
            wraps=__import__(
                "frigate.util.image", fromlist=["calculate_region"]
            ).calculate_region,
        ) as mock_calc:
            proc.process_frame(obj_data, yuv)
            self.assertEqual(mock_calc.call_args[0][0], (960, 1280, 3))
