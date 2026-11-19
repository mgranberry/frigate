"""Verify RKNN backend selection and inference for custom classification."""

import os
import sys
import tempfile
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

from frigate.config.classification import CustomClassificationConfig  # noqa: E402
from frigate.data_processing.real_time.custom_classification import (  # noqa: E402
    RKNNClassifierBackend,
    load_classification_backend,
)


class TestCustomClsRknn(unittest.TestCase):
    def test_rknn_model_loading_and_predict(self):
        fake_runner = MagicMock()
        fake_runner.get_input_names.return_value = ["input"]
        fake_runner.run.return_value = [np.array([[0.1, 2.5]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "labelmap.txt"), "w").write("a\nb\n")
            open(os.path.join(td, "model.rknn"), "wb").write(b"")
            cfg = CustomClassificationConfig(name="test", device="RKNN")
            with patch(
                "frigate.detectors.detection_runners.RKNNModelRunner",
                return_value=fake_runner,
            ):
                bk, lm = load_classification_backend(td, cfg)
                self.assertIsInstance(bk, RKNNClassifierBackend)
                self.assertEqual(lm, {0: "a", 1: "b"})
                out = bk.predict(np.zeros((224, 224, 3), dtype=np.uint8))
                self.assertGreater(float(out[1]), float(out[0]))
