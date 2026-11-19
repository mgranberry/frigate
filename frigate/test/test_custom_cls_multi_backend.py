"""Verify ONNX, EdgeTPU, TFLite backends, ModelConfig-aligned tensor config, and crop strategies."""

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

from frigate.config.classification import (  # noqa: E402
    CustomClassificationConfig,
    CustomClassificationObjectConfig,
)
from frigate.data_processing.real_time.custom_classification import (  # noqa: E402
    ONNXClassifierBackend,
    TFLiteClassifierBackend,
    _crop_and_resize_object,
    _normalize_scores,
)
from frigate.detectors.detector_config import (  # noqa: E402
    InputDTypeEnum,
    InputTensorEnum,
    PixelFormatEnum,
)


class TestCustomClsMultiBackend(unittest.TestCase):
    def test_config_defaults(self):
        cfg = CustomClassificationConfig(name="wildlife")
        self.assertEqual(cfg.device, "AUTO")
        self.assertEqual(cfg.width, 224)
        self.assertEqual(cfg.height, 224)
        self.assertEqual(cfg.input_tensor, InputTensorEnum.nhwc)
        self.assertEqual(cfg.input_pixel_format, PixelFormatEnum.rgb)
        self.assertEqual(cfg.input_dtype, InputDTypeEnum.int)

    def test_normalize_scores_probs_and_logits(self):
        probs = _normalize_scores(np.array([0.2, 0.8], dtype=np.float32))
        self.assertTrue(np.allclose(probs, [0.2, 0.8]))
        logits = _normalize_scores(np.array([-2.0, 2.0], dtype=np.float32))
        self.assertAlmostEqual(float(logits.sum()), 1.0, places=5)
        self.assertGreater(float(logits[1]), 0.98)

    def test_onnx_backend_explicit_layouts_pixel_format_and_dtypes(self):
        for tensor_layout, px_fmt, dtype_enum, w, h, exp_shape, exp_dtype in [
            (
                InputTensorEnum.nhwc,
                PixelFormatEnum.rgb,
                InputDTypeEnum.int,
                480,
                480,
                (1, 480, 480, 3),
                np.uint8,
            ),
            (
                InputTensorEnum.nchw,
                PixelFormatEnum.bgr,
                InputDTypeEnum.float,
                224,
                224,
                (1, 3, 224, 224),
                np.float32,
            ),
            (
                InputTensorEnum.nhwc,
                PixelFormatEnum.rgb,
                InputDTypeEnum.float_denorm,
                224,
                224,
                (1, 224, 224, 3),
                np.float32,
            ),
        ]:
            fake_session = MagicMock()
            fake_session.get_providers.return_value = ["CPUExecutionProvider"]
            fake_inp = MagicMock()
            fake_inp.name = "images"
            fake_session.get_inputs.return_value = [fake_inp]
            fake_session.run.return_value = [np.array([[1.0, 3.0]], dtype=np.float32)]

            with (
                patch("onnxruntime.InferenceSession", return_value=fake_session),
                patch(
                    "frigate.data_processing.real_time.custom_classification.get_ort_providers",
                    return_value=(["CPUExecutionProvider"], [{}]),
                ),
            ):
                bk = ONNXClassifierBackend(
                    "/tmp/m.onnx",
                    device="CPU",
                    width=w,
                    height=h,
                    input_tensor=tensor_layout,
                    input_pixel_format=px_fmt,
                    input_dtype=dtype_enum,
                )
                self.assertEqual(bk.input_size, (w, h))
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                rgb[:, :, 0] = 255  # pure red in RGB -> channel 2 in BGR
                out = bk.predict(rgb)
                self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
                fed = fake_session.run.call_args[0][1]["images"]
                self.assertEqual(fed.shape, exp_shape)
                self.assertEqual(fed.dtype, exp_dtype)
                if (
                    px_fmt == PixelFormatEnum.bgr
                    and tensor_layout == InputTensorEnum.nchw
                ):
                    self.assertAlmostEqual(float(fed[0, 2, 0, 0]), 1.0, places=5)
                    self.assertAlmostEqual(float(fed[0, 0, 0, 0]), 0.0, places=5)
                elif dtype_enum == InputDTypeEnum.float_denorm:
                    self.assertAlmostEqual(float(fed[0, 0, 0, 0]), 255.0, places=5)

    def test_edgetpu_selection(self):
        with (
            patch(
                "frigate.data_processing.real_time.custom_classification.load_delegate"
            ) as mock_del,
            patch(
                "frigate.data_processing.real_time.custom_classification.Interpreter"
            ) as mock_interp,
        ):
            mock_inst = mock_interp.return_value
            mock_inst.get_input_details.return_value = [
                {"shape": [1, 224, 224, 3], "index": 0}
            ]
            mock_inst.get_output_details.return_value = [{"index": 0}]
            bk = TFLiteClassifierBackend("/tmp/m.tflite", device="edgetpu")
            mock_del.assert_called_once_with("libedgetpu.so.1.0", {})
            self.assertEqual(bk.backend_name, "EdgeTPU")

    def test_crop_strategies(self):
        obj_cfg = CustomClassificationObjectConfig(objects=["animal"])
        self.assertEqual(obj_cfg.crop_strategy, "square")
        self.assertEqual(obj_cfg.crop_scale, 1.0)

        rgb = np.full((400, 600, 3), 200, dtype=np.uint8)
        box = [200, 150, 400, 250]

        sq_crop, sq_res = _crop_and_resize_object(rgb, box, (224, 224), "square", 1.0)
        self.assertEqual(sq_crop.shape[:2], (200, 200))
        self.assertEqual(sq_res.shape, (224, 224, 3))

        tight_crop, tight_res = _crop_and_resize_object(
            rgb, box, (480, 480), "tight", 1.0
        )
        self.assertEqual(tight_crop.shape[:2], (100, 200))
        self.assertEqual(tight_res.shape, (480, 480, 3))

        lb_crop, lb_res = _crop_and_resize_object(
            rgb, box, (224, 224), "letterbox", 1.0
        )
        self.assertEqual(lb_crop.shape[:2], (100, 200))
        self.assertEqual(int(lb_res[0, 112, 0]), 0)
        self.assertEqual(int(lb_res[112, 112, 0]), 200)

        scaled_crop, _ = _crop_and_resize_object(rgb, box, (224, 224), "tight", 1.2)
        self.assertEqual(scaled_crop.shape[:2], (120, 240))
