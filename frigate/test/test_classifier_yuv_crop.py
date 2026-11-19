"""Verify yuv_crop_to_rgb matches full-frame YUV2RGB + slice/pad."""

import sys
import unittest
from unittest.mock import MagicMock

import cv2
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
    yuv_crop_to_rgb,
)


class TestClassifierYuvCrop(unittest.TestCase):
    def test_pixel_parity_and_edge_cases(self):
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
        full_rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)

        for x, y, x2, y2 in [
            (10, 12, 50, 60),
            (11, 13, 51, 61),
            (-4, -2, 24, 26),
            (140, 100, 175, 135),
        ]:
            got = yuv_crop_to_rgb(yuv, x, y, x2, y2)
            ref = full_rgb[max(0, y) : min(120, y2), max(0, x) : min(160, x2)]
            ref = cv2.copyMakeBorder(
                ref,
                max(0, -y),
                max(0, y2 - 120),
                max(0, -x),
                max(0, x2 - 160),
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            self.assertEqual(got.shape, (y2 - y, x2 - x, 3))
            self.assertTrue(np.array_equal(got, ref))

        for box in [
            (1, 10, 1, 20),
            (10, 5, 20, 5),
            (-10, 10, -2, 20),
            (200, 10, 220, 20),
        ]:
            out = yuv_crop_to_rgb(yuv, *box)
            self.assertEqual(out.ndim, 3)
            self.assertEqual(out.shape[2], 3)
