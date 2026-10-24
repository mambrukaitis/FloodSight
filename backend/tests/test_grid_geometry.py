"""Request validation, bbox handling and grid resampling helpers."""

from __future__ import annotations
from __future__ import annotations
import unittest
import numpy as np
from floodsight.grid_geometry import resample_continuous_grid
from floodsight.grid_geometry import validate_request


class GridGeometryTests(unittest.TestCase):
    def test_continuous_resampler_smooths_coarse_source_grid(self) -> None:
        result = resample_continuous_grid(
            np.array([[0.0, 2.0], [2.0, 4.0]]),
            (3, 3),
        )
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[1, 1], 2.0)
        self.assertEqual(result[2, 2], 4.0)

    def test_request_rejects_coordinate_outside_bbox(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside"):
            validate_request(29.65, -82.34, (-82.35, 29.64, -82.33, 29.646))

    def test_request_accepts_regional_bbox_and_rejects_oversized_bbox(self) -> None:
        validate_request(27.0, -82.0, (-84.0, 25.0, -80.0, 29.0))
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_request(27.0, -82.0, (-86.0, 24.0, -78.0, 30.0))
