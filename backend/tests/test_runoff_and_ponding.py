"""SCS curve-number runoff, D8 catchment ponding and the legacy retained-rainfall solver."""

from __future__ import annotations
from __future__ import annotations
import unittest
import numpy as np
from floodsight.scs_runoff import curve_number_grid
from floodsight.scs_runoff import scs_runoff_depth_grid
from floodsight.terrain_ponding_solver import solve_d8_catchment_ponding
from floodsight.terrain_ponding_solver import solve_retained_rainfall_surface


class RunoffAndPondingTests(unittest.TestCase):
    def test_landcover_class_fractions_are_averaged_to_finer_grid(self) -> None:
        landcover = np.array([[21, 22]], dtype=np.uint8)
        soil = np.full((1, 3), 2.0)
        curve_numbers, _ = curve_number_grid(
            landcover,
            soil,
            np.zeros((1, 3), dtype=bool),
        )
        np.testing.assert_allclose(curve_numbers, [[79, 80, 81]])

    def test_curve_numbers_combine_landcover_and_soil_group(self) -> None:
        landcover = np.array([[21, 24, 71, 11]], dtype=np.uint8)
        soil = np.array([[0, 3, 2, 1]], dtype=np.uint8)
        water = np.array([[False, False, False, True]])
        numbers, diagnostics = curve_number_grid(landcover, soil, water)
        self.assertEqual(numbers[0, 0], 49)
        self.assertEqual(numbers[0, 1], 95)
        self.assertEqual(numbers[0, 2], 74)
        self.assertTrue(np.isnan(numbers[0, 3]))
        self.assertEqual(diagnostics["unmapped_nlcd_classes"], [])

    def test_scs_curve_number_runoff_and_ponding(self) -> None:
        rainfall_inches = 2.0
        curve_numbers = np.full((3, 3), 80.0)
        runoff = scs_runoff_depth_grid(rainfall_inches, curve_numbers)
        self.assertGreater(float(runoff[0, 0]), 0)
        elevations = np.full((3, 3), 10.0)
        elevations[1, 1] = 0.0
        depths, levels, routing = solve_d8_catchment_ponding(
            elevations,
            runoff,
            1.0,
            None,
        )
        self.assertIsNotNone(levels)
        self.assertGreater(depths[1, 1], 0)
        self.assertAlmostEqual(
            routing["routed_runoff_volume_m3"],
            float(runoff.sum()),
        )

    def test_d8_routing_pools_interior_catchment_not_boundary_outflow(self) -> None:
        elevations = np.full((5, 5), 5.0)
        elevations[2, 2] = 0.0
        depths, levels, summary = solve_d8_catchment_ponding(
            elevations,
            np.ones((5, 5)),
            1.0,
            None,
        )
        self.assertGreater(depths[2, 2], 0)
        self.assertIsNotNone(levels)
        self.assertEqual(summary["routed_runoff_volume_m3"], 25.0)
        self.assertGreater(summary["boundary_outflow_volume_m3"], 0)
        self.assertAlmostEqual(summary["unaccounted_volume_m3"], 0.0)

        overflow_depths, _, overflow_summary = solve_d8_catchment_ponding(
            elevations,
            np.full((5, 5), 100.0),
            1.0,
            None,
        )
        self.assertGreater(overflow_summary["interior_basin_overflow_m3"], 0)
        self.assertLessEqual(float(np.nanmax(overflow_depths)), 5.0)
        self.assertAlmostEqual(overflow_summary["unaccounted_volume_m3"], 0.0)

    def test_retained_rainfall_solver_uses_explicit_volume_fraction(self) -> None:
        level, depths, pixel_area = solve_retained_rainfall_surface(
            np.full((10, 10), 10.0),
            (0.0, 0.0, 10.0, 10.0),
            rainfall_inches=10,
            retained_water_fraction=0.02,
            nodata=None,
        )
        self.assertAlmostEqual(pixel_area, 1.0)
        self.assertAlmostEqual(level - 10.0, 0.00508, places=8)
        self.assertTrue(np.allclose(depths, 0.00508))

    def test_retained_rainfall_solver_excludes_mapped_water(self) -> None:
        water_mask = np.array([[True, False], [False, False]])
        _, depths, _ = solve_retained_rainfall_surface(
            np.full((2, 2), 10.0),
            (0.0, 0.0, 2.0, 2.0),
            rainfall_inches=10,
            retained_water_fraction=0.02,
            nodata=None,
            water_mask=water_mask,
        )
        self.assertTrue(np.isnan(depths[0, 0]))
        self.assertTrue(np.allclose(depths[~water_mask], 0.00508))

    def test_entirely_water_bbox_has_no_rainfall_depth_surface(self) -> None:
        surface, depths, _ = solve_retained_rainfall_surface(
            np.full((2, 2), 10.0),
            (0.0, 0.0, 2.0, 2.0),
            rainfall_inches=10,
            retained_water_fraction=0.02,
            nodata=None,
            water_mask=np.ones((2, 2), dtype=bool),
        )
        self.assertIsNone(surface)
        self.assertFalse(np.isfinite(depths).any())
