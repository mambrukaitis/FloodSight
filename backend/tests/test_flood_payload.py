"""End-to-end `build_flood_payload` with all remote sources mocked: payload shape, rainfall conversion and diagnostics."""

from __future__ import annotations
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np
from floodsight.flood_payload import build_flood_payload
from floodsight.request_caching import SPATIAL_CACHE
from floodsight.request_caching import SPATIAL_CACHE_LOCK
from floodsight.scs_runoff import scs_runoff_depth_grid


class FloodPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        with SPATIAL_CACHE_LOCK:
            SPATIAL_CACHE.clear()

    def test_intensity_and_duration_are_converted_to_rainfall_total(self) -> None:
        bbox = (-82.35, 29.64, -82.34, 29.65)
        groundsource = {
            "source": "Groundsource",
            "records_scanned": 0,
            "malformed_records": 0,
            "events_intersecting_area": 0,
            "events": [],
        }
        with (
            patch(
                "floodsight.source_fetching.fetch_dem",
                return_value=(np.full((2, 2), 34.8), (0, 0, 2, 2), None, 32617),
            ),
            patch("floodsight.flood_payload.bilinear_sample", return_value=34.8),
            patch("floodsight.vertical_datum_source.dem_vertical_datum", return_value=(None, None)),
            patch(
                "floodsight.vertical_datum_source.navd88_to_lmsl_offset",
                return_value=(0.0, {"status": "unsupported_source_datum"}),
            ),
            patch("floodsight.groundsource_history.groundsource_event_count", return_value=groundsource),
            patch("floodsight.source_fetching.fetch_waterbodies", return_value=([], [])),
            patch(
                "floodsight.source_fetching.fetch_nlcd_landcover",
                return_value=np.full((2, 2), 71, dtype=np.uint8),
            ),
            patch(
                "floodsight.source_fetching.fetch_ssurgo_hydrologic_soil_grid",
                return_value=(
                    np.full((2, 2), 3, dtype=np.uint8),
                    {"samples_with_soil_attributes": 16},
                ),
            ),
        ):
            _, diagnostics = build_flood_payload(
                29.645,
                -82.345,
                bbox,
                rainfall_intensity_inches_per_hour=5.4,
                rainfall_duration_hours=0.5,
            )
        self.assertEqual(diagnostics["request"]["rainfall_inches"], 2.7)
        self.assertEqual(
            diagnostics["request"]["rainfall_intensity_inches_per_hour"],
            5.4,
        )
        self.assertGreaterEqual(
            diagnostics["performance"]["total_build_seconds"],
            diagnostics["performance"]["parallel_source_fetch_wall_seconds"],
        )
        self.assertFalse(diagnostics["performance"]["d8_skipped"])
        self.assertIn(
            "no spatially matched numeric rainfall-flood depths",
            diagnostics["performance"]["why_d8_remains_enabled"],
        )
        expected_runoff_inches = float(
            scs_runoff_depth_grid(2.7, np.full((2, 2), 80.0))[0, 0]
            * 39.3700787402
        )
        self.assertAlmostEqual(
            diagnostics["model"]["effective_rainfall_inches"],
            expected_runoff_inches,
        )

    def test_payload_emits_numeric_assumption_based_payload(self) -> None:
        bbox = (-82.35, 29.64, -82.34, 29.65)
        groundsource = {
            "source": "Groundsource",
            "records_scanned": 2_646_302,
            "malformed_records": 0,
            "events_intersecting_area": 0,
            "events": [],
        }
        with (
            patch(
                "floodsight.source_fetching.fetch_dem",
                return_value=(np.full((2, 2), 34.8), (0, 0, 2, 2), None, 32617),
            ),
            patch("floodsight.flood_payload.bilinear_sample", return_value=34.8),
            patch(
                "floodsight.vertical_datum_source.dem_vertical_datum",
                return_value=("NAVD88", None),
            ),
            patch(
                "floodsight.vertical_datum_source.navd88_to_lmsl_offset",
                return_value=(
                    0.125,
                    {"status": "converted", "applied_offset_meters": 0.125},
                ),
            ),
            patch(
                "floodsight.groundsource_history.groundsource_event_count",
                return_value=groundsource,
            ),
            patch("floodsight.source_fetching.fetch_waterbodies", return_value=([], [])),
            patch(
                "floodsight.source_fetching.fetch_nlcd_landcover",
                return_value=np.full((2, 2), 71, dtype=np.uint8),
            ),
            patch(
                "floodsight.source_fetching.fetch_ssurgo_hydrologic_soil_grid",
                return_value=(
                    np.full((2, 2), 3, dtype=np.uint8),
                    {"samples_with_soil_attributes": 16},
                ),
            ),
        ):
            payload, diagnostics = build_flood_payload(
                29.645,
                -82.345,
                bbox,
                retained_water_fraction=0.02,
            )
        self.assertEqual(set(payload), {"ar_flood_data", "map_contours"})
        self.assertEqual(
            payload["ar_flood_data"],
            {
                "ground_elevation_msl_meters": 34.925,
                "water_surface_msl_meters": 34.93,
                "water_depth_meters": 0.005,
                "is_flooded": True,
                "elevation_datum": "LMSL",
            },
        )
        self.assertEqual(payload["map_contours"], [])
        self.assertEqual(
            diagnostics["status"],
            "screening_scs_curve_number_estimate",
        )
        self.assertEqual(
            diagnostics["dem"]["sampled_ground_elevation_meters_unconverted"],
            34.8,
        )
        self.assertEqual(
            diagnostics["model"]["effective_rainfall_inches"],
            0.2,
        )
        self.assertIn("legacy", diagnostics["model"]["method"])
        self.assertTrue(diagnostics["model"]["notes"])
