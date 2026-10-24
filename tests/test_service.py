from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box, shape

from service import (
    _SPATIAL_CACHE,
    _SPATIAL_CACHE_LOCK,
    _cached_spatial_data,
    _curve_number_grid,
    _decode_nlcd_values,
    _solve_d8_catchment_ponding,
    _latest_daily_stage,
    _navd88_to_lmsl_offset,
    _resample_continuous_grid,
    _spatial_cache_key,
    _scs_runoff_depth_grid,
    _solve_retained_rainfall_surface,
    _validate_request,
    _waterbody_feature_collection,
    bbox_around_point,
    build_flood_payload,
    create_request_handler,
    vectorize_depth_thresholds,
)


class FloodPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        with _SPATIAL_CACHE_LOCK:
            _SPATIAL_CACHE.clear()

    def test_spatial_data_cache_uses_ttl_and_returns_independent_values(self) -> None:
        calls = 0

        def load() -> dict[str, list[int]]:
            nonlocal calls
            calls += 1
            return {"values": [1, 2]}

        first, first_hit, _ = _cached_spatial_data(("test",), 60, load)
        first["values"].append(3)
        second, second_hit, _ = _cached_spatial_data(("test",), 60, load)

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(calls, 1)
        self.assertEqual(second, {"values": [1, 2]})

    def test_cache_key_supports_station_id_sequences(self) -> None:
        self.assertEqual(
            _spatial_cache_key("daily", ("USGS-0123", "USGS-0456")),
            ("daily", ("USGS-0123", "USGS-0456")),
        )

    def test_nlcd_palette_indices_decode_to_landcover_classes(self) -> None:
        values = np.array([[3, 4, 5, 6, 10, 20]], dtype=np.uint8)
        color_map = np.zeros((3, 256), dtype=np.uint16)
        palette = {
            3: (221, 201, 201),
            4: (216, 147, 130),
            5: (237, 0, 0),
            6: (170, 0, 0),
            10: (28, 99, 48),
            20: (186, 216, 234),
        }
        for index, rgb in palette.items():
            color_map[:, index] = np.asarray(rgb, dtype=np.uint16) * 257
        decoded = _decode_nlcd_values(values, color_map)
        np.testing.assert_array_equal(decoded, [[21, 22, 23, 24, 42, 90]])

    def test_nlcd_decoder_rejects_unrecognized_palette_colors(self) -> None:
        values = np.array([[1]], dtype=np.uint8)
        color_map = np.zeros((3, 256), dtype=np.uint16)
        color_map[:, 1] = np.asarray((10, 20, 30), dtype=np.uint16) * 257
        with self.assertRaisesRegex(RuntimeError, "do not match known class colors"):
            _decode_nlcd_values(values, color_map)

    def test_default_contours_include_shallow_flood_depths(self) -> None:
        contours = vectorize_depth_thresholds(
            np.array([[0.0, 0.15]]),
            (0, 0, 2, 1),
            32617,
            water_surface_meters=0.2,
        )
        self.assertEqual(
            [contour["depth_meters"] for contour in contours],
            [0.1, 0.2],
        )

    def test_contours_are_clipped_to_requested_map_extent(self) -> None:
        contours = vectorize_depth_thresholds(
            np.zeros((10, 10)),
            (0, 0, 10, 10),
            32617,
            water_surface_meters=0.2,
            clip_bounds_metric=(2, 2, 8, 8),
        )
        self.assertEqual(
            [contour["depth_meters"] for contour in contours],
            [0.1, 0.2],
        )

    def test_contours_can_be_clipped_to_polygon_geometry(self) -> None:
        clip_geometry = box(500_002, 3_200_002, 500_008, 3_200_008)
        inverse = Transformer.from_crs(
            "EPSG:32617",
            "EPSG:4326",
            always_xy=True,
        )
        forward = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32617",
            always_xy=True,
        )
        contours = vectorize_depth_thresholds(
            np.zeros((10, 10)),
            (500_000, 3_200_000, 500_010, 3_200_010),
            32617,
            water_surface_meters=0.2,
            clip_geometry_metric=clip_geometry,
        )
        self.assertEqual(
            [contour["depth_meters"] for contour in contours],
            [0.1, 0.2],
        )
        for contour in contours:
            for coordinate in contour["coordinates"]:
                point_x, point_y = forward.transform(
                    coordinate["lng"],
                    coordinate["lat"],
                )
                self.assertTrue(
                    clip_geometry.buffer(1e-6).covers(Point(point_x, point_y))
                )
        coordinates = contours[0]["coordinates"]
        self.assertLessEqual(len(coordinates), 15)

    def test_continuous_resampler_smooths_coarse_source_grid(self) -> None:
        result = _resample_continuous_grid(
            np.array([[0.0, 2.0], [2.0, 4.0]]),
            (3, 3),
        )
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[1, 1], 2.0)
        self.assertEqual(result[2, 2], 4.0)

    def test_unknown_soils_use_average_group_c(self) -> None:
        landcover = np.array([[71]], dtype=np.uint8)
        soil = np.array([[2.0]])
        curve_numbers, _ = _curve_number_grid(
            landcover,
            soil,
            np.array([[False]]),
        )
        self.assertEqual(curve_numbers[0, 0], 74)

    def test_landcover_class_fractions_are_averaged_to_finer_grid(self) -> None:
        landcover = np.array([[21, 22]], dtype=np.uint8)
        soil = np.full((1, 3), 2.0)
        curve_numbers, _ = _curve_number_grid(
            landcover,
            soil,
            np.zeros((1, 3), dtype=bool),
        )
        np.testing.assert_allclose(curve_numbers, [[79, 80, 81]])

    def test_d8_routing_pools_interior_catchment_not_boundary_outflow(self) -> None:
        elevations = np.full((5, 5), 5.0)
        elevations[2, 2] = 0.0
        depths, levels, summary = _solve_d8_catchment_ponding(
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

        overflow_depths, _, overflow_summary = _solve_d8_catchment_ponding(
            elevations,
            np.full((5, 5), 100.0),
            1.0,
            None,
        )
        self.assertGreater(overflow_summary["interior_basin_overflow_m3"], 0)
        self.assertLessEqual(float(np.nanmax(overflow_depths)), 5.0)
        self.assertAlmostEqual(overflow_summary["unaccounted_volume_m3"], 0.0)

    def test_vdatum_lmsl_conversion_and_unsupported_coverage(self) -> None:
        session = SimpleNamespace(
            get=lambda *args, **kwargs: SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "region": "CONTIGUOUS",
                    "t_z": "0.125",
                    "uncertainty": "0.04",
                },
            )
        )
        offset, metadata = _navd88_to_lmsl_offset(
            session,
            29.64,
            -82.36,
            "North American Vertical Datum of 1988 (NAVD 88)",
        )
        self.assertEqual(offset, 0.125)
        self.assertEqual(metadata["status"], "converted")

        unsupported_session = SimpleNamespace(
            get=lambda *args, **kwargs: SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"region": "CONTIGUOUS", "t_z": "-999999"},
            )
        )
        offset, metadata = _navd88_to_lmsl_offset(
            unsupported_session,
            29.64,
            -82.36,
            "NAVD88",
        )
        self.assertEqual(offset, 0.0)
        self.assertEqual(metadata["status"], "unsupported_location")

    def test_request_rejects_coordinate_outside_bbox(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside"):
            _validate_request(29.65, -82.34, (-82.35, 29.64, -82.33, 29.646))

    def test_request_accepts_regional_bbox_and_rejects_oversized_bbox(self) -> None:
        _validate_request(27.0, -82.0, (-84.0, 25.0, -80.0, 29.0))
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _validate_request(27.0, -82.0, (-86.0, 24.0, -78.0, 30.0))

    def test_depth_thresholds_create_closed_layered_polygons(self) -> None:
        elevations = np.array(
            [
                [0.0, 0.2],
                [0.8, 1.2],
            ]
        )
        result = vectorize_depth_thresholds(
            elevations,
            (500_000, 3_200_000, 500_002, 3_200_002),
            32617,
            water_surface_meters=1.2,
            thresholds_meters=(0.3, 1.0),
        )
        self.assertEqual([item["depth_meters"] for item in result], [0.3, 1.0])
        for contour in result:
            coordinates = contour["coordinates"]
            self.assertEqual(coordinates[0], coordinates[-1])
            self.assertLessEqual(len(coordinates), 15)
            self.assertTrue(all(set(point) == {"lat", "lng"} for point in coordinates))
        less_deep = shape(
            {
                "type": "Polygon",
                "coordinates": [
                    [(point["lng"], point["lat"]) for point in result[0]["coordinates"]]
                ],
            }
        )
        deeper = shape(
            {
                "type": "Polygon",
                "coordinates": [
                    [(point["lng"], point["lat"]) for point in result[1]["coordinates"]]
                ],
            }
        )
        self.assertTrue(less_deep.buffer(1e-8).covers(deeper))

    def test_retained_rainfall_solver_uses_explicit_volume_fraction(self) -> None:
        level, depths, pixel_area = _solve_retained_rainfall_surface(
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
        _, depths, _ = _solve_retained_rainfall_surface(
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
        surface, depths, _ = _solve_retained_rainfall_surface(
            np.full((2, 2), 10.0),
            (0.0, 0.0, 2.0, 2.0),
            rainfall_inches=10,
            retained_water_fraction=0.02,
            nodata=None,
            water_mask=np.ones((2, 2), dtype=bool),
        )
        self.assertIsNone(surface)
        self.assertFalse(np.isfinite(depths).any())

    def test_curve_numbers_combine_landcover_and_soil_group(self) -> None:
        landcover = np.array([[21, 24, 71, 11]], dtype=np.uint8)
        soil = np.array([[0, 3, 2, 1]], dtype=np.uint8)
        water = np.array([[False, False, False, True]])
        numbers, diagnostics = _curve_number_grid(landcover, soil, water)
        self.assertEqual(numbers[0, 0], 49)
        self.assertEqual(numbers[0, 1], 95)
        self.assertEqual(numbers[0, 2], 74)
        self.assertTrue(np.isnan(numbers[0, 3]))
        self.assertEqual(diagnostics["unmapped_nlcd_classes"], [])

    def test_scs_curve_number_runoff_and_ponding(self) -> None:
        rainfall_inches = 2.0
        curve_numbers = np.full((3, 3), 80.0)
        runoff = _scs_runoff_depth_grid(rainfall_inches, curve_numbers)
        self.assertGreater(float(runoff[0, 0]), 0)
        elevations = np.full((3, 3), 10.0)
        elevations[1, 1] = 0.0
        depths, levels, routing = _solve_d8_catchment_ponding(
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

    def test_waterbody_geojson_is_separate_and_typed(self) -> None:
        coordinates = [
            (500 + 10 * np.cos(angle), 3200 + 10 * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, 40, endpoint=False)
        ]
        circle = Polygon(coordinates)
        bbox = (-82.35, 29.64, -82.34, 29.65)
        with patch(
            "service._fetch_waterbodies",
            return_value=(
                [{
                    "geometry": circle,
                    "name": "Test pond",
                    "source": "test",
                    "feature_id": "pond-1",
                }],
                [],
            ),
        ):
            collection = _waterbody_feature_collection(object(), bbox)
        feature = collection["features"][0]
        self.assertEqual(feature["properties"]["category"], "waterbody")
        self.assertEqual(feature["properties"]["name"], "Test pond")
        self.assertLessEqual(len(feature["geometry"]["coordinates"][0]), 10)
        self.assertEqual(
            collection["metadata"]["category"],
            "mapped_surface_water_not_flood_depth",
        )

    def test_intensity_and_duration_are_converted_to_rainfall_total(self) -> None:
        bbox = (-82.35, 29.64, -82.34, 29.65)
        usgs = SimpleNamespace(status="success", stations=[], latest=[], errors=[])
        groundsource = {
            "source": "Groundsource",
            "records_scanned": 0,
            "malformed_records": 0,
            "events_intersecting_area": 0,
            "events": [],
        }
        with (
            patch(
                "service._fetch_dem",
                return_value=(np.full((2, 2), 34.8), (0, 0, 2, 2), None, 32617),
            ),
            patch("service._bilinear_sample", return_value=34.8),
            patch("service._dem_vertical_datum", return_value=(None, None)),
            patch(
                "service._navd88_to_lmsl_offset",
                return_value=(0.0, {"status": "unsupported_source_datum"}),
            ),
            patch("service.run_usgs", return_value=usgs),
            patch("service._groundsource_event_count", return_value=groundsource),
            patch("service._fetch_waterbodies", return_value=([], [])),
            patch(
                "service._fetch_nlcd_landcover",
                return_value=np.full((2, 2), 71, dtype=np.uint8),
            ),
            patch(
                "service._fetch_ssurgo_hydrologic_soil_grid",
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
            _scs_runoff_depth_grid(2.7, np.full((2, 2), 80.0))[0, 0]
            * 39.3700787402
        )
        self.assertAlmostEqual(
            diagnostics["model"]["effective_rainfall_inches"],
            expected_runoff_inches,
        )

    def test_contour_rejects_nonfinite_surface(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            vectorize_depth_thresholds(
                np.ones((1, 1)),
                (0, 0, 1, 1),
                32617,
                water_surface_meters=float("nan"),
            )

    def test_latest_daily_stage_preserves_observation_date_and_age(self) -> None:
        station = {
            "station_id": "USGS-02240976",
            "name": "Tumblin Creek",
            "distance_from_center_km": 1.465,
        }
        response = {
            "features": [
                {
                    "properties": {
                        "parameter_code": "00065",
                        "value": "2.67",
                        "unit_of_measure": "ft",
                        "time": "1998-09-29",
                    }
                }
            ]
        }
        with patch("service.get_json", return_value=response):
            observations, errors = _latest_daily_stage(object(), [station])
        self.assertEqual(errors, [])
        self.assertEqual(observations[0]["value"], "2.67")
        self.assertEqual(observations[0]["time"], "1998-09-29")
        self.assertGreater(observations[0]["age_days"], 7)

    def test_payload_emits_numeric_assumption_based_payload(self) -> None:
        bbox = (-82.35, 29.64, -82.34, 29.65)
        usgs = SimpleNamespace(
            status="success",
            stations=[],
            latest=[],
            errors=[],
        )
        groundsource = {
            "source": "Groundsource",
            "records_scanned": 2_646_302,
            "malformed_records": 0,
            "events_intersecting_area": 0,
            "events": [],
        }
        with (
            patch(
                "service._fetch_dem",
                return_value=(np.full((2, 2), 34.8), (0, 0, 2, 2), None, 32617),
            ),
            patch("service._bilinear_sample", return_value=34.8),
            patch(
                "service._dem_vertical_datum",
                return_value=("NAVD88", None),
            ),
            patch(
                "service._navd88_to_lmsl_offset",
                return_value=(
                    0.125,
                    {"status": "converted", "applied_offset_meters": 0.125},
                ),
            ),
            patch("service.run_usgs", return_value=usgs),
            patch(
                "service._groundsource_event_count",
                return_value=groundsource,
            ),
            patch("service._fetch_waterbodies", return_value=([], [])),
            patch(
                "service._fetch_nlcd_landcover",
                return_value=np.full((2, 2), 71, dtype=np.uint8),
            ),
            patch(
                "service._fetch_ssurgo_hydrologic_soil_grid",
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


class FloodHttpServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), create_request_handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_post_returns_only_payload(self) -> None:
        payload = {
            "ar_flood_data": {
                "ground_elevation_msl_meters": 34.8,
                "water_surface_msl_meters": 34.805,
                "water_depth_meters": 0.005,
                "is_flooded": True,
            },
            "map_contours": [],
        }
        body = {
            "latitude": 29.645,
            "longitude": -82.345,
            "bbox_wgs84": [-82.35, 29.64, -82.34, 29.65],
            "rainfall_inches": 5,
            "retained_water_fraction": 0.1,
        }
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            with patch(
                "service.build_flood_payload",
                return_value=(payload, {"internal": "diagnostics"}),
            ) as build:
                connection.request(
                    "POST",
                    "/flood",
                    body=json.dumps(body),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                content = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(content, payload)
            build.assert_called_once_with(
                29.645,
                -82.345,
                (-82.35, 29.64, -82.34, 29.65),
                rainfall_inches=5,
                rainfall_intensity_inches_per_hour=None,
                rainfall_duration_hours=None,
                retained_water_fraction=0.1,
            )
        finally:
            connection.close()

    def test_post_rejects_invalid_json_body(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            connection.request(
                "POST",
                "/flood",
                body="not json",
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            content = json.loads(response.read())
            self.assertEqual(response.status, 400)
            self.assertIn("error", content)
        finally:
            connection.close()

    def test_waterbodies_get_returns_geojson_with_distinct_category(self) -> None:
        expected = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"category": "mapped_surface_water_not_flood_depth"},
        }
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            with patch(
                "service._waterbody_feature_collection",
                return_value=expected,
            ) as fetch:
                connection.request(
                    "GET",
                    "/waterbodies?bbox_wgs84=-82.35,29.64,-82.34,29.65",
                )
                response = connection.getresponse()
                content = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(content, expected)
            fetch.assert_called_once()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
