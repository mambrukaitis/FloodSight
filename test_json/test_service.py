from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from shapely.geometry import shape

from test_json.service import (
    _latest_daily_stage,
    _solve_retained_rainfall_surface,
    _validate_request,
    build_flood_payload,
    create_request_handler,
    reitz_bbox,
    vectorize_depth_thresholds,
)


class FloodPayloadTests(unittest.TestCase):
    def test_reitz_bbox_contains_existing_demo_coordinate(self) -> None:
        west, south, east, north = reitz_bbox()
        self.assertLess(west, -82.34834)
        self.assertGreater(east, -82.34834)
        self.assertLess(south, 29.64489)
        self.assertGreater(north, 29.64489)

    def test_request_rejects_coordinate_outside_bbox(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside"):
            _validate_request(29.65, -82.34, (-82.35, 29.64, -82.33, 29.646))

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
        self.assertTrue(less_deep.covers(deeper))

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
        with patch("test_json.service.get_json", return_value=response):
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
                "test_json.service._fetch_dem",
                return_value=(np.full((2, 2), 34.8), (0, 0, 2, 2), None, 32617),
            ),
            patch("test_json.service._bilinear_sample", return_value=34.8),
            patch(
                "test_json.service._dem_vertical_datum",
                return_value=("North American Vertical Datum of 1988", None),
            ),
            patch("test_json.service.run_usgs", return_value=usgs),
            patch(
                "test_json.service._groundsource_report",
                return_value=(groundsource, []),
            ),
        ):
            payload, diagnostics = build_flood_payload(
                29.645,
                -82.345,
                bbox,
            )
        self.assertEqual(set(payload), {"ar_flood_data", "map_contours"})
        self.assertEqual(
            payload["ar_flood_data"],
            {
                "ground_elevation_msl_meters": 34.8,
                "water_surface_msl_meters": 34.805,
                "water_depth_meters": 0.005,
                "is_flooded": True,
            },
        )
        self.assertEqual(payload["map_contours"], [])
        self.assertEqual(
            diagnostics["status"],
            "illustrative_rainfall_retention_estimate",
        )
        self.assertEqual(
            diagnostics["dem"]["sampled_ground_elevation_meters_unconverted"],
            34.8,
        )
        self.assertEqual(
            diagnostics["model"]["effective_rainfall_inches"],
            0.2,
        )
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
                "test_json.service.build_flood_payload",
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


if __name__ == "__main__":
    unittest.main()
