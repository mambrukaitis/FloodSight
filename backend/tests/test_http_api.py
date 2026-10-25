"""HTTP API behaviour: /health, /flood and /waterbodies responses, JSON errors and input validation."""

from __future__ import annotations
from __future__ import annotations
import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from backend.floodsight.http_api import create_request_handler


class FloodHttpServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), create_request_handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method: str, path: str, body: object | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            connection.request(
                method,
                path,
                body=None if body is None else json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_health_endpoint(self) -> None:
        status, content = self._request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(content["status"], "ok")

    def test_unknown_paths_return_json_404(self) -> None:
        self.assertEqual(self._request("GET", "/nope")[0], 404)
        status, content = self._request("POST", "/nope", {})
        self.assertEqual(status, 404)
        self.assertIn("error", content)

    def test_missing_and_non_numeric_fields_return_400(self) -> None:
        status, content = self._request("POST", "/flood", {"longitude": -82.3})
        self.assertEqual(status, 400)
        self.assertIn("latitude", content["error"])
        status, _ = self._request(
            "POST",
            "/flood",
            {"latitude": "29.6", "longitude": -82.3, "bbox_wgs84": [-82.4, 29.5, -82.2, 29.7]},
        )
        self.assertEqual(status, 400)

    def test_unexpected_error_returns_json_500(self) -> None:
        body = {
            "latitude": 29.645,
            "longitude": -82.345,
            "bbox_wgs84": [-82.35, 29.64, -82.34, 29.65],
        }
        with patch("floodsight.http_api.build_flood_payload", side_effect=ZeroDivisionError):
            status, content = self._request("POST", "/flood", body)
        self.assertEqual(status, 500)
        self.assertIn("error", content)

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
                "floodsight.http_api.build_flood_payload",
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
                "floodsight.http_api.waterbody_feature_collection",
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
