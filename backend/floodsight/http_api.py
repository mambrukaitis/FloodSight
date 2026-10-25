"""HTTP request handler for the FloodSight API: GET /health, GET /waterbodies and POST
/flood, with strict input parsing and JSON error responses (400, 404, 500, 503).
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from floodsight.flood_payload import build_flood_payload, build_ar_depth_payload
from floodsight.grid_geometry import validate_request
from floodsight.groundsource_history import GROUND_SOURCE_PATH
from floodsight.waterbody_source import waterbody_feature_collection


MAX_REQUEST_BYTES = 64_000


def create_request_handler():
    """Build the HTTP handler class serving GET /health, GET /waterbodies and POST
    /flood.
    """
    class FloodRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._write_json(
                    200,
                    {
                        "status": "ok",
                        "groundsource_available": GROUND_SOURCE_PATH.is_file(),
                    },
                )
                return
            if parsed.path != "/waterbodies":
                self._write_json(404, {"error": "GET endpoints are /health and /waterbodies."})
                return
            try:
                raw_bbox = parse_qs(parsed.query)["bbox_wgs84"][0]
                bbox = tuple(float(value) for value in raw_bbox.split(","))
                if len(bbox) != 4:
                    raise ValueError("bbox_wgs84 must contain west,south,east,north.")
                west, south, east, north = bbox
                validate_request((south + north) / 2, (west + east) / 2, bbox)
                with requests.Session() as session:
                    collection = waterbody_feature_collection(session, bbox)
                self._write_json(200, collection)
            except (KeyError, TypeError, ValueError) as exc:
                self._write_json(400, {"error": f"Invalid request: {exc}"})
            except (OSError, requests.RequestException, RuntimeError) as exc:
                self._write_json(503, {"error": str(exc)})
            except Exception:
                self._write_json(500, {"error": "Internal server error."})

        def do_POST(self) -> None:

            if self.path == "/ar-depth":
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < content_length <= MAX_REQUEST_BYTES:
                        raise ValueError(f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes.")
                    body = json.loads(self.rfile.read(content_length))
                    if not isinstance(body, dict):
                        raise ValueError("Request body must be a JSON object.")
                    
                    lat = _number(body["latitude"], "latitude")
                    lng = _number(body["longitude"], "longitude")
                    rain = _optional_number(body, "rainfall_inches")
                    if rain is None:
                        rain = 10.0  # Default fallback rainfall

                    # FIX: Create a ~500m surrounding buffer box so water can pool via D8 routing
                    buffer = 0.005
                    ar_bbox = [
                        lng - buffer,  # west
                        lat - buffer,  # south
                        lng + buffer,  # east
                        lat + buffer   # north
                    ]

                    payload, _ = build_ar_depth_payload(
                        latitude=lat,
                        longitude=lng,
                        rainfall_inches=rain,
                    )
                    
                    # Return only the ar_flood_data subset for lightning-fast AR plane anchoring
                    print("Payload keys:", list(payload.keys()), flush=True)

                    if "ar_flood_data" in payload:
                        ar_response = payload["ar_flood_data"]
                    else:
                        ar_response = payload

                    self._write_json(200, ar_response)
                except KeyError as exc:
                    self._write_json(400, {"error": f"Missing required field: {exc.args[0]}"})
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    import traceback
                    traceback.print_exc()
                    self._write_json(400, {"error": f"Invalid request: {exc}"})
                except (OSError, requests.RequestException, RuntimeError) as exc:
                    self._write_json(503, {"error": str(exc)})
                except Exception:
                    self._write_json(500, {"error": "Internal server error."})
                return

            if self.path != "/flood":
                self._write_json(404, {"error": "POST endpoint must be /flood or /ar-depth."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= MAX_REQUEST_BYTES:
                    raise ValueError(
                        f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes."
                    )
                body = json.loads(self.rfile.read(content_length))
                if not isinstance(body, dict):
                    raise ValueError("Request body must be a JSON object.")
                payload, _ = build_flood_payload(
                    _number(body["latitude"], "latitude"),
                    _number(body["longitude"], "longitude"),
                    tuple(_number(value, "bbox_wgs84") for value in body["bbox_wgs84"]),
                    rainfall_inches=_optional_number(body, "rainfall_inches"),
                    rainfall_intensity_inches_per_hour=_optional_number(
                        body, "rainfall_intensity_inches_per_hour"
                    ),
                    rainfall_duration_hours=_optional_number(
                        body, "rainfall_duration_hours"
                    ),
                    retained_water_fraction=_optional_number(
                        body, "retained_water_fraction"
                    ),
                )
                self._write_json(200, payload)
            except KeyError as exc:
                import traceback
                print("MISSING FIELD:", repr(exc), flush=True)
                traceback.print_exc()
                self._write_json(400, {"error": f"Missing required field: {exc.args[0]}"})
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                import traceback
                print("FLOOD ERROR:", repr(exc), flush=True)
                traceback.print_exc()
                self._write_json(400, {"error": f"Invalid request: {exc}"})
            except (OSError, requests.RequestException, RuntimeError) as exc:
                self._write_json(503, {"error": str(exc)})
            except Exception:
                self._write_json(500, {"error": "Internal server error."})

        def _write_json(self, status: int, data: dict[str, Any]) -> None:
            encoded = json.dumps(data, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return FloodRequestHandler


def _number(value: Any, name: str) -> float:
    """Return value if it is a real JSON number (not a bool or string); otherwise raise
    ValueError.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must contain numbers.")
    return value


def _optional_number(body: dict[str, Any], name: str) -> float | None:
    """Return body[name] as a number, or None when the field is absent or null."""
    value = body.get(name)
    return None if value is None else _number(value, name)