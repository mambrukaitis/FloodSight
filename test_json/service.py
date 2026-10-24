from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import requests
import tifffile
from pyproj import Transformer
from shapely import transform
from shapely.geometry import box
from shapely.ops import unary_union

from flood_heartbeat.geo import StudyArea
from flood_heartbeat.common import get_json
from flood_heartbeat.groundsource import REQUIRED_COLUMNS
from flood_heartbeat.usgs import COLLECTIONS_URL, run_usgs
from reitz_2x2_demo import scan_groundsource_box
from reitz_terrain_screen import REITZ_UNION, THREEDEP_EXPORT_URL

THREEDEP_IDENTIFY_URL = THREEDEP_EXPORT_URL.rsplit("/", 1)[0] + "/identify"
GROUND_SOURCE_PATH = Path(".cache/groundsource/groundsource_2026.parquet")
DEFAULT_BBOX_SIDE_METERS = 100.0
USGS_SEARCH_RADIUS_KM = 5.0
MAX_DEM_PIXELS = 250_000
DEPTH_THRESHOLDS_METERS = (0.3, 1.0)


def reitz_bbox(side_meters: float = DEFAULT_BBOX_SIDE_METERS) -> tuple[float, float, float, float]:
    if not math.isfinite(side_meters) or side_meters <= 0:
        raise ValueError("Bounding-box side length must be positive and finite.")
    latitude = REITZ_UNION["latitude"]
    longitude = REITZ_UNION["longitude"]
    metric_crs = _local_utm_epsg(latitude, longitude)
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    inverse = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
    center_x, center_y = forward.transform(longitude, latitude)
    polygon = transform(
        box(
            center_x - side_meters / 2,
            center_y - side_meters / 2,
            center_x + side_meters / 2,
            center_y + side_meters / 2,
        ),
        inverse.transform,
        interleaved=False,
    )
    return polygon.bounds


def _local_utm_epsg(latitude: float, longitude: float) -> int:
    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    return (32600 if latitude >= 0 else 32700) + zone


def _validate_request(
    latitude: float,
    longitude: float,
    bbox_wgs84: tuple[float, float, float, float],
) -> None:
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Latitude must be finite and between -90 and 90.")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Longitude must be finite and between -180 and 180.")
    if len(bbox_wgs84) != 4 or not all(math.isfinite(v) for v in bbox_wgs84):
        raise ValueError("Bounding box must contain four finite WGS84 coordinates.")
    west, south, east, north = bbox_wgs84
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Bounding box must be ordered west, south, east, north.")
    if not (west <= longitude <= east and south <= latitude <= north):
        raise ValueError("The GPS coordinate must lie inside the requested box.")
    if east - west > 2 or north - south > 2:
        raise ValueError("Bounding box is too large for this local demo (maximum 2°).")


def _fetch_dem(
    session: requests.Session,
    latitude: float,
    longitude: float,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float], float | None, int]:
    metric_crs = _local_utm_epsg(latitude, longitude)
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    region_metric = transform(
        box(*bbox_wgs84),
        forward.transform,
        interleaved=False,
    )
    xmin, ymin, xmax, ymax = region_metric.bounds
    width_m = xmax - xmin
    height_m = ymax - ymin
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Bounding box has no area after projection.")
    pixel_size_m = max(1.0, math.sqrt(width_m * height_m / MAX_DEM_PIXELS))
    columns = max(1, math.ceil(width_m / pixel_size_m))
    rows = max(1, math.ceil(height_m / pixel_size_m))
    if rows * columns > MAX_DEM_PIXELS:
        raise ValueError("Requested DEM exceeds the configured pixel limit.")

    response = session.get(
        THREEDEP_EXPORT_URL,
        params={
            "bbox": ",".join(f"{value:.8f}" for value in (xmin, ymin, xmax, ymax)),
            "bboxSR": metric_crs,
            "size": f"{columns},{rows}",
            "imageSR": metric_crs,
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_NearestNeighbor",
            "adjustAspectRatio": "false",
            "f": "image",
        },
        timeout=(10, 90),
    )
    response.raise_for_status()
    with tifffile.TiffFile(io.BytesIO(response.content)) as raster:
        page = raster.pages[0]
        values = page.asarray().astype(float)
        if values.shape != (rows, columns):
            raise RuntimeError(
                f"3DEP returned raster shape {values.shape}; expected {(rows, columns)}."
            )
        scale_tag = page.tags.get("ModelPixelScaleTag")
        tiepoint_tag = page.tags.get("ModelTiepointTag")
        if scale_tag is None or tiepoint_tag is None:
            raise RuntimeError("3DEP TIFF is missing georeferencing tags.")
        scale_x, scale_y = map(float, scale_tag.value[:2])
        i0, j0, _, x0, y0, _ = map(float, tiepoint_tag.value[:6])
        if scale_x <= 0 or scale_y <= 0:
            raise RuntimeError("3DEP TIFF has invalid pixel scale.")
        raster_bounds = (
            x0 - i0 * scale_x,
            y0 - (j0 + rows) * scale_y,
            x0 + (columns - i0) * scale_x,
            y0 - j0 * scale_y,
        )
        nodata_tag = page.tags.get("GDAL_NODATA")
        nodata = float(nodata_tag.value) if nodata_tag else None
        geo_keys = page.tags.get("GeoKeyDirectoryTag")
        if geo_keys is None or metric_crs not in geo_keys.value:
            raise RuntimeError("3DEP raster projection does not match the requested UTM CRS.")
        return values, raster_bounds, nodata, metric_crs


def _bilinear_sample(
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    longitude: float,
    latitude: float,
    metric_crs: int,
    nodata: float | None,
) -> float:
    xmin, ymin, xmax, ymax = bounds
    rows, columns = values.shape
    forward = Transformer.from_crs("EPSG:4326", f"EPSG:{metric_crs}", always_xy=True)
    x, y = forward.transform(longitude, latitude)
    column_position = (x - xmin) / ((xmax - xmin) / columns) - 0.5
    row_position = (ymax - y) / ((ymax - ymin) / rows) - 0.5
    column0 = math.floor(column_position)
    row0 = math.floor(row_position)
    column_fraction = column_position - column0
    row_fraction = row_position - row0
    samples: list[tuple[float, float]] = []
    for row, row_weight in (
        (row0, 1 - row_fraction),
        (row0 + 1, row_fraction),
    ):
        for column, column_weight in (
            (column0, 1 - column_fraction),
            (column0 + 1, column_fraction),
        ):
            if not (0 <= row < rows and 0 <= column < columns):
                continue
            value = float(values[row, column])
            if math.isfinite(value) and (nodata is None or value != nodata):
                samples.append((value, row_weight * column_weight))
    weight_sum = sum(weight for _, weight in samples)
    if weight_sum <= 0:
        raise ValueError("DEM has no valid pixels at the requested GPS coordinate.")
    return sum(value * weight for value, weight in samples) / weight_sum


def vectorize_depth_thresholds(
    elevation_grid_meters: np.ndarray,
    bounds_metric: tuple[float, float, float, float],
    metric_crs: int,
    water_surface_meters: float,
    *,
    nodata: float | None = None,
    thresholds_meters: tuple[float, ...] = DEPTH_THRESHOLDS_METERS,
) -> list[dict[str, Any]]:
    """Vectorize a constant water surface against a DEM in the same vertical datum."""
    if not math.isfinite(water_surface_meters):
        raise ValueError("Water-surface elevation must be finite.")
    xmin, ymin, xmax, ymax = bounds_metric
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Projected raster bounds must have positive area.")
    values = np.asarray(elevation_grid_meters, dtype=float)
    if values.ndim != 2:
        raise ValueError("Elevation grid must be two-dimensional.")
    if any(not math.isfinite(value) or value <= 0 for value in thresholds_meters):
        raise ValueError("Depth thresholds must be positive finite values.")
    rows, columns = values.shape
    pixel_width = (xmax - xmin) / columns
    pixel_height = (ymax - ymin) / rows
    forward = Transformer.from_crs(f"EPSG:{metric_crs}", "EPSG:4326", always_xy=True)
    contours: list[dict[str, Any]] = []
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    depth = np.maximum(water_surface_meters - values, 0)
    for threshold in sorted(set(thresholds_meters)):
        row_indices, column_indices = np.nonzero(valid & (depth >= threshold))
        if not len(row_indices):
            continue
        pixels = [
            box(
                xmin + column * pixel_width,
                ymax - (row + 1) * pixel_height,
                xmin + (column + 1) * pixel_width,
                ymax - row * pixel_height,
            )
            for row, column in zip(row_indices, column_indices)
        ]
        dissolved = unary_union(pixels)
        polygons = [dissolved] if dissolved.geom_type == "Polygon" else list(dissolved.geoms)
        for polygon in polygons:
            if polygon.geom_type != "Polygon" or polygon.interiors:
                raise ValueError(
                    "Output schema cannot represent contour holes or non-polygon parts."
                )
            geographic = transform(
                polygon,
                forward.transform,
                interleaved=False,
            )
            coordinates = [
                {"lat": latitude, "lng": longitude}
                for longitude, latitude in geographic.exterior.coords
            ]
            contours.append(
                {
                    "depth_meters": threshold,
                    "coordinates": coordinates,
                }
            )
    return contours


def _dem_vertical_datum(
    session: requests.Session,
    latitude: float,
    longitude: float,
    metric_crs: int,
) -> tuple[str | None, str | None]:
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    x, y = forward.transform(longitude, latitude)
    try:
        response = session.get(
            THREEDEP_IDENTIFY_URL,
            params={
                "geometry": json.dumps(
                    {"x": x, "y": y, "spatialReference": {"wkid": metric_crs}}
                ),
                "geometryType": "esriGeometryPoint",
                "sr": metric_crs,
                "returnGeometry": "false",
                "returnCatalogItems": "true",
                "f": "json",
            },
            timeout=(10, 45),
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return None, f"Could not inspect 3DEP vertical-datum metadata: {exc}"
    catalog = data.get("catalogItems", {}).get("features", [])
    datums = {
        str(feature.get("attributes", {}).get("VerticalDatum")).strip()
        for feature in catalog
        if feature.get("attributes", {}).get("VerticalDatum")
    }
    if len(datums) == 1:
        return next(iter(datums)), None
    if not datums:
        return None, "3DEP source catalog reports no vertical datum for the sampled tile."
    return None, f"3DEP source tiles report inconsistent vertical datums: {sorted(datums)}."


def _latest_daily_stage(
    session: requests.Session,
    stream_stations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    endpoint = f"{COLLECTIONS_URL}/latest-daily/items"
    observations = []
    errors = []
    for station in stream_stations:
        try:
            data = get_json(
                session,
                endpoint,
                {
                    "f": "json",
                    "limit": 100,
                    "monitoring_location_id": station["station_id"],
                    "parameter_code": "00065",
                },
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            errors.append(f"Latest daily stage for {station['station_id']}: {exc}")
            continue
        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            if str(properties.get("parameter_code", "")) != "00065":
                continue
            observed_at = str(properties.get("time", ""))
            try:
                observed_date = datetime.fromisoformat(observed_at[:10]).date()
                age_days = (datetime.now(timezone.utc).date() - observed_date).days
            except ValueError:
                age_days = None
            observations.append(
                {
                    "station_id": station["station_id"],
                    "station_name": station["name"],
                    "distance_km": station["distance_from_center_km"],
                    "value": properties.get("value"),
                    "unit": properties.get("unit_of_measure"),
                    "time": observed_at,
                    "age_days": age_days,
                    "approval_status": properties.get("approval_status"),
                    "vertical_datum": properties.get("vertical_datum_name")
                    or properties.get("vertical_datum"),
                }
            )
    observations.sort(
        key=lambda item: (item["time"], item["distance_km"]),
        reverse=True,
    )
    return observations, errors


def _groundsource_report(
    bbox_wgs84: tuple[float, float, float, float],
    parquet_path: Path,
    cache_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Groundsource Parquet file not found: {parquet_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        json.dumps(
            {
                "bounds": [round(value, 12) for value in bbox_wgs84],
                "path": str(parquet_path.resolve()),
                "size": parquet_path.stat().st_size,
                "mtime_ns": parquet_path.stat().st_mtime_ns,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    scan_path = cache_dir / f"groundsource_{key}.json"
    expected_bounds = list(bbox_wgs84)
    if scan_path.is_file():
        data = json.loads(scan_path.read_text(encoding="utf-8"))
        if (
            data.get("query_bounds_wgs84") == expected_bounds
            and data.get("parquet_path") == str(parquet_path)
            and data.get("parquet_size_bytes") == parquet_path.stat().st_size
            and data.get("parquet_mtime_ns") == parquet_path.stat().st_mtime_ns
        ):
            return data, [f"Groundsource scan cache hit: {scan_path}"]

    polygon = box(*bbox_wgs84)
    report = scan_groundsource_box(
        parquet_path,
        [{"id": "bbox", "geometry": polygon}],
        scan_path,
    )
    report["parquet_size_bytes"] = parquet_path.stat().st_size
    report["parquet_mtime_ns"] = parquet_path.stat().st_mtime_ns
    scan_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, [
        f"Groundsource full archive scan completed; required schema: "
        f"{sorted(REQUIRED_COLUMNS)}."
    ]


def build_flood_payload(
    latitude: float,
    longitude: float,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    groundsource_parquet: Path = GROUND_SOURCE_PATH,
    cache_dir: Path = Path(__file__).parent / "cache",
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the iOS payload and a separate evidence/availability report.

    Unknown MSL elevations and flood state are JSON null, not guessed values or
    false/dry results. Groundsource and USGS observations are supporting evidence;
    neither is treated as a spatial depth raster.
    """
    _validate_request(latitude, longitude, bbox_wgs84)
    session = session or requests.Session()
    dem_values, dem_bounds, dem_nodata, metric_crs = _fetch_dem(
        session, latitude, longitude, bbox_wgs84
    )
    dem_sample = _bilinear_sample(
        dem_values,
        dem_bounds,
        longitude,
        latitude,
        metric_crs,
        dem_nodata,
    )
    dem_datum, datum_note = _dem_vertical_datum(
        session, latitude, longitude, metric_crs
    )
    usgs_result = run_usgs(
        StudyArea(
            latitude=latitude,
            longitude=longitude,
            radius_km=USGS_SEARCH_RADIUS_KM,
        ),
        session=session,
    )
    streams = [
        station
        for station in usgs_result.stations
        if station["site_type_code"] == "ST"
        or "stream" in str(station["site_type"]).lower()
        or "river" in str(station["site_type"]).lower()
    ]
    nearest_stream = streams[0] if streams else None
    stage_observations = [
        observation
        for observation in usgs_result.latest
        if observation["parameter_code"] == "00065"
    ]
    latest_daily, daily_errors = _latest_daily_stage(session, streams)
    groundsource, scan_notes = _groundsource_report(
        bbox_wgs84,
        groundsource_parquet,
        cache_dir,
    )

    reasons = []
    recent_daily = [
        observation
        for observation in latest_daily
        if observation["age_days"] is not None
        and 0 <= observation["age_days"] <= 7
    ]
    if not stage_observations and not recent_daily:
        latest_daily_note = (
            " Latest daily value was "
            f"{latest_daily[0]['value']} {latest_daily[0]['unit']} at "
            f"{latest_daily[0]['station_name']} on {latest_daily[0]['time']}."
            if latest_daily
            else ""
        )
        reasons.append(
            "No current USGS WDFN gage-height observation (parameter 00065) was "
            "returned for a stream station within the 5 km search radius."
            + latest_daily_note
        )
    reasons.append(
        "A streamgage stage is local to its gauge; it is not a spatial water surface. "
        "No verified gage-datum elevation or hydraulic model connects it to this box."
    )
    is_dem_msl = bool(
        dem_datum
        and (
            dem_datum.strip().lower() == "msl"
            or "mean sea level" in dem_datum.strip().lower()
        )
    )
    if not is_dem_msl:
        reasons.append(
            f"The 3DEP elevation is referenced to {dem_datum or 'an unknown datum'}, "
            "not verified MSL; no vertical-datum conversion was applied."
        )
    if groundsource.get("events_intersecting_area", 0):
        reasons.append(
            "Groundsource contributes historical event footprints only; it does not "
            "contain point water levels or depth measurements."
        )

    payload = {
        "ar_flood_data": {
            "ground_elevation_msl_meters": dem_sample if is_dem_msl else None,
            "water_surface_msl_meters": None,
            "water_depth_meters": None,
            "is_flooded": None,
        },
        "map_contours": [],
    }
    diagnostics = {
        "status": "insufficient_observations_for_flood_surface",
        "request": {
            "latitude": latitude,
            "longitude": longitude,
            "bbox_wgs84_west_south_east_north": list(bbox_wgs84),
        },
        "dem": {
            "source": THREEDEP_EXPORT_URL,
            "metric_crs": f"EPSG:{metric_crs}",
            "raster_shape_rows_columns": list(dem_values.shape),
            "pixel_resolution_meters": [
                (dem_bounds[2] - dem_bounds[0]) / dem_values.shape[1],
                (dem_bounds[3] - dem_bounds[1]) / dem_values.shape[0],
            ],
            "sampled_ground_elevation_meters_unconverted": dem_sample,
            "vertical_datum": dem_datum,
            "vertical_datum_note": datum_note,
        },
        "usgs_wdfn_v1": {
            "status": usgs_result.status,
            "endpoint": COLLECTIONS_URL,
            "search_radius_km": USGS_SEARCH_RADIUS_KM,
            "nearest_stream_station": (
                {
                    "station_id": nearest_stream["station_id"],
                    "name": nearest_stream["name"],
                    "distance_km": nearest_stream["distance_from_center_km"],
                    "site_vertical_datum": nearest_stream["metadata_json"].get(
                        "vertical_datum_name"
                    )
                    or nearest_stream["metadata_json"].get("vertical_datum"),
                }
                if nearest_stream
                else None
            ),
            "available_gage_height_observations": stage_observations,
            "latest_daily_gage_height_observations": latest_daily,
            "errors": usgs_result.errors + daily_errors,
        },
        "groundsource": {
            "source": groundsource.get("source"),
            "records_scanned": groundsource.get("records_scanned"),
            "malformed_records": groundsource.get("malformed_records"),
            "historical_footprints_intersecting_bbox": groundsource.get(
                "events_intersecting_area"
            ),
            "event_date_range": (
                [
                    min(event["start_date"] for event in groundsource["events"]),
                    max(event["end_date"] for event in groundsource["events"]),
                ]
                if groundsource.get("events")
                else None
            ),
            "available_fields": sorted(REQUIRED_COLUMNS),
            "notes": scan_notes,
        },
        "withheld_output_reasons": reasons,
        "interpretation": (
            "Null AR fields mean unknown, not dry. No depth raster or contours "
            "were emitted because the available sources do not establish a local "
            "water-surface elevation in a verified common vertical datum."
        ),
    }
    return payload, diagnostics


def write_reitz_example(output_dir: Path) -> tuple[Path, Path]:
    bbox = reitz_bbox()
    payload, diagnostics = build_flood_payload(
        REITZ_UNION["latitude"],
        REITZ_UNION["longitude"],
        bbox,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "reitz_union_flood.json"
    diagnostics_path = output_dir / "reitz_union_diagnostics.json"
    payload_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload_path, diagnostics_path


def create_request_handler():
    class FloodRequestHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/flood":
                self.send_error(404, "POST endpoint must be /flood.")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 64_000:
                    raise ValueError("Request body must be between 1 and 64000 bytes.")
                body = json.loads(self.rfile.read(content_length))
                if not isinstance(body, dict):
                    raise ValueError("Request body must be a JSON object.")
                latitude = body["latitude"]
                longitude = body["longitude"]
                bbox = tuple(body["bbox_wgs84"])
                payload, _ = build_flood_payload(latitude, longitude, bbox)
                self._write_json(200, payload)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self._write_json(400, {"error": str(exc)})
            except (OSError, requests.RequestException, RuntimeError) as exc:
                self._write_json(503, {"error": str(exc)})

        def _write_json(self, status: int, data: dict[str, Any]) -> None:
            encoded = json.dumps(data, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return FloodRequestHandler


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), create_request_handler())
    print(f"Flood JSON service listening at http://{host}:{port}/flood")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Flood JSON service.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Reitz Union Swift flood JSON and diagnostics."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the HTTP POST /flood microservice instead of writing files.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
    )
    args = parser.parse_args()
    if args.serve:
        serve(args.host, args.port)
        return
    payload_path, diagnostics_path = write_reitz_example(args.output_dir)
    print(f"Wrote payload: {payload_path}")
    print(f"Wrote diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
