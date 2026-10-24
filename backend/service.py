"""FloodSight flood payload builder and HTTP API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests
from pyproj import Transformer
from shapely import transform
from shapely.geometry import Point, box

from hydrology import (
    DEPTH_THRESHOLDS_METERS,
    FALLBACK_NLCD_CLASS,
    _analysis_domain,
    _bilinear_sample,
    _curve_number_grid,
    _local_utm_epsg,
    _resample_continuous_grid,
    _scs_runoff_depth_grid,
    _solve_d8_catchment_ponding,
    _solve_retained_rainfall_surface,
    _validate_request,
    _waterbody_raster_mask,
    vectorize_depth_thresholds,
)
from data_sources import (
    _groundsource_florida_events,
    FALLBACK_HYDROLOGIC_SOIL_GROUP,
    GROUNDSOURCE_BUFFER_DEGREES,
    GROUND_SOURCE_PATH,
    MRLC_NLCD_LAYER,
    MRLC_WMS_URL,
    SSURGO_QUERY_URL,
    THREEDEP_EXPORT_URL,
    _buffered_bbox,
    _cached_source_job,
    _datum_and_offset,
    _dem_grid_shape,
    _fetch_dem,
    _fetch_nlcd_landcover,
    _fetch_ssurgo_hydrologic_soil_grid,
    _fetch_waterbodies,
    _groundsource_job,
    _spatial_cache_key,
    _waterbody_feature_collection,
)

USGS_SEARCH_RADIUS_KM = 5.0

MAX_PARALLEL_DATA_REQUESTS = 6

DEM_CACHE_TTL_SECONDS = 86_400
NLCD_CACHE_TTL_SECONDS = 30 * 86_400
SSURGO_CACHE_TTL_SECONDS = 7 * 86_400
WATERBODY_CACHE_TTL_SECONDS = 86_400
VDATUM_CACHE_TTL_SECONDS = 30 * 86_400
USGS_CACHE_TTL_SECONDS = 300

DEFAULT_RAINFALL_INCHES = 10.0

def build_flood_payload(
    latitude: float,
    longitude: float,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    rainfall_inches: float | None = None,
    rainfall_intensity_inches_per_hour: float | None = None,
    rainfall_duration_hours: float | None = None,
    retained_water_fraction: float | None = None,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a screening rainfall-runoff estimate and source diagnostics for any Florida coordinate."""
    request_started = perf_counter()
    _validate_request(latitude, longitude, bbox_wgs84)
    has_intensity = rainfall_intensity_inches_per_hour is not None
    has_duration = rainfall_duration_hours is not None
    if has_intensity != has_duration:
        raise ValueError(
            "Rainfall intensity and duration must be supplied together."
        )
    if has_intensity:
        if rainfall_inches is not None:
            raise ValueError(
                "Provide rainfall_inches or intensity plus duration, not both."
            )
        if (
            not math.isfinite(rainfall_intensity_inches_per_hour)
            or rainfall_intensity_inches_per_hour < 0
        ):
            raise ValueError("Rainfall intensity must be finite and non-negative.")
        if (
            not math.isfinite(rainfall_duration_hours)
            or rainfall_duration_hours <= 0
        ):
            raise ValueError("Rainfall duration must be positive and finite.")
        rainfall_inches = (
            rainfall_intensity_inches_per_hour * rainfall_duration_hours
        )
    elif rainfall_inches is None:
        rainfall_inches = DEFAULT_RAINFALL_INCHES
    if not math.isfinite(rainfall_inches) or rainfall_inches < 0:
        raise ValueError("Rainfall must be finite and non-negative.")
    if retained_water_fraction is not None and (
        not math.isfinite(retained_water_fraction)
        or not 0 <= retained_water_fraction <= 1
    ):
        raise ValueError("Retained-water fraction must be between 0 and 1.")
    metric_crs = _local_utm_epsg(latitude, longitude)
    forward = Transformer.from_crs(
        "EPSG:4326",
        metric_crs,
        always_xy=True,
    )
    requested_geographic = box(*bbox_wgs84).segmentize(0.0001)
    requested_projected = transform(
        requested_geographic,
        forward.transform,
        interleaved=False,
    )
    requested_metric_bounds = requested_projected.bounds
    analysis_bbox_wgs84, analysis_metric_bounds = _analysis_domain(
        bbox_wgs84,
        metric_crs,
    )
    predicted_dem_shape = _dem_grid_shape(analysis_bbox_wgs84, metric_crs)
    source_timings: dict[str, Any] = {}
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DATA_REQUESTS) as executor:
        futures = {
            "dem": executor.submit(
                _cached_source_job,
                "3DEP elevation",
                _spatial_cache_key(
                    "dem",
                    metric_crs,
                    analysis_metric_bounds,
                    predicted_dem_shape,
                ),
                DEM_CACHE_TTL_SECONDS,
                session,
                lambda active_session: _fetch_dem(
                    active_session,
                    latitude,
                    longitude,
                    analysis_bbox_wgs84,
                ),
            ),
            "datum": executor.submit(
                _cached_source_job,
                "3DEP vertical datum + NOAA VDatum offset",
                _spatial_cache_key(
                    "dem_datum_offset",
                    metric_crs,
                    round(latitude, 5),
                    round(longitude, 5),
                ),
                VDATUM_CACHE_TTL_SECONDS,
                session,
                lambda active_session: _datum_and_offset(
                    active_session,
                    latitude,
                    longitude,
                    metric_crs,
                ),
            ),
            "groundsource": executor.submit(
                _groundsource_job,
                _buffered_bbox(analysis_bbox_wgs84, GROUNDSOURCE_BUFFER_DEGREES),
                GROUND_SOURCE_PATH,
            ),
            "waterbodies": executor.submit(
                _cached_source_job,
                "surface-water polygons",
                _spatial_cache_key(
                    "waterbodies",
                    analysis_bbox_wgs84,
                    metric_crs,
                ),
                WATERBODY_CACHE_TTL_SECONDS,
                session,
                lambda active_session: _fetch_waterbodies(
                    active_session,
                    analysis_bbox_wgs84,
                    metric_crs,
                    analysis_metric_bounds,
                ),
            ),
            "nlcd": executor.submit(
                _cached_source_job,
                "MRLC NLCD",
                _spatial_cache_key(
                    "nlcd",
                    analysis_metric_bounds,
                    metric_crs,
                ),
                NLCD_CACHE_TTL_SECONDS,
                session,
                lambda active_session: _fetch_nlcd_landcover(
                    active_session,
                    analysis_metric_bounds,
                    metric_crs,
                    predicted_dem_shape,
                ),
            ),
            "soils": executor.submit(
                _cached_source_job,
                "USDA SSURGO",
                _spatial_cache_key(
                    "ssurgo",
                    analysis_bbox_wgs84,
                    predicted_dem_shape,
                ),
                SSURGO_CACHE_TTL_SECONDS,
                session,
                lambda active_session: _fetch_ssurgo_hydrologic_soil_grid(
                    active_session,
                    analysis_bbox_wgs84,
                    predicted_dem_shape,
                ),
            ),
        }
        dem_result, source_timings["dem"] = futures["dem"].result()
        dem_values, dem_bounds, dem_nodata, metric_crs = dem_result
        datum_result, source_timings["dem_datum"] = futures["datum"].result()
        (dem_datum, datum_note), vertical_result = datum_result
        vertical_offset_meters, vertical_conversion = vertical_result
        (groundsource, source_timings["groundsource"]) = futures[
            "groundsource"
        ].result()
        historical_event_count = groundsource.get("events_intersecting_area", 0)
        urban_drainage_multiplier = 1.0
        (waterbodies, waterbody_notes), source_timings["waterbodies"] = (
            futures["waterbodies"].result()
        )
        try:
            landcover, source_timings["nlcd"] = futures["nlcd"].result()
            nlcd_error = None
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            landcover = np.full((1, 1), FALLBACK_NLCD_CLASS, dtype=np.uint8)
            nlcd_error = exc
            source_timings["nlcd"] = {
                "elapsed_seconds": perf_counter() - started,
                "cache": "fallback",
                "source": "MRLC NLCD",
                "error": str(exc),
            }
        try:
            (soil_groups, ssurgo_source), source_timings["soils"] = (
                futures["soils"].result()
            )
            soil_error = None
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
            soil_groups = np.full(
                predicted_dem_shape,
                ord(FALLBACK_HYDROLOGIC_SOIL_GROUP) - ord("A"),
                dtype=np.uint8,
            )
            soil_error = exc
            source_timings["soils"] = {
                "elapsed_seconds": perf_counter() - started,
                "cache": "fallback",
                "source": "USDA SSURGO",
                "error": str(exc),
            }
        source_timings["parallel_fetch_wall_seconds"] = perf_counter() - started

    dem_sample = _bilinear_sample(
        dem_values,
        dem_bounds,
        longitude,
        latitude,
        metric_crs,
        dem_nodata,
    )
    water_mask = _waterbody_raster_mask(
        waterbodies,
        dem_bounds,
        dem_values.shape,
    )
    source_notes: list[str] = []
    if nlcd_error is None:
        nlcd_source = {
            "source": "MRLC NLCD 2021 Land Cover",
            "url": MRLC_WMS_URL,
            "layer": MRLC_NLCD_LAYER,
            "status": "used",
            "pixel_size_meters": [30, 30],
            "interpolation": (
                "30-meter classes are bilinearly averaged to the DEM analysis "
                "grid; interpolated pixels are estimates, not new observations"
            ),
        }
    else:
        source_notes.append(
            "NLCD request failed; fallback grassland cover class was used: "
            f"{nlcd_error}"
        )
        nlcd_source = {
            "source": "MRLC NLCD 2021 Land Cover",
            "url": MRLC_WMS_URL,
            "layer": MRLC_NLCD_LAYER,
            "status": "fallback",
            "fallback_class_code": FALLBACK_NLCD_CLASS,
            "error": str(nlcd_error),
        }
    if soil_error is None:
        ssurgo_source["status"] = (
            "used"
            if ssurgo_source["samples_with_soil_attributes"]
            else "fallback"
        )
        if not ssurgo_source["samples_with_soil_attributes"]:
            source_notes.append(
                "SSURGO returned no usable hydrologic soil groups; average "
                "hydrologic soil group C was used."
            )
    else:
        source_notes.append(
            "SSURGO request failed; average hydrologic soil group C was used: "
            f"{soil_error}"
        )
        ssurgo_source = {
            "source": "USDA NRCS SSURGO Soil Data Access",
            "url": SSURGO_QUERY_URL,
            "status": "fallback",
            "fallback_group": FALLBACK_HYDROLOGIC_SOIL_GROUP,
            "error": str(soil_error),
        }
    if soil_groups.shape != dem_values.shape:
        soil_groups = _resample_continuous_grid(soil_groups, dem_values.shape)
    curve_numbers, curve_number_summary = _curve_number_grid(
        landcover,
        soil_groups,
        water_mask,
    )
    valid_dem = np.isfinite(dem_values)
    if dem_nodata is not None:
        valid_dem &= dem_values != dem_nodata
    curve_numbers[~valid_dem] = np.nan
    if retained_water_fraction is None:
        runoff_depth_grid = _scs_runoff_depth_grid(rainfall_inches, curve_numbers)

        # Up to +15% runoff depth bump for areas with chronic historical flood news reports
        urban_drainage_multiplier = min(1.15, 1.0 + (historical_event_count * 0.03))
        runoff_depth_grid = runoff_depth_grid * urban_drainage_multiplier

        depth_grid, basin_surface_level_range, routing_summary = (
            _solve_d8_catchment_ponding(
                dem_values,
                runoff_depth_grid,
                (dem_bounds[2] - dem_bounds[0])
                * (dem_bounds[3] - dem_bounds[1])
                / dem_values.size,
                dem_nodata,
            )
        )
        water_surface_datum = None
        pixel_area_m2 = (
            (dem_bounds[2] - dem_bounds[0])
            * (dem_bounds[3] - dem_bounds[1])
            / dem_values.size
        )
        runoff_volume_m3 = routing_summary["routed_runoff_volume_m3"]
        model_method = (
            "SCS curve-number rainfall excess from MRLC NLCD land cover and "
            "USDA SSURGO hydrologic soil groups, routed by D8 into per-sink "
            "catchment ponding over an expanded DEM context domain"
        )
        land_cell_count = int(np.isfinite(runoff_depth_grid).sum())
        effective_rainfall_inches = (
            float(np.nansum(runoff_depth_grid) / land_cell_count * 39.3700787402)
            if land_cell_count
            else 0.0
        )
        model_runoff_fraction = (
            float(
                np.nansum(runoff_depth_grid)
                / (rainfall_inches * 0.0254 * land_cell_count)
            )
            if rainfall_inches > 0 and land_cell_count
            else 0.0
        )
    else:
        basin_surface_level_range = None
        water_surface_datum, depth_grid, pixel_area_m2 = _solve_retained_rainfall_surface(
            dem_values,
            dem_bounds,
            rainfall_inches,
            retained_water_fraction,
            dem_nodata,
            water_mask,
        )
        runoff_volume_m3 = (
            rainfall_inches
            * 0.0254
            * retained_water_fraction
            * pixel_area_m2
            * int(np.count_nonzero(np.isfinite(depth_grid)))
        )
        model_method = (
            "legacy explicit retained-rainfall-fraction closed-box DEM scenario; "
            "NLCD and SSURGO are diagnostics only"
        )
        effective_rainfall_inches = rainfall_inches * retained_water_fraction
        model_runoff_fraction = retained_water_fraction
        routing_summary = {
            "method": "disabled_for_legacy_retained_rainfall_mode",
        }
    forward = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{metric_crs}",
        always_xy=True,
    )
    point_x, point_y = forward.transform(longitude, latitude)
    point_is_waterbody = any(
        waterbody["geometry"].covers(Point(point_x, point_y))
        for waterbody in waterbodies
    )
    if point_is_waterbody:
        point_depth = 0.0
    elif retained_water_fraction is None:
        point_depth = _bilinear_sample(
            depth_grid,
            dem_bounds,
            longitude,
            latitude,
            metric_crs,
            None,
        )
    else:
        point_depth = (
            max(water_surface_datum - dem_sample, 0.0)
            if water_surface_datum is not None
            else 0.0
        )
    if (
        rainfall_inches == 0
        or (retained_water_fraction == 0)
        or point_is_waterbody
    ):
        point_depth = 0.0
    water_surface_for_output = dem_sample + point_depth
    contours = vectorize_depth_thresholds(
        dem_values,
        dem_bounds,
        metric_crs,
        water_surface_datum if water_surface_datum is not None else dem_sample,
        nodata=dem_nodata,
        thresholds_meters=DEPTH_THRESHOLDS_METERS,
        water_mask=water_mask,
        clip_geometry_metric=requested_projected,
        depth_grid_meters=depth_grid,
    )
    actual_output_datum = (
        "LMSL"
        if vertical_conversion["status"] == "converted"
        else dem_datum or "unknown"
    )
    ground_elevation_output = round(dem_sample + vertical_offset_meters, 3)
    water_surface_output = round(
        water_surface_for_output + vertical_offset_meters,
        3,
    )
    water_depth_output = round(
        max(water_surface_output - ground_elevation_output, 0.0),
        3,
    )

    notes = [
        "This is a screening scenario, not an observed or locally calibrated street-flood forecast.",
        "The depth model routes runoff down a D8 terrain network and balances volume within each interior sink catchment.",
        "The DEM is expanded beyond the output bbox to include nearby terrain and clipped back.",
        "30-meter NLCD classes and SSURGO soil samples are interpolated to the finer terrain grid.",
    ]
    payload = {
        "ar_flood_data": {
            "ground_elevation_msl_meters": ground_elevation_output,
            "water_surface_msl_meters": water_surface_output,
            "water_depth_meters": water_depth_output,
            "is_flooded": water_depth_output > 0,
            "elevation_datum": actual_output_datum,
        },
        "map_contours": contours,
    }
    diagnostics = {
        "status": "screening_scs_curve_number_estimate",
        "request": {
            "latitude": latitude,
            "longitude": longitude,
            "bbox_wgs84_west_south_east_north": list(bbox_wgs84),
            "rainfall_inches": rainfall_inches,
            "rainfall_intensity_inches_per_hour": rainfall_intensity_inches_per_hour,
            "rainfall_duration_hours": rainfall_duration_hours,
        },
        "model": {
            "method": model_method,
            "historical_groundsource_events": historical_event_count,
            "backend_urban_drainage_calibration_multiplier": urban_drainage_multiplier,
            "calibrated_for_location": False,
            "calibration_status": "not_calibrated_no_point_depth_labels_available",
            "retained_water_fraction": retained_water_fraction,
            "effective_rainfall_inches": effective_rainfall_inches,
            "effective_runoff_fraction": model_runoff_fraction,
            "point_depth_meters_unrounded": point_depth,
            "maximum_grid_depth_meters": float(np.nanmax(depth_grid)) if np.isfinite(depth_grid).any() else 0.0,
            "notes": notes,
        },
        "performance": {
            "parallel_source_fetch_wall_seconds": source_timings[
                "parallel_fetch_wall_seconds"
            ],
            "total_build_seconds": perf_counter() - request_started,
            "source_timings": source_timings,
            "post_parallel_fetch_seconds": max(
                0.0,
                perf_counter()
                - request_started
                - source_timings["parallel_fetch_wall_seconds"],
            ),
            "d8_skipped": False,
            "why_d8_remains_enabled": (
                "Current direct-observation sources provide no spatially matched "
                "numeric rainfall-flood depths; event footprints and hazard zones "
                "cannot substitute for modeled depth."
            ),
        },
        "dem": {
            "source": THREEDEP_EXPORT_URL,
            "sampled_ground_elevation_meters_unconverted": dem_sample,
            "vertical_datum": dem_datum,
            "vertical_datum_conversion": vertical_conversion,
        }, 
    }
    return payload, diagnostics


MAX_REQUEST_BYTES = 64_000


def warm_up() -> None:
    """Pre-load the Groundsource index so the first real request is not slow."""
    if GROUND_SOURCE_PATH.is_file():
        _groundsource_florida_events(GROUND_SOURCE_PATH)


def create_request_handler():
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
                _validate_request((south + north) / 2, (west + east) / 2, bbox)
                with requests.Session() as session:
                    collection = _waterbody_feature_collection(session, bbox)
                self._write_json(200, collection)
            except (KeyError, TypeError, ValueError) as exc:
                self._write_json(400, {"error": f"Invalid request: {exc}"})
            except (OSError, requests.RequestException, RuntimeError) as exc:
                self._write_json(503, {"error": str(exc)})
            except Exception:
                self._write_json(500, {"error": "Internal server error."})

        def do_POST(self) -> None:
            if self.path != "/flood":
                self._write_json(404, {"error": "POST endpoint must be /flood."})
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
                self._write_json(400, {"error": f"Missing required field: {exc.args[0]}"})
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must contain numbers.")
    return value


def _optional_number(body: dict[str, Any], name: str) -> float | None:
    value = body.get(name)
    return None if value is None else _number(value, name)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), create_request_handler())
    threading.Thread(target=warm_up, daemon=True).start()
    print(f"FloodSight service listening at http://{host}:{port} (POST /flood, GET /waterbodies, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping FloodSight service.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FloodSight Florida-wide flood JSON microservice."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        default=True,
        help="Run the HTTP POST /flood microservice.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
