"""Builds the flood estimate returned by `POST /flood`.

`build_flood_payload` validates a request, fetches the source datasets
(`source_fetching`), runs the depth model (`flood_depth_model`), samples the
modeled depth at the requested point, vectorizes depth contours, converts
elevations to the output vertical datum, and returns the app payload plus a
diagnostics dictionary (model details and per-source timings) for logging.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import requests
from pyproj import Transformer
from shapely import transform
from shapely.geometry import Point, box

from floodsight.contour_polygons import DEPTH_THRESHOLDS_METERS, vectorize_depth_thresholds
from floodsight.dem_elevation_source import THREEDEP_EXPORT_URL
from floodsight.flood_depth_model import DepthModelResult, run_depth_model
from floodsight.grid_geometry import (
    analysis_domain,
    bilinear_sample,
    local_utm_epsg,
    validate_request,
)
from floodsight.rainfall_inputs import (
    resolve_rainfall_inches,
    validate_retained_water_fraction,
)
from floodsight.source_fetching import SourceData, fetch_source_data
from floodsight.terrain_ponding_solver import waterbody_raster_mask

MODEL_NOTES = [
    "This is a screening scenario, not an observed or locally calibrated street-flood forecast.",
    "The depth model routes runoff down a D8 terrain network and balances volume within each interior sink catchment.",
    "The DEM is expanded beyond the output bbox to include nearby terrain and clipped back.",
    "30-meter NLCD classes and SSURGO soil samples are interpolated to the finer terrain grid.",
]


def _point_depth_meters(
    latitude: float,
    longitude: float,
    rainfall_inches: float,
    retained_water_fraction: float | None,
    sources: SourceData,
    model: DepthModelResult,
    dem_sample: float,
) -> float:
    """Modeled water depth at the requested point; zero on mapped surface water."""
    forward = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{sources.metric_crs}", always_xy=True
    )
    point_x, point_y = forward.transform(longitude, latitude)
    point = Point(point_x, point_y)
    point_is_waterbody = any(
        waterbody["geometry"].covers(point) for waterbody in sources.waterbodies
    )
    if point_is_waterbody or rainfall_inches == 0 or retained_water_fraction == 0:
        return 0.0
    if retained_water_fraction is None:
        return bilinear_sample(
            model.depth_grid,
            sources.dem_bounds,
            longitude,
            latitude,
            sources.metric_crs,
            None,
        )
    if model.water_surface_datum is None:
        return 0.0
    return max(model.water_surface_datum - dem_sample, 0.0)


def _build_diagnostics(
    latitude: float,
    longitude: float,
    bbox_wgs84: tuple[float, float, float, float],
    rainfall_inches: float,
    rainfall_intensity_inches_per_hour: float | None,
    rainfall_duration_hours: float | None,
    retained_water_fraction: float | None,
    sources: SourceData,
    model: DepthModelResult,
    point_depth: float,
    dem_sample: float,
    request_started: float,
) -> dict[str, Any]:
    """Assemble the diagnostics dict (request echo, model details, timings, DEM datum)
    that accompanies the payload.
    """
    depth_grid = model.depth_grid
    parallel_wall = sources.source_timings["parallel_fetch_wall_seconds"]
    return {
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
            "method": model.model_method,
            "historical_groundsource_events": model.historical_event_count,
            "backend_urban_drainage_calibration_multiplier": model.urban_drainage_multiplier,
            "calibrated_for_location": False,
            "calibration_status": "not_calibrated_no_point_depth_labels_available",
            "retained_water_fraction": retained_water_fraction,
            "effective_rainfall_inches": model.effective_rainfall_inches,
            "effective_runoff_fraction": model.model_runoff_fraction,
            "point_depth_meters_unrounded": point_depth,
            "maximum_grid_depth_meters": (
                float(np.nanmax(depth_grid)) if np.isfinite(depth_grid).any() else 0.0
            ),
            "notes": MODEL_NOTES,
        },
        "performance": {
            "parallel_source_fetch_wall_seconds": parallel_wall,
            "total_build_seconds": perf_counter() - request_started,
            "source_timings": sources.source_timings,
            "post_parallel_fetch_seconds": max(
                0.0, perf_counter() - request_started - parallel_wall
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
            "vertical_datum": sources.dem_datum,
            "vertical_datum_conversion": sources.vertical_conversion,
        },
    }


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
    validate_request(latitude, longitude, bbox_wgs84)
    rainfall_inches = resolve_rainfall_inches(
        rainfall_inches,
        rainfall_intensity_inches_per_hour,
        rainfall_duration_hours,
    )
    validate_retained_water_fraction(retained_water_fraction)

    metric_crs = local_utm_epsg(latitude, longitude)
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    requested_projected = transform(
        box(*bbox_wgs84).segmentize(0.0001),
        forward.transform,
        interleaved=False,
    )
    analysis_bbox_wgs84, analysis_metric_bounds = analysis_domain(
        bbox_wgs84, metric_crs
    )

    sources = fetch_source_data(
        latitude,
        longitude,
        analysis_bbox_wgs84,
        analysis_metric_bounds,
        metric_crs,
        session,
    )
    dem_sample = bilinear_sample(
        sources.dem_values,
        sources.dem_bounds,
        longitude,
        latitude,
        sources.metric_crs,
        sources.dem_nodata,
    )
    water_mask = waterbody_raster_mask(
        sources.waterbodies, sources.dem_bounds, sources.dem_values.shape
    )
    model = run_depth_model(sources, rainfall_inches, retained_water_fraction, water_mask)
    point_depth = _point_depth_meters(
        latitude,
        longitude,
        rainfall_inches,
        retained_water_fraction,
        sources,
        model,
        dem_sample,
    )

    contours = vectorize_depth_thresholds(
        sources.dem_values,
        sources.dem_bounds,
        sources.metric_crs,
        model.water_surface_datum if model.water_surface_datum is not None else dem_sample,
        nodata=sources.dem_nodata,
        thresholds_meters=DEPTH_THRESHOLDS_METERS,
        water_mask=water_mask,
        clip_geometry_metric=requested_projected,
        depth_grid_meters=model.depth_grid,
    )
    output_datum = (
        "LMSL"
        if sources.vertical_conversion["status"] == "converted"
        else sources.dem_datum or "unknown"
    )
    offset = sources.vertical_offset_meters
    ground_elevation_output = round(dem_sample + offset, 3)
    water_surface_output = round(dem_sample + point_depth + offset, 3)
    water_depth_output = round(
        max(water_surface_output - ground_elevation_output, 0.0), 3
    )
    payload = {
        "ar_flood_data": {
            "ground_elevation_msl_meters": ground_elevation_output,
            "water_surface_msl_meters": water_surface_output,
            "water_depth_meters": water_depth_output,
            "is_flooded": water_depth_output > 0,
            "elevation_datum": output_datum,
        },
        "map_contours": contours,
    }
    diagnostics = _build_diagnostics(
        latitude,
        longitude,
        bbox_wgs84,
        rainfall_inches,
        rainfall_intensity_inches_per_hour,
        rainfall_duration_hours,
        retained_water_fraction,
        sources,
        model,
        point_depth,
        dem_sample,
        request_started,
    )
    return payload, diagnostics
