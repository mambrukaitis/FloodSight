"""Concurrent download of every remote dataset one flood estimate needs.

`fetch_source_data` starts the 3DEP elevation, vertical-datum, Groundsource,
surface-water, NLCD land-cover and SSURGO soil lookups in parallel (each one
cached with its own time-to-live) and gathers the results into a `SourceData`
record. Land-cover and soil failures are tolerated with documented fallback
classes; elevation, datum and water failures propagate to the caller.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import requests

from floodsight.dem_elevation_source import dem_grid_shape, fetch_dem
from floodsight.groundsource_history import (
    GROUND_SOURCE_PATH,
    GROUNDSOURCE_BUFFER_DEGREES,
    buffered_bbox,
    groundsource_job,
)
from floodsight.nlcd_landcover_source import fetch_nlcd_landcover
from floodsight.request_caching import cached_source_job, spatial_cache_key
from floodsight.scs_runoff import FALLBACK_NLCD_CLASS
from floodsight.ssurgo_soil_source import (
    FALLBACK_HYDROLOGIC_SOIL_GROUP,
    fetch_ssurgo_hydrologic_soil_grid,
)
from floodsight.vertical_datum_source import datum_and_offset
from floodsight.waterbody_source import fetch_waterbodies

MAX_PARALLEL_DATA_REQUESTS = 6

DEM_CACHE_TTL_SECONDS = 86_400
NLCD_CACHE_TTL_SECONDS = 30 * 86_400
SSURGO_CACHE_TTL_SECONDS = 7 * 86_400
WATERBODY_CACHE_TTL_SECONDS = 86_400
VDATUM_CACHE_TTL_SECONDS = 30 * 86_400


@dataclass
class SourceData:
    """Everything fetched for one request, plus per-source timing records."""

    dem_values: np.ndarray
    dem_bounds: tuple[float, float, float, float]
    dem_nodata: float | None
    metric_crs: int
    dem_datum: str | None
    vertical_offset_meters: float
    vertical_conversion: dict[str, Any]
    groundsource: dict[str, Any]
    waterbodies: list[dict[str, Any]]
    landcover: np.ndarray
    soil_groups: np.ndarray
    source_timings: dict[str, Any]


def _fallback_timing(started: float, source: str, exc: Exception) -> dict[str, Any]:
    """Timing record for a source that failed and was replaced by its fallback value."""
    return {
        "elapsed_seconds": perf_counter() - started,
        "cache": "fallback",
        "source": source,
        "error": str(exc),
    }


def fetch_source_data(
    latitude: float,
    longitude: float,
    analysis_bbox_wgs84: tuple[float, float, float, float],
    analysis_metric_bounds: tuple[float, float, float, float],
    metric_crs: int,
    session: requests.Session | None,
) -> SourceData:
    """Fetch all remote and local datasets for the analysis domain in parallel."""
    predicted_dem_shape = dem_grid_shape(analysis_bbox_wgs84, metric_crs)
    source_timings: dict[str, Any] = {}
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DATA_REQUESTS) as executor:
        futures = {
            "dem": executor.submit(
                cached_source_job,
                "3DEP elevation",
                spatial_cache_key(
                    "dem", metric_crs, analysis_metric_bounds, predicted_dem_shape
                ),
                DEM_CACHE_TTL_SECONDS,
                session,
                lambda active_session: fetch_dem(
                    active_session, latitude, longitude, analysis_bbox_wgs84
                ),
            ),
            "datum": executor.submit(
                cached_source_job,
                "3DEP vertical datum + NOAA VDatum offset",
                spatial_cache_key(
                    "dem_datum_offset",
                    metric_crs,
                    round(latitude, 5),
                    round(longitude, 5),
                ),
                VDATUM_CACHE_TTL_SECONDS,
                session,
                lambda active_session: datum_and_offset(
                    active_session, latitude, longitude, metric_crs
                ),
            ),
            "groundsource": executor.submit(
                groundsource_job,
                buffered_bbox(analysis_bbox_wgs84, GROUNDSOURCE_BUFFER_DEGREES),
                GROUND_SOURCE_PATH,
            ),
            "waterbodies": executor.submit(
                cached_source_job,
                "surface-water polygons",
                spatial_cache_key("waterbodies", analysis_bbox_wgs84, metric_crs),
                WATERBODY_CACHE_TTL_SECONDS,
                session,
                lambda active_session: fetch_waterbodies(
                    active_session,
                    analysis_bbox_wgs84,
                    metric_crs,
                    analysis_metric_bounds,
                ),
            ),
            "nlcd": executor.submit(
                cached_source_job,
                "MRLC NLCD",
                spatial_cache_key("nlcd", analysis_metric_bounds, metric_crs),
                NLCD_CACHE_TTL_SECONDS,
                session,
                lambda active_session: fetch_nlcd_landcover(
                    active_session,
                    analysis_metric_bounds,
                    metric_crs,
                    predicted_dem_shape,
                ),
            ),
            "soils": executor.submit(
                cached_source_job,
                "USDA SSURGO",
                spatial_cache_key("ssurgo", analysis_bbox_wgs84, predicted_dem_shape),
                SSURGO_CACHE_TTL_SECONDS,
                session,
                lambda active_session: fetch_ssurgo_hydrologic_soil_grid(
                    active_session, analysis_bbox_wgs84, predicted_dem_shape
                ),
            ),
        }
        dem_result, source_timings["dem"] = futures["dem"].result()
        dem_values, dem_bounds, dem_nodata, metric_crs = dem_result
        datum_result, source_timings["dem_datum"] = futures["datum"].result()
        (dem_datum, _datum_note), vertical_result = datum_result
        vertical_offset_meters, vertical_conversion = vertical_result
        groundsource, source_timings["groundsource"] = futures["groundsource"].result()
        (waterbodies, _waterbody_notes), source_timings["waterbodies"] = futures[
            "waterbodies"
        ].result()
        try:
            landcover, source_timings["nlcd"] = futures["nlcd"].result()
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            landcover = np.full((1, 1), FALLBACK_NLCD_CLASS, dtype=np.uint8)
            source_timings["nlcd"] = _fallback_timing(started, "MRLC NLCD", exc)
        try:
            (soil_groups, _ssurgo_metadata), source_timings["soils"] = futures[
                "soils"
            ].result()
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
            soil_groups = np.full(
                predicted_dem_shape,
                ord(FALLBACK_HYDROLOGIC_SOIL_GROUP) - ord("A"),
                dtype=np.uint8,
            )
            source_timings["soils"] = _fallback_timing(started, "USDA SSURGO", exc)
        source_timings["parallel_fetch_wall_seconds"] = perf_counter() - started
    return SourceData(
        dem_values=dem_values,
        dem_bounds=dem_bounds,
        dem_nodata=dem_nodata,
        metric_crs=metric_crs,
        dem_datum=dem_datum,
        vertical_offset_meters=vertical_offset_meters,
        vertical_conversion=vertical_conversion,
        groundsource=groundsource,
        waterbodies=waterbodies,
        landcover=landcover,
        soil_groups=soil_groups,
        source_timings=source_timings,
    )
