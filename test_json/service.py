from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import threading
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET

import numpy as np
import requests
import tifffile
from pyproj import Transformer
from shapely import contains_xy, transform
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import polygonize, unary_union

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
MAX_DEM_PIXELS = 50_000
MAX_PARALLEL_DATA_REQUESTS = 6
DEM_CACHE_TTL_SECONDS = 86_400
NLCD_CACHE_TTL_SECONDS = 30 * 86_400
SSURGO_CACHE_TTL_SECONDS = 7 * 86_400
WATERBODY_CACHE_TTL_SECONDS = 86_400
VDATUM_CACHE_TTL_SECONDS = 30 * 86_400
USGS_CACHE_TTL_SECONDS = 300
DEPTH_THRESHOLDS_METERS = (0.1, 0.2, 0.3, 1.0)
DEFAULT_RAINFALL_INCHES = 10.0
MAX_CONTOUR_VERTICES = 15
MAX_BBOX_EXTENT_DEGREES = 7.0
LAKE_ALICE_EXAMPLE_SIDE_METERS = 250.0
TERRAIN_CONTEXT_BUFFER_METERS = 250.0
NOAA_VDATUM_URL = "https://vdatum.noaa.gov/vdatumweb/api/convert"
NHD_MAPSERVER_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
OSM_API_URL = "https://api.openstreetmap.org/api/0.6"
MRLC_WMS_URL = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
MRLC_NLCD_LAYER = "NLCD_2021_Land_Cover_L48"
SSURGO_QUERY_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
SSURGO_SAMPLE_GRID_SIZE = 4
FALLBACK_HYDROLOGIC_SOIL_GROUP = "C"
FALLBACK_NLCD_CLASS = 71
SCS_CN_BY_NLCD_CLASS: dict[int, tuple[float, float, float, float]] = {
    21: (49, 69, 79, 84),
    22: (57, 72, 81, 86),
    23: (77, 85, 90, 92),
    24: (89, 92, 94, 95),
    31: (77, 86, 91, 94),
    41: (30, 55, 70, 77),
    42: (30, 55, 70, 77),
    43: (30, 55, 70, 77),
    52: (35, 56, 70, 77),
    71: (39, 61, 74, 80),
    81: (49, 69, 79, 84),
    82: (67, 78, 85, 89),
    90: (70, 80, 87, 91),
    95: (77, 86, 91, 94),
}
NLCD_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    11: (70, 107, 159),
    12: (209, 222, 248),
    21: (222, 197, 197),
    22: (217, 146, 130),
    23: (235, 0, 0),
    24: (171, 0, 0),
    31: (179, 172, 159),
    41: (104, 171, 95),
    42: (28, 95, 44),
    43: (181, 197, 143),
    51: (175, 150, 60),
    52: (204, 184, 121),
    71: (223, 223, 194),
    72: (209, 209, 130),
    73: (163, 204, 81),
    74: (130, 186, 65),
    81: (220, 217, 57),
    82: (171, 108, 40),
    90: (184, 217, 235),
    95: (108, 159, 184),
}
NLCD_COLOR_MATCH_TOLERANCE = 30
LAKE_ALICE = {
    "latitude": 29.64289215,
    "longitude": -82.36122382833753,
    "osm_relation_id": 1630473,
}
_SPATIAL_CACHE: dict[tuple[Any, ...], tuple[float, Any]] = {}
_SPATIAL_CACHE_LOCK = threading.Lock()


def bbox_around_point(
    latitude: float,
    longitude: float,
    side_meters: float = DEFAULT_BBOX_SIDE_METERS,
) -> tuple[float, float, float, float]:
    if not math.isfinite(side_meters) or side_meters <= 0:
        raise ValueError("Bounding-box side length must be positive and finite.")
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


def reitz_bbox(
    side_meters: float = DEFAULT_BBOX_SIDE_METERS,
) -> tuple[float, float, float, float]:
    return bbox_around_point(
        REITZ_UNION["latitude"],
        REITZ_UNION["longitude"],
        side_meters,
    )


def lake_alice_bbox(
    side_meters: float = LAKE_ALICE_EXAMPLE_SIDE_METERS,
) -> tuple[float, float, float, float]:
    return bbox_around_point(
        LAKE_ALICE["latitude"],
        LAKE_ALICE["longitude"],
        side_meters,
    )


def _analysis_domain(
    bbox_wgs84: tuple[float, float, float, float],
    metric_crs: int,
    buffer_meters: float = TERRAIN_CONTEXT_BUFFER_METERS,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    inverse = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
    projected = transform(box(*bbox_wgs84), forward.transform, interleaved=False)
    xmin, ymin, xmax, ymax = projected.bounds
    expanded_metric = (
        xmin - buffer_meters,
        ymin - buffer_meters,
        xmax + buffer_meters,
        ymax + buffer_meters,
    )
    expanded_wgs84 = transform(
        box(*expanded_metric),
        inverse.transform,
        interleaved=False,
    )
    return expanded_wgs84.bounds, expanded_metric


def _spatial_cache_key(source: str, *values: Any) -> tuple[Any, ...]:
    def normalize(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return tuple(normalize(item) for item in value)
        if isinstance(value, float):
            return round(value, 7)
        return value

    normalized = tuple(normalize(value) for value in values)
    return (source, *normalized)


def _cached_spatial_data(
    key: tuple[Any, ...],
    ttl_seconds: float,
    loader: Any,
) -> tuple[Any, bool, float]:
    started = perf_counter()
    now = started
    with _SPATIAL_CACHE_LOCK:
        cached = _SPATIAL_CACHE.get(key)
        if cached is not None and cached[0] > now:
            return copy.deepcopy(cached[1]), True, perf_counter() - started
        if cached is not None:
            del _SPATIAL_CACHE[key]
    value = loader()
    with _SPATIAL_CACHE_LOCK:
        _SPATIAL_CACHE[key] = (perf_counter() + ttl_seconds, copy.deepcopy(value))
    return value, False, perf_counter() - started


def _using_session(session: requests.Session | None, operation: Any) -> Any:
    if session is not None:
        return operation(session)
    with requests.Session() as local_session:
        return operation(local_session)


def _dem_grid_shape(
    bbox_wgs84: tuple[float, float, float, float],
    metric_crs: int,
) -> tuple[int, int]:
    projected = transform(
        box(*bbox_wgs84),
        Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True).transform,
        interleaved=False,
    )
    xmin, ymin, xmax, ymax = projected.bounds
    width_m, height_m = xmax - xmin, ymax - ymin
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Bounding box has no area after projection.")
    pixel_size_m = max(1.0, math.sqrt(width_m * height_m / MAX_DEM_PIXELS))
    columns = max(1, math.ceil(width_m / pixel_size_m))
    rows = max(1, math.ceil(height_m / pixel_size_m))
    while rows * columns > MAX_DEM_PIXELS:
        pixel_size_m *= math.sqrt(rows * columns / MAX_DEM_PIXELS)
        columns = max(1, math.ceil(width_m / pixel_size_m))
        rows = max(1, math.ceil(height_m / pixel_size_m))
    return rows, columns


def _cached_source_job(
    source_name: str,
    key: tuple[Any, ...],
    ttl_seconds: float,
    session: requests.Session | None,
    operation: Any,
) -> tuple[Any, dict[str, Any]]:
    value, cache_hit, elapsed = _cached_spatial_data(
        key,
        ttl_seconds,
        lambda: _using_session(session, operation),
    )
    return value, {
        "elapsed_seconds": elapsed,
        "cache": "hit" if cache_hit else "miss",
        "source": source_name,
    }


def _groundsource_job(
    bbox_wgs84: tuple[float, float, float, float],
    parquet_path: Path,
    cache_dir: Path,
) -> tuple[tuple[dict[str, Any], list[str]], dict[str, Any]]:
    started = perf_counter()
    result = _groundsource_report(bbox_wgs84, parquet_path, cache_dir)
    report, notes = result
    return result, {
        "elapsed_seconds": perf_counter() - started,
        "cache": (
            "disk-hit"
            if any("cache hit" in note.lower() for note in notes)
            else "disk-miss"
        ),
        "source": "Groundsource",
    }


def _resample_continuous_grid(
    values: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if source.ndim != 2:
        raise ValueError("Source grid must be two-dimensional.")
    if min(source.shape) < 1 or min(target_shape) < 1:
        raise ValueError("Source and target grid dimensions must be positive.")
    if source.shape == target_shape:
        return source.copy()

    row_positions = np.linspace(0, source.shape[0] - 1, target_shape[0])
    column_positions = np.linspace(0, source.shape[1] - 1, target_shape[1])
    row0 = np.floor(row_positions).astype(int)
    column0 = np.floor(column_positions).astype(int)
    row1 = np.minimum(row0 + 1, source.shape[0] - 1)
    column1 = np.minimum(column0 + 1, source.shape[1] - 1)
    row_fraction = row_positions - row0
    column_fraction = column_positions - column0

    top = (
        source[row0[:, None], column0[None, :]]
        * (1 - column_fraction[None, :])
        + source[row0[:, None], column1[None, :]]
        * column_fraction[None, :]
    )
    bottom = (
        source[row1[:, None], column0[None, :]]
        * (1 - column_fraction[None, :])
        + source[row1[:, None], column1[None, :]]
        * column_fraction[None, :]
    )
    return (
        top * (1 - row_fraction[:, None])
        + bottom * row_fraction[:, None]
    )


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
    if east - west > MAX_BBOX_EXTENT_DEGREES or north - south > MAX_BBOX_EXTENT_DEGREES:
        raise ValueError(
            f"Bounding-box dimensions cannot exceed {MAX_BBOX_EXTENT_DEGREES}°."
        )


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
    while rows * columns > MAX_DEM_PIXELS:
        pixel_size_m *= math.sqrt(rows * columns / MAX_DEM_PIXELS)
        columns = max(1, math.ceil(width_m / pixel_size_m))
        rows = max(1, math.ceil(height_m / pixel_size_m))

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


def _is_osm_waterbody(tags: dict[str, str]) -> bool:
    return (
        tags.get("natural") == "water"
        or "water" in tags
        or tags.get("landuse") == "reservoir"
    )


def _osm_relation_polygon(
    session: requests.Session,
    relation_id: str,
) -> Any:
    response = session.get(
        f"{OSM_API_URL}/relation/{relation_id}/full.json",
        timeout=(5, 20),
    )
    response.raise_for_status()
    elements = response.json().get("elements", [])
    nodes = {
        element["id"]: (element["lon"], element["lat"])
        for element in elements
        if element.get("type") == "node"
    }
    ways = {
        element["id"]: element
        for element in elements
        if element.get("type") == "way"
    }
    relation = next(
        (
            element
            for element in elements
            if element.get("type") == "relation"
            and str(element.get("id")) == relation_id
        ),
        None,
    )
    if relation is None:
        raise RuntimeError(f"OSM relation {relation_id} was not returned.")

    member_lines: dict[str, list[LineString]] = {"outer": [], "inner": []}
    for member in relation.get("members", []):
        way = ways.get(member.get("ref"))
        role = member.get("role")
        if way is None or role not in member_lines:
            continue
        coordinates = [
            nodes[node_id]
            for node_id in way.get("nodes", [])
            if node_id in nodes
        ]
        if len(coordinates) >= 4 and coordinates[0] == coordinates[-1]:
            member_lines[role].append(LineString(coordinates))
    if not member_lines["outer"]:
        raise RuntimeError(f"OSM waterbody relation {relation_id} has no outer ring.")
    outer = unary_union(list(polygonize(unary_union(member_lines["outer"]))))
    inner = unary_union(list(polygonize(unary_union(member_lines["inner"]))))
    return outer.difference(inner) if not inner.is_empty else outer


def _fetch_osm_waterbodies(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    west, south, east, north = bbox_wgs84
    response = session.get(
        f"{OSM_API_URL}/map",
        params={"bbox": f"{west},{south},{east},{north}"},
        timeout=(5, 25),
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    nodes = {
        node.get("id"): (float(node.get("lon", "nan")), float(node.get("lat", "nan")))
        for node in root.findall("node")
    }
    features: list[dict[str, Any]] = []
    errors: list[str] = []
    relation_ids: set[str] = set()
    for relation in root.findall("relation"):
        tags = {tag.get("k", ""): tag.get("v", "") for tag in relation.findall("tag")}
        if _is_osm_waterbody(tags):
            relation_ids.add(relation.get("id", ""))
    for relation_id in sorted(relation_ids):
        if not relation_id:
            continue
        try:
            geometry = _osm_relation_polygon(session, relation_id)
            if not geometry.is_empty:
                features.append(
                    {
                        "geometry": geometry,
                        "name": next(
                            (
                                tag.get("v")
                                for tag in root.findall(
                                    f".//relation[@id='{relation_id}']/tag"
                                )
                                if tag.get("k") == "name"
                            ),
                            None,
                        ),
                        "source": "OpenStreetMap",
                        "feature_id": f"relation/{relation_id}",
                    }
                )
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
            errors.append(f"OSM waterbody relation {relation_id}: {exc}")
    for way in root.findall("way"):
        tags = {tag.get("k", ""): tag.get("v", "") for tag in way.findall("tag")}
        if not _is_osm_waterbody(tags):
            continue
        coordinates = [
            nodes.get(node.get("ref", ""))
            for node in way.findall("nd")
        ]
        coordinates = [coordinate for coordinate in coordinates if coordinate]
        if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
            continue
        geometry = Polygon(coordinates)
        if geometry.is_valid and not geometry.is_empty:
            features.append(
                {
                    "geometry": geometry,
                    "name": tags.get("name"),
                    "source": "OpenStreetMap",
                    "feature_id": f"way/{way.get('id')}",
                }
            )
    return features, errors


def _fetch_nhd_waterbodies(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    west, south, east, north = bbox_wgs84
    layer_id = 10 if max(east - west, north - south) > 0.5 else 12
    response = session.get(
        f"{NHD_MAPSERVER_URL}/{layer_id}/query",
        params={
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "resultRecordCount": 500,
            "f": "geojson",
        },
        timeout=(10, 45),
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"USGS NHD query failed: {data['error']}")
    features = [
        {
            "geometry": shape(feature["geometry"]),
            "name": feature.get("properties", {}).get("GNIS_NAME"),
            "source": "USGS National Hydrography Dataset",
            "feature_id": str(
                feature.get("properties", {}).get("OBJECTID", "unknown")
            ),
        }
        for feature in data.get("features", [])
        if feature.get("geometry")
    ]
    notes = (
        ["USGS NHD response was truncated at its 500-feature request limit."]
        if data.get("exceededTransferLimit")
        else []
    )
    return features, notes


def _fetch_waterbodies(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
    metric_crs: int,
    bounds_metric: tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    west, south, east, north = bbox_wgs84
    geographic_area = (east - west) * (north - south)
    notes: list[str] = []
    candidates: list[dict[str, Any]] = []
    local_source_queried = False
    if geographic_area <= 0.25:
        try:
            candidates, osm_errors = _fetch_osm_waterbodies(session, bbox_wgs84)
            notes.extend(osm_errors)
            local_source_queried = not osm_errors
            if not candidates and local_source_queried:
                notes.append("OpenStreetMap returned no mapped waterbody in the box.")
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            notes.append(f"OSM local waterbody lookup failed: {exc}")
    if not candidates and not local_source_queried:
        try:
            candidates, nhd_notes = _fetch_nhd_waterbodies(session, bbox_wgs84)
            notes.extend(nhd_notes)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            notes.append(f"USGS NHD waterbody lookup failed: {exc}")

    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    region = box(*bounds_metric)
    waterbodies: list[dict[str, Any]] = []
    for feature in candidates:
        projected = transform(
            feature["geometry"],
            forward.transform,
            interleaved=False,
        ).intersection(region)
        if projected.is_empty or projected.area <= 0:
            continue
        waterbodies.append({**feature, "geometry": projected})
    return waterbodies, notes


def _waterbody_feature_collection(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
) -> dict[str, Any]:
    west, south, east, north = bbox_wgs84
    center_latitude = (south + north) / 2
    center_longitude = (west + east) / 2
    _validate_request(center_latitude, center_longitude, bbox_wgs84)
    metric_crs = _local_utm_epsg(center_latitude, center_longitude)
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    inverse = Transformer.from_crs(f"EPSG:{metric_crs}", "EPSG:4326", always_xy=True)
    projected_bbox = transform(
        box(*bbox_wgs84),
        forward.transform,
        interleaved=False,
    )
    waterbodies, notes = _fetch_waterbodies(
        session,
        bbox_wgs84,
        metric_crs,
        projected_bbox.bounds,
    )
    features = []
    for waterbody in waterbodies:
        geometry = waterbody["geometry"]
        polygons = (
            [geometry]
            if geometry.geom_type == "Polygon"
            else [part for part in geometry.geoms if part.geom_type == "Polygon"]
        )
        geographic_polygons = []
        for polygon in polygons:
            ring = _polygon_ring_coordinates(polygon)
            geographic = transform(
                Polygon(ring),
                inverse.transform,
                interleaved=False,
            )
            geographic_polygons.append(
                [
                    [
                        [longitude, latitude]
                        for longitude, latitude in geographic.exterior.coords
                    ]
                ]
            )
        if not geographic_polygons:
            continue
        geometry_type = (
            "Polygon" if len(geographic_polygons) == 1 else "MultiPolygon"
        )
        coordinates = (
            geographic_polygons[0]
            if geometry_type == "Polygon"
            else geographic_polygons
        )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "category": "waterbody",
                    "name": waterbody["name"],
                    "source": waterbody["source"],
                    "source_id": waterbody["feature_id"],
                },
                "geometry": {
                    "type": geometry_type,
                    "coordinates": coordinates,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "bbox_wgs84": list(bbox_wgs84),
            "category": "mapped_surface_water_not_flood_depth",
            "notes": notes,
        },
    }


def _polygon_ring_coordinates(
    polygon: Polygon,
    max_vertices: int = MAX_CONTOUR_VERTICES,
) -> list[tuple[float, float]]:
    if max_vertices < 4:
        raise ValueError("A closed polygon requires a maximum of at least 4 vertices.")
    simplified = polygon
    tolerance = max(polygon.length, 1.0) * 1e-9
    for _ in range(64):
        coordinates = list(simplified.exterior.coords)
        if len(coordinates) <= max_vertices:
            return coordinates
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        tolerance *= 2
        if simplified.is_empty:
            break
    simplified = polygon.minimum_rotated_rectangle
    return list(simplified.exterior.coords)


def vectorize_depth_thresholds(
    elevation_grid_meters: np.ndarray,
    bounds_metric: tuple[float, float, float, float],
    metric_crs: int,
    water_surface_meters: float,
    *,
    nodata: float | None = None,
    thresholds_meters: tuple[float, ...] = DEPTH_THRESHOLDS_METERS,
    water_mask: np.ndarray | None = None,
    clip_bounds_metric: tuple[float, float, float, float] | None = None,
    clip_geometry_metric: Polygon | None = None,
    depth_grid_meters: np.ndarray | None = None,
    max_vertices: int = MAX_CONTOUR_VERTICES,
) -> list[dict[str, Any]]:
    """Vectorize depth thresholds from a water surface or per-cell depth grid."""
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
    if water_mask is not None:
        if water_mask.shape != values.shape:
            raise ValueError("Water-body mask must match the elevation grid shape.")
        valid &= ~water_mask
    if depth_grid_meters is None:
        depth = np.maximum(water_surface_meters - values, 0)
    else:
        depth = np.asarray(depth_grid_meters, dtype=float)
        if depth.shape != values.shape:
            raise ValueError("Depth grid must match the elevation grid shape.")
        valid &= np.isfinite(depth)
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
        if clip_geometry_metric is not None:
            dissolved = dissolved.intersection(clip_geometry_metric)
        elif clip_bounds_metric is not None:
            dissolved = dissolved.intersection(box(*clip_bounds_metric))
        polygons = (
            [dissolved]
            if dissolved.geom_type == "Polygon"
            else list(getattr(dissolved, "geoms", []))
        )
        for polygon in polygons:
            if polygon.geom_type != "Polygon" or polygon.is_empty:
                continue
            polygon = polygon.simplify(
                max(pixel_width, pixel_height) * 1e-9,
                preserve_topology=True,
            )
            ring = _polygon_ring_coordinates(polygon, max_vertices)
            geographic = transform(
                Polygon(ring),
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


def _fetch_nlcd_landcover(
    session: requests.Session,
    bounds_metric: tuple[float, float, float, float],
    metric_crs: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    if len(grid_shape) != 2 or min(grid_shape) < 1:
        raise ValueError("Target land-cover grid dimensions must be positive.")
    xmin, ymin, xmax, ymax = bounds_metric
    rows = max(1, math.ceil((ymax - ymin) / 30))
    columns = max(1, math.ceil((xmax - xmin) / 30))
    response = session.get(
        MRLC_WMS_URL,
        params={
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetMap",
            "layers": MRLC_NLCD_LAYER,
            "styles": "",
            "srs": f"EPSG:{metric_crs}",
            "bbox": f"{xmin},{ymin},{xmax},{ymax}",
            "width": columns,
            "height": rows,
            "format": "image/geotiff",
            "transparent": "false",
        },
        timeout=(10, 90),
    )
    response.raise_for_status()
    try:
        with tifffile.TiffFile(io.BytesIO(response.content)) as raster:
            page = raster.pages[0]
            values = page.asarray()
            color_map_tag = page.colormap
    except (tifffile.TiffFileError, ValueError, IndexError) as exc:
        raise RuntimeError(f"Could not decode the MRLC NLCD GeoTIFF: {exc}") from exc
    if values.shape != (rows, columns):
        raise RuntimeError(
            f"MRLC returned land-cover shape {values.shape}; expected {(rows, columns)}."
        )
    if values.ndim != 2:
        raise RuntimeError("MRLC land-cover raster must be a single-band image.")
    return _decode_nlcd_values(values, color_map_tag)


def _decode_nlcd_values(
    values: np.ndarray,
    color_map: np.ndarray | None,
) -> np.ndarray:
    if color_map is None:
        decoded = values.astype(np.int64)
        recognized = np.isin(decoded, list(NLCD_CLASS_COLORS) + [0])
        if not recognized.all():
            unknown = np.unique(decoded[~recognized]).tolist()
            raise RuntimeError(
                f"MRLC returned unrecognized NLCD class values: {unknown[:10]}."
            )
        return decoded.astype(np.uint8)

    palette = np.asarray(color_map)
    if palette.shape[0] != 3:
        raise RuntimeError(
            f"MRLC returned an unsupported NLCD color-map shape {palette.shape}."
        )
    palette_rgb = np.rint(palette.astype(float) / 257.0).astype(np.int16).T
    class_colors = np.asarray(list(NLCD_CLASS_COLORS.values()), dtype=np.int16)
    class_codes = np.asarray(list(NLCD_CLASS_COLORS), dtype=np.uint8)
    decoded = np.zeros(values.shape, dtype=np.uint8)
    unknown_indices: list[int] = []
    for index in np.unique(values):
        index_int = int(index)
        if index_int == 0:
            continue
        if index_int < 0 or index_int >= palette_rgb.shape[0]:
            unknown_indices.append(index_int)
            continue
        color = palette_rgb[index_int]
        distances = np.max(np.abs(class_colors - color), axis=1)
        nearest = int(np.argmin(distances))
        if int(distances[nearest]) > NLCD_COLOR_MATCH_TOLERANCE:
            unknown_indices.append(index_int)
            continue
        decoded[values == index] = class_codes[nearest]
    if unknown_indices:
        raise RuntimeError(
            "MRLC returned NLCD palette entries that do not match known class "
            f"colors: {unknown_indices[:10]}."
        )
    return decoded


def _parse_ssurgo_hydrologic_group(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in {"A", "B", "C", "D"}:
        return ord(normalized) - ord("A")
    groups = [group for group in normalized.split("/") if group in {"A", "B", "C", "D"}]
    if groups:
        return max(ord(group) - ord("A") for group in groups)
    return None


def _fetch_ssurgo_hydrologic_soil_grid(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    west, south, east, north = bbox_wgs84
    sample_count = SSURGO_SAMPLE_GRID_SIZE
    sample_points = [
        (
            row,
            column,
            west + (column + 0.5) * (east - west) / sample_count,
            north - (row + 0.5) * (north - south) / sample_count,
        )
        for row in range(sample_count)
        for column in range(sample_count)
    ]
    sql = " UNION ALL ".join(
        (
            f"SELECT '{row}_{column}' AS sample_id, c.hydgrp, c.comppct_r "
            "FROM SDA_Get_Mukey_from_intersection_with_WktWgs84("
            f"'POINT({longitude:.10f} {latitude:.10f})') s "
            "JOIN component c ON s.mukey=c.mukey "
            "WHERE c.majcompflag='Yes'"
        )
        for row, column, longitude, latitude in sample_points
    )
    response = session.post(
        SSURGO_QUERY_URL,
        data={"query": sql, "format": "JSON+ColumnName"},
        timeout=(10, 90),
    )
    response.raise_for_status()
    data = response.json()
    if "Table" not in data or not data["Table"]:
        raise RuntimeError("USDA SSURGO response did not contain a result table.")
    headers = [str(value).lower() for value in data["Table"][0]]
    try:
        sample_column = headers.index("sample_id")
        group_column = headers.index("hydgrp")
        component_column = headers.index("comppct_r")
    except ValueError as exc:
        raise RuntimeError(
            f"USDA SSURGO response has unexpected columns: {headers}."
        ) from exc

    groups_by_sample: dict[tuple[int, int], dict[int, float]] = {}
    for record in data["Table"][1:]:
        sample_parts = str(record[sample_column]).split("_")
        if len(sample_parts) != 2:
            continue
        sample_index = (int(sample_parts[0]), int(sample_parts[1]))
        group = _parse_ssurgo_hydrologic_group(record[group_column])
        if group is None:
            continue
        try:
            component_percent = float(record[component_column] or 0)
        except (TypeError, ValueError):
            component_percent = 0.0
        groups_by_sample.setdefault(sample_index, {}).setdefault(group, 0.0)
        groups_by_sample[sample_index][group] += component_percent

    fallback_group = ord(FALLBACK_HYDROLOGIC_SOIL_GROUP) - ord("A")
    sample_groups = np.full(
        (sample_count, sample_count),
        fallback_group,
        dtype=float,
    )
    used_samples = 0
    group_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for row in range(sample_count):
        for column in range(sample_count):
            weights = groups_by_sample.get((row, column), {})
            if weights:
                group = max(weights, key=weights.get)
                used_samples += 1
            else:
                group = fallback_group
            sample_groups[row, column] = group
            group_counts[chr(ord("A") + group)] += 1

    soil_grid = _resample_continuous_grid(sample_groups, grid_shape)
    return soil_grid, {
        "source": "USDA NRCS SSURGO Soil Data Access",
        "url": SSURGO_QUERY_URL,
        "hydrologic_soil_group_sample_grid": [sample_count, sample_count],
        "samples_with_soil_attributes": used_samples,
        "samples_total": sample_count**2,
        "dominant_group_sample_counts": group_counts,
        "fallback_group": FALLBACK_HYDROLOGIC_SOIL_GROUP,
        "fallback_sample_count": sample_count**2 - used_samples,
        "dual_groups_use_wetter_group": True,
        "interpolation": "bilinear between the 4x4 sampled soil groups",
    }


def _curve_number_grid(
    landcover: np.ndarray,
    soil_groups: np.ndarray,
    water_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    soil_grid = np.asarray(soil_groups, dtype=float)
    if soil_grid.ndim != 2 or water_mask.shape != soil_grid.shape:
        raise ValueError("Soil and water-mask grids must have matching 2D shapes.")
    if (
        not np.isfinite(soil_grid).all()
        or np.any((soil_grid < -1e-9) | (soil_grid > 3 + 1e-9))
    ):
        raise ValueError("Hydrologic soil groups must be values from A=0 through D=3.")
    soil_grid = np.clip(soil_grid, 0, 3)
    source_classes = np.asarray(landcover)
    if source_classes.ndim != 2:
        raise ValueError("Land-cover grid must be two-dimensional.")

    supported_classes = sorted(SCS_CN_BY_NLCD_CLASS)
    fraction_classes = supported_classes + [11]
    fallback_code = FALLBACK_NLCD_CLASS
    source_classes = source_classes.astype(np.int64)
    known_classes = set(fraction_classes)
    unknown_class_mask = ~np.isin(source_classes, list(known_classes))
    missing_classes = {
        int(code)
        for code in np.unique(source_classes)
        if int(code) not in known_classes
    }
    normalized_classes = source_classes.copy()
    if missing_classes:
        normalized_classes[np.isin(normalized_classes, list(missing_classes))] = (
            fallback_code
        )

    class_fractions = np.zeros(
        (len(fraction_classes), *soil_grid.shape),
        dtype=float,
    )
    for index, code in enumerate(fraction_classes):
        source_fraction = (normalized_classes == code).astype(float)
        class_fractions[index] = _resample_continuous_grid(
            source_fraction,
            soil_grid.shape,
        )
    fraction_sum = class_fractions.sum(axis=0)
    missing_fraction = fraction_sum <= 0
    fallback_index = fraction_classes.index(fallback_code)
    class_fractions[fallback_index, missing_fraction] = 1.0
    fraction_sum[missing_fraction] = 1.0
    class_fractions /= fraction_sum
    unknown_class_fraction = _resample_continuous_grid(
        unknown_class_mask.astype(float),
        soil_grid.shape,
    )

    lower_group = np.floor(soil_grid).astype(int)
    upper_group = np.ceil(soil_grid).astype(int)
    group_fraction = soil_grid - lower_group
    curve_numbers = np.zeros(soil_grid.shape, dtype=float)
    for index, code in enumerate(fraction_classes):
        if code == 11:
            class_curve_numbers = np.full(soil_grid.shape, 98.0)
        else:
            values = np.asarray(SCS_CN_BY_NLCD_CLASS[code], dtype=float)
            class_curve_numbers = (
                values[lower_group] * (1 - group_fraction)
                + values[upper_group] * group_fraction
            )
        curve_numbers += class_fractions[index] * class_curve_numbers
    curve_numbers[water_mask] = np.nan
    return curve_numbers, {
        "mapped_class_pixel_count": int(
            np.count_nonzero(np.isin(source_classes, list(known_classes)))
        ),
        "fallback_landcover_pixel_count": int(
            np.count_nonzero((unknown_class_fraction > 0) & ~water_mask)
        ),
        "unmapped_nlcd_classes": sorted(missing_classes),
        "fallback_curve_number": float(
            SCS_CN_BY_NLCD_CLASS[fallback_code][2]
        ),
        "curve_number_basis": (
            "NRCS TR-55-style curve numbers associated with NLCD land-cover classes "
            "and SSURGO hydrologic soil groups; class mappings are generalized, "
            "not locally calibrated."
        ),
        "landcover_interpolation": (
            "bilinearly averaged class fractions from the 30-meter NLCD grid; "
            "fine cells are interpolations, not new observations"
        ),
        "soil_group_interpolation": (
            "bilinearly interpolated hydrologic group, then interpolated curve "
            "numbers between adjacent soil groups"
        ),
    }


def _scs_runoff_depth_grid(
    rainfall_inches: float,
    curve_numbers: np.ndarray,
) -> np.ndarray:
    if not math.isfinite(rainfall_inches) or rainfall_inches < 0:
        raise ValueError("Rainfall must be finite and non-negative.")
    curve_numbers = np.asarray(curve_numbers, dtype=float)
    if np.any(np.isfinite(curve_numbers) & ((curve_numbers <= 0) | (curve_numbers > 100))):
        raise ValueError("Curve numbers must be greater than 0 and at most 100.")
    rainfall_mm = rainfall_inches * 25.4
    storage_mm = np.full(curve_numbers.shape, np.nan, dtype=float)
    valid = np.isfinite(curve_numbers)
    storage_mm[valid] = 25400 / curve_numbers[valid] - 254
    initial_abstraction_mm = 0.2 * storage_mm
    runoff_mm = np.zeros(curve_numbers.shape, dtype=float)
    runoff_mm[valid] = np.maximum(
        rainfall_mm - initial_abstraction_mm[valid],
        0,
    ) ** 2 / (rainfall_mm + 0.8 * storage_mm[valid])
    runoff_mm[~valid] = np.nan
    return runoff_mm / 1000


def _solve_d8_catchment_ponding(
    elevation_grid_meters: np.ndarray,
    runoff_depth_grid_meters: np.ndarray,
    pixel_area_m2: float,
    nodata: float | None,
) -> tuple[np.ndarray, tuple[float, float] | None, dict[str, Any]]:
    elevations = np.asarray(elevation_grid_meters, dtype=float)
    runoff = np.asarray(runoff_depth_grid_meters, dtype=float)
    if elevations.ndim != 2 or runoff.shape != elevations.shape:
        raise ValueError("Elevation and runoff grids must have the same 2D shape.")
    valid = np.isfinite(elevations) & np.isfinite(runoff)
    if nodata is not None:
        valid &= elevations != nodata
    rows, columns = elevations.shape
    receiver = np.full(elevations.size, -1, dtype=np.int64)
    flat_elevations = elevations.ravel()
    flat_valid = valid.ravel()
    for row, column in zip(*np.nonzero(valid)):
        source_index = row * columns + column
        source_elevation = elevations[row, column]
        steepest_slope = 0.0
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if not (
                    0 <= neighbor_row < rows
                    and 0 <= neighbor_column < columns
                    and valid[neighbor_row, neighbor_column]
                ):
                    continue
                drop = source_elevation - elevations[neighbor_row, neighbor_column]
                neighbor_index = neighbor_row * columns + neighbor_column
                if drop < 0 or (drop == 0 and neighbor_index >= source_index):
                    continue
                distance = math.hypot(row_offset, column_offset)
                slope = drop / distance
                if slope > steepest_slope or (
                    slope == 0 and receiver[source_index] < 0
                ):
                    steepest_slope = slope
                    receiver[source_index] = neighbor_index

    accumulated_volume = np.zeros(elevations.size, dtype=float)
    accumulated_volume[flat_valid] = (
        np.maximum(runoff.ravel()[flat_valid], 0) * pixel_area_m2
    )
    ordered_cells = np.flatnonzero(flat_valid)
    ordered_cells = ordered_cells[
        np.lexsort((ordered_cells, flat_elevations[ordered_cells]))
    ]
    terminal = np.full(elevations.size, -1, dtype=np.int64)
    for source_index in ordered_cells:
        downstream_index = receiver[source_index]
        if downstream_index >= 0:
            terminal[source_index] = terminal[downstream_index]
        else:
            terminal[source_index] = source_index

    for source_index in ordered_cells[::-1]:
        downstream_index = receiver[source_index]
        if downstream_index >= 0:
            accumulated_volume[downstream_index] += accumulated_volume[source_index]

    terminal_indices, inverse = np.unique(
        terminal[flat_valid],
        return_inverse=True,
    )
    valid_indices = np.flatnonzero(flat_valid)
    member_order = np.argsort(inverse, kind="stable")
    group_starts = np.r_[0, np.flatnonzero(np.diff(inverse[member_order])) + 1]
    group_ends = np.r_[group_starts[1:], len(member_order)]
    depth_grid = np.full(elevations.shape, np.nan, dtype=float)
    depth_grid[valid] = 0.0
    basin_levels: list[float] = []
    boundary_outflow_volume = 0.0
    basin_volume = 0.0
    basin_overflow_volume = 0.0
    ponded_basins = 0
    sinks = flat_valid & (receiver < 0)
    largest_sink_volume = 0.0

    for group_index, (start, end) in enumerate(zip(group_starts, group_ends)):
        sink_index = int(terminal_indices[group_index])
        sink_volume = float(accumulated_volume[sink_index])
        largest_sink_volume = max(largest_sink_volume, sink_volume)
        sink_row, sink_column = divmod(sink_index, columns)
        members = valid_indices[member_order[start:end]]
        if (
            sink_row == 0
            or sink_column == 0
            or sink_row == rows - 1
            or sink_column == columns - 1
        ):
            boundary_outflow_volume += sink_volume
            continue
        if sink_volume <= 0 or not members.size:
            continue

        basin_elevations = flat_elevations[members]
        spill_candidates: list[float] = []
        for member_index in members:
            row, column = divmod(int(member_index), columns)
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbor_row = row + row_offset
                    neighbor_column = column + column_offset
                    if not (
                        0 <= neighbor_row < rows
                        and 0 <= neighbor_column < columns
                    ):
                        spill_candidates.append(float(flat_elevations[member_index]))
                        continue
                    neighbor_index = neighbor_row * columns + neighbor_column
                    if not flat_valid[neighbor_index]:
                        continue
                    if terminal[neighbor_index] != sink_index:
                        spill_candidates.append(
                            max(
                                float(flat_elevations[member_index]),
                                float(flat_elevations[neighbor_index]),
                            )
                        )
        spill_level = (
            min(spill_candidates)
            if spill_candidates
            else float(basin_elevations.max())
        )
        basin_capacity = float(
            np.maximum(spill_level - basin_elevations, 0).sum()
            * pixel_area_m2
        )
        retained_volume = min(sink_volume, basin_capacity)
        basin_overflow_volume += sink_volume - retained_volume
        if retained_volume <= 0:
            continue
        if retained_volume >= basin_capacity:
            level = spill_level
            basin_depths = np.maximum(level - basin_elevations, 0)
        elif members.size == 1:
            basin_depths = np.array([retained_volume / pixel_area_m2])
            level = float(basin_elevations[0] + basin_depths[0])
        else:
            low = float(basin_elevations.min())
            high = spill_level
            for _ in range(60):
                level = (low + high) / 2
                volume = float(
                    np.maximum(level - basin_elevations, 0).sum()
                    * pixel_area_m2
                )
                if volume < sink_volume:
                    low = level
                else:
                    high = level
            level = (low + high) / 2
            basin_depths = np.maximum(level - basin_elevations, 0)
        depth_grid.ravel()[members] = basin_depths
        basin_levels.append(level)
        basin_volume += retained_volume
        ponded_basins += 1

    runoff_volume = float(
        np.maximum(runoff[valid], 0).sum() * pixel_area_m2
    )
    return depth_grid, (
        (min(basin_levels), max(basin_levels)) if basin_levels else None
    ), {
        "method": "D8 steepest-descent catchment routing and per-sink ponding",
        "routed_runoff_volume_m3": runoff_volume,
        "ponded_catchment_count": ponded_basins,
        "sink_cell_count": int(np.count_nonzero(sinks)),
        "largest_sink_contributing_volume_m3": largest_sink_volume,
        "ponded_volume_m3": basin_volume,
        "interior_basin_overflow_m3": basin_overflow_volume,
        "boundary_outflow_volume_m3": boundary_outflow_volume,
        "unaccounted_volume_m3": max(
            runoff_volume
            - basin_volume
            - boundary_outflow_volume
            - basin_overflow_volume,
            0.0,
        ),
        "limitations": (
            "D8 routes each cell to one downslope neighbor; per-sink ponding is "
            "a static catchment approximation, not a time-dependent hydraulic "
            "simulation. Flat cells use a deterministic index tie-break. Basin "
            "overflow is counted as outflow, not routed into downstream basins."
        ),
    }


def _solve_retained_rainfall_surface(
    elevation_grid_meters: np.ndarray,
    bounds_metric: tuple[float, float, float, float],
    rainfall_inches: float,
    retained_water_fraction: float,
    nodata: float | None,
    water_mask: np.ndarray | None = None,
) -> tuple[float | None, np.ndarray, float]:
    values = np.asarray(elevation_grid_meters, dtype=float)
    if values.ndim != 2:
        raise ValueError("Elevation grid must be two-dimensional.")
    if not math.isfinite(rainfall_inches) or rainfall_inches < 0:
        raise ValueError("Rainfall must be finite and non-negative.")
    if (
        not math.isfinite(retained_water_fraction)
        or not 0 <= retained_water_fraction <= 1
    ):
        raise ValueError("Retained-water fraction must be between 0 and 1.")
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    if not valid.any():
        raise ValueError("DEM has no valid elevation pixels in the requested box.")
    if water_mask is not None:
        if water_mask.shape != values.shape:
            raise ValueError("Water-body mask must match the elevation grid shape.")
        valid &= ~water_mask
    elevations = values[valid]
    xmin, ymin, xmax, ymax = bounds_metric
    valid_pixels = int(np.count_nonzero(valid))
    pixel_area_m2 = (xmax - xmin) * (ymax - ymin) / values.size
    depth_grid = np.full(values.shape, np.nan, dtype=float)
    if elevations.size == 0:
        return None, depth_grid, pixel_area_m2
    target_volume_m3 = (
        rainfall_inches * 0.0254 * retained_water_fraction * pixel_area_m2 * valid_pixels
    )
    low = float(np.min(elevations))
    high = float(np.max(elevations) + target_volume_m3 / pixel_area_m2 + 1)
    for _ in range(80):
        level = (low + high) / 2
        volume = float(np.maximum(level - elevations, 0).sum() * pixel_area_m2)
        if volume < target_volume_m3:
            low = level
        else:
            high = level
    surface = (low + high) / 2
    depth_grid[valid] = np.maximum(surface - values[valid], 0)
    return surface, depth_grid, pixel_area_m2


def _waterbody_raster_mask(
    waterbodies: list[dict[str, Any]],
    bounds_metric: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    rows, columns = grid_shape
    xmin, ymin, xmax, ymax = bounds_metric
    x_coordinates = xmin + (np.arange(columns) + 0.5) * (xmax - xmin) / columns
    y_coordinates = ymax - (np.arange(rows) + 0.5) * (ymax - ymin) / rows
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    mask = np.zeros(grid_shape, dtype=bool)
    for waterbody in waterbodies:
        mask |= contains_xy(waterbody["geometry"], x_grid, y_grid)
    return mask


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
            timeout=(2, 2),
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


def _navd88_to_lmsl_offset(
    session: requests.Session,
    latitude: float,
    longitude: float,
    source_datum: str | None,
) -> tuple[float, dict[str, Any]]:
    if not source_datum or "NAVD88" not in source_datum.upper().replace(" ", ""):
        return 0.0, {
            "status": "unsupported_source_datum",
            "source_datum": source_datum or "unknown",
            "target_datum": "LMSL",
            "applied_offset_meters": 0.0,
        }
    try:
        response = session.get(
            NOAA_VDATUM_URL,
            params={
                "s_x": longitude,
                "s_y": latitude,
                "s_z": 0,
                "s_h_frame": "NAD83_2011",
                "s_coor": "geo",
                "s_v_frame": "NAVD88",
                "s_v_unit": "m",
                "s_v_elevation": "height",
                "t_v_frame": "LMSL",
                "t_v_unit": "m",
            },
            timeout=(10, 30),
        )
        response.raise_for_status()
        result = response.json()
        offset = float(result["t_z"])
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return 0.0, {
            "status": "conversion_service_error",
            "source_datum": source_datum,
            "target_datum": "LMSL",
            "applied_offset_meters": 0.0,
            "error": str(exc),
        }
    if not math.isfinite(offset) or abs(offset) >= 999_000:
        return 0.0, {
            "status": "unsupported_location",
            "source_datum": source_datum,
            "target_datum": "LMSL",
            "applied_offset_meters": 0.0,
            "response_region": result.get("region"),
            "error": "NOAA VDatum reports no valid transformation at this location.",
        }
    return offset, {
        "status": "converted",
        "source_datum": source_datum,
        "target_datum": "LMSL",
        "applied_offset_meters": offset,
        "uncertainty_meters": (
            float(result["uncertainty"])
            if result.get("uncertainty") not in (None, "")
            else None
        ),
        "source": NOAA_VDATUM_URL,
    }


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
    rainfall_inches: float | None = None,
    rainfall_intensity_inches_per_hour: float | None = None,
    rainfall_duration_hours: float | None = None,
    retained_water_fraction: float | None = None,
    groundsource_parquet: Path = GROUND_SOURCE_PATH,
    cache_dir: Path = Path(__file__).parent / "cache",
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a screening rainfall-runoff estimate and source diagnostics.

    The requested schema has no field for uncertainty or provenance, so those
    details are returned in diagnostics and documented alongside the payload.
    """
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
                "3DEP vertical datum metadata",
                _spatial_cache_key(
                    "dem_datum",
                    metric_crs,
                    latitude,
                    longitude,
                ),
                VDATUM_CACHE_TTL_SECONDS,
                session,
                lambda active_session: _dem_vertical_datum(
                    active_session,
                    latitude,
                    longitude,
                    metric_crs,
                ),
            ),
            "usgs": executor.submit(
                _cached_source_job,
                "USGS WDFN",
                _spatial_cache_key(
                    "usgs",
                    round(latitude, 4),
                    round(longitude, 4),
                    USGS_SEARCH_RADIUS_KM,
                ),
                USGS_CACHE_TTL_SECONDS,
                session,
                lambda active_session: run_usgs(
                    StudyArea(
                        latitude=latitude,
                        longitude=longitude,
                        radius_km=USGS_SEARCH_RADIUS_KM,
                    ),
                    session=active_session,
                ),
            ),
            "groundsource": executor.submit(
                _groundsource_job,
                bbox_wgs84,
                groundsource_parquet,
                cache_dir,
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
        dem_datum, datum_note = datum_result
        usgs_result, source_timings["usgs"] = futures["usgs"].result()
        (groundsource, scan_notes), source_timings["groundsource"] = (
            futures["groundsource"].result()
        )
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
    latest_daily_result, source_timings["usgs_latest_daily"] = _cached_source_job(
        "USGS latest-daily gage height",
        _spatial_cache_key(
            "usgs_latest_daily",
            tuple(station["station_id"] for station in streams),
        ),
        USGS_CACHE_TTL_SECONDS,
        session,
        lambda active_session: _latest_daily_stage(active_session, streams),
    )
    latest_daily, daily_errors = latest_daily_result
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
    vertical_result, source_timings["vdatum_conversion"] = _cached_source_job(
        "NOAA VDatum conversion",
        _spatial_cache_key(
            "vdatum_conversion",
            round(latitude, 5),
            round(longitude, 5),
            dem_datum,
        ),
        VDATUM_CACHE_TTL_SECONDS,
        session,
        lambda active_session: _navd88_to_lmsl_offset(
            active_session,
            latitude,
            longitude,
            dem_datum,
        ),
    )
    vertical_offset_meters, vertical_conversion = vertical_result
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
        "This is a screening scenario, not an observed or locally calibrated "
        "street-flood forecast.",
        "The depth model routes runoff down a D8 terrain network and balances "
        "volume within each interior sink catchment; it does not simulate "
        "time-dependent hydraulics, streets, pipes, culverts, or lake exchange.",
        "The DEM is expanded beyond the output bbox to include nearby terrain; "
        "the returned contours are clipped back to the requested bbox.",
        "30-meter NLCD classes and sparse SSURGO soil samples are interpolated for "
        "the finer terrain grid; interpolation adds no new source observations.",
        "USGS gage stage is reported as context only and is not transferred from a "
        "distant gage to the local flood surface.",
        "Groundsource event footprints are recent historical event evidence, not "
        "point-depth labels and are not used to fit a depth predictor.",
        f"Contour rings are simplified to at most {MAX_CONTOUR_VERTICES} coordinates, "
        "including the closing coordinate; holes are omitted by the payload schema.",
        "Mapped surface-water polygons are served separately by /waterbodies and "
        "are not encoded as zero-depth flood contours.",
        "Coastal storm-surge products and FEMA hazard zones are not converted to local "
        "rainfall depth. Surge needs a coastal boundary condition and compatible datum.",
    ]
    recent_daily = [
        observation
        for observation in latest_daily
        if observation["age_days"] is not None
        and 0 <= observation["age_days"] <= 7
    ]
    if not stage_observations and not recent_daily:
        notes.append(
            "No current local WDFN gage-height observation is available; the most "
            "recent daily stage is historical and is not used to compute this estimate."
        )
    if groundsource.get("events_intersecting_area", 0):
        notes.append(
            "Groundsource has historical event footprints in this box but no "
            "point water-level or depth observations."
        )
    if waterbody_notes:
        notes.extend(waterbody_notes)
    notes.extend(source_notes)
    if vertical_conversion["status"] != "converted":
        notes.append(
            "NOAA VDatum could not convert this location to LMSL; absolute elevation "
            "values remain in the source DEM datum identified in ar_flood_data."
        )
    if point_is_waterbody:
        notes.append(
            "The requested point lies in a mapped surface-water polygon. The DEM value "
            "there is treated as a water-surface sample, not submerged ground elevation."
        )
    if not np.isfinite(depth_grid).any():
        notes.append(
            "No land DEM cells remain after masking mapped water bodies; rainfall "
            "ponding depth is not estimated for this box."
        )

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
            "calibrated_for_location": False,
            "calibration_status": (
                "not_calibrated_no_point_depth_labels_available"
            ),
            "retained_water_fraction": retained_water_fraction,
            "effective_rainfall_inches": effective_rainfall_inches,
            "effective_runoff_fraction": model_runoff_fraction,
            "effective_runoff_volume_m3": runoff_volume_m3,
            "modeled_water_surface_native_datum_meters": water_surface_datum,
            "modeled_basin_surface_level_range_native_datum_meters": (
                list(basin_surface_level_range)
                if basin_surface_level_range
                else None
            ),
            "point_depth_meters_unrounded": point_depth,
            "maximum_grid_depth_meters": (
                float(np.nanmax(depth_grid))
                if np.isfinite(depth_grid).any()
                else 0.0
            ),
            "pixel_area_m2": pixel_area_m2,
            "masked_waterbody_pixel_count": int(np.count_nonzero(water_mask)),
            "d8_runoff_routing": routing_summary,
            "curve_number_summary": curve_number_summary,
            "contour_thresholds_meters": list(DEPTH_THRESHOLDS_METERS),
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
            "metric_crs": f"EPSG:{metric_crs}",
            "raster_shape_rows_columns": list(dem_values.shape),
            "requested_bbox_wgs84": list(bbox_wgs84),
            "analysis_bbox_wgs84": list(analysis_bbox_wgs84),
            "terrain_context_buffer_meters": TERRAIN_CONTEXT_BUFFER_METERS,
            "pixel_resolution_meters": [
                (dem_bounds[2] - dem_bounds[0]) / dem_values.shape[1],
                (dem_bounds[3] - dem_bounds[1]) / dem_values.shape[0],
            ],
            "sampled_ground_elevation_meters_unconverted": dem_sample,
            "vertical_datum": dem_datum,
            "vertical_datum_note": datum_note,
            "payload_elevation_datum": actual_output_datum,
            "vertical_datum_conversion": vertical_conversion,
        },
        "landcover": {
            **nlcd_source,
            "fallback_class_pixel_count": curve_number_summary[
                "fallback_landcover_pixel_count"
            ],
            "unmapped_nlcd_classes": curve_number_summary["unmapped_nlcd_classes"],
        },
        "soils": ssurgo_source,
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
        "data_source_roles": {
            "usgs_wdfn_v1": "Live/cached station and gage observations provide context; unavailable/stale gage data is not extrapolated to the bbox.",
            "usgs_3dep": "Supplies terrain elevations for DEM-based ponding screening.",
            "groundsource": (
                "2026 historical flood-event footprints provide recent event "
                "context and candidate hindcast areas; they contain no depth labels."
            ),
            "mrlc_nlcd": (
                "Land-cover class raster supplies land-surface categories used "
                "to select generalized runoff curve numbers."
            ),
            "usda_ssurgo": (
                "Spatially sampled hydrologic soil groups contribute to the runoff "
                "curve-number calculation; missing coverage falls back to average "
                f"group {FALLBACK_HYDROLOGIC_SOIL_GROUP}."
            ),
            "surface_water": (
                "USGS NHD at broader scales and OpenStreetMap in small boxes identify "
                "existing waterbody polygons; these are not flood-depth observations."
            ),
            "fema_nfhl": (
                "Not queried by this scenario function; mapped hazard zones must remain "
                "a separate hazard-context overlay, not rainfall depth input."
            ),
            "noaa_storm_surge": (
                "Not applied to these inland Gainesville examples. Coastal surge needs "
                "a spatial water-level boundary condition and vertical-datum conversion."
            ),
        },
        "surface_water": {
            "mapped_waterbody_count": len(waterbodies),
            "features": [
                {
                    "name": waterbody["name"],
                    "source": waterbody["source"],
                    "feature_id": waterbody["feature_id"],
                }
                for waterbody in waterbodies
            ],
            "notes": waterbody_notes,
        },
        "groundsource": {
            "source": groundsource.get("source"),
            "dataset_title": "Groundsource: A Dataset of Flood Events from News",
            "doi": "https://doi.org/10.5281/zenodo.18647054",
            "publication_date": "2026-02-15",
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
        "interpretation": (
            "The rainfall excess is estimated with a generalized SCS curve-number method "
            "using land-cover and soil layers, then routed on a D8 terrain network "
            "to static, spill-capped sink catchments. It is not locally calibrated "
            "and does not simulate time-dependent drainage or full hydraulic flow; "
            "it is not suitable for life-safety decisions."
        ),
    }
    return payload, diagnostics


def write_reitz_example(
    output_dir: Path,
    rainfall_inches: float = DEFAULT_RAINFALL_INCHES,
    retained_water_fraction: float | None = None,
) -> tuple[Path, Path]:
    bbox = reitz_bbox()
    payload, diagnostics = build_flood_payload(
        REITZ_UNION["latitude"],
        REITZ_UNION["longitude"],
        bbox,
        rainfall_inches=rainfall_inches,
        retained_water_fraction=retained_water_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "reitz_union_flood.json"
    diagnostics_path = output_dir / "reitz_union_diagnostics.json"
    payload_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with requests.Session() as session:
        waterbody_path = output_dir / "reitz_union_waterbodies.geojson"
        waterbody_path.write_text(
            json.dumps(
                _waterbody_feature_collection(session, bbox),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return payload_path, diagnostics_path


def write_lake_alice_example(
    output_dir: Path,
    rainfall_inches: float = DEFAULT_RAINFALL_INCHES,
    retained_water_fraction: float | None = None,
) -> tuple[Path, Path]:
    bbox = lake_alice_bbox()
    payload, diagnostics = build_flood_payload(
        LAKE_ALICE["latitude"],
        LAKE_ALICE["longitude"],
        bbox,
        rainfall_inches=rainfall_inches,
        retained_water_fraction=retained_water_fraction,
    )
    diagnostics["location_label"] = "Lake Alice, Gainesville, Florida"
    diagnostics["coordinate_source"] = {
        "source": "OpenStreetMap Lake Alice water-body relation",
        "relation_id": LAKE_ALICE["osm_relation_id"],
        "url": f"https://www.openstreetmap.org/relation/{LAKE_ALICE['osm_relation_id']}",
        "selection": (
            "Representative point computed from the mapped Lake Alice polygon; "
            "point-in-polygon check passed."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "lake_alice_flood.json"
    diagnostics_path = output_dir / "lake_alice_diagnostics.json"
    payload_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with requests.Session() as session:
        waterbody_path = output_dir / "lake_alice_waterbodies.geojson"
        waterbody_path.write_text(
            json.dumps(
                _waterbody_feature_collection(session, bbox),
                indent=2,
                allow_nan=False,
            ),
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
                rainfall_inches = body.get("rainfall_inches")
                rainfall_intensity = body.get(
                    "rainfall_intensity_inches_per_hour"
                )
                rainfall_duration = body.get("rainfall_duration_hours")
                retained_water_fraction = body.get(
                    "retained_water_fraction"
                )
                payload, _ = build_flood_payload(
                    latitude,
                    longitude,
                    bbox,
                    rainfall_inches=rainfall_inches,
                    rainfall_intensity_inches_per_hour=rainfall_intensity,
                    rainfall_duration_hours=rainfall_duration,
                    retained_water_fraction=retained_water_fraction,
                )
                self._write_json(200, payload)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self._write_json(400, {"error": str(exc)})
            except (OSError, requests.RequestException, RuntimeError) as exc:
                self._write_json(503, {"error": str(exc)})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/waterbodies":
                self.send_error(404, "GET endpoint must be /waterbodies.")
                return
            try:
                raw_bbox = parse_qs(parsed.query)["bbox_wgs84"][0]
                bbox = tuple(float(value) for value in raw_bbox.split(","))
                if len(bbox) != 4:
                    raise ValueError(
                        "bbox_wgs84 must contain west,south,east,north."
                    )
                west, south, east, north = bbox
                _validate_request(
                    (south + north) / 2,
                    (west + east) / 2,
                    bbox,
                )
                collection = _waterbody_feature_collection(
                    requests.Session(),
                    bbox,
                )
                self._write_json(200, collection)
            except (KeyError, TypeError, ValueError) as exc:
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
        "--rainfall-inches",
        type=float,
        default=DEFAULT_RAINFALL_INCHES,
    )
    parser.add_argument(
        "--retained-water-fraction",
        type=float,
        default=None,
        help=(
            "Optional legacy override; otherwise use the NLCD/SSURGO curve-number "
            "runoff model."
        ),
    )
    parser.add_argument(
        "--location",
        choices=("reitz", "lake-alice"),
        default="reitz",
        help="Named location for the generated JSON example.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
    )
    args = parser.parse_args()
    if args.serve:
        serve(args.host, args.port)
        return
    writer = (
        write_lake_alice_example
        if args.location == "lake-alice"
        else write_reitz_example
    )
    payload_path, diagnostics_path = writer(
        args.output_dir, args.rainfall_inches, args.retained_water_fraction
    )
    print(f"Wrote payload: {payload_path}")
    print(f"Wrote diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
