"""Remote and local data sources: 3DEP, NLCD, SSURGO, surface water, datums, Groundsource, plus request caching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import pickle
import threading
from time import perf_counter
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq
import requests
import shapely
import tifffile
from pyproj import Transformer
from shapely import STRtree, transform
from shapely.geometry import LineString, Polygon, box, shape
from shapely.ops import polygonize, unary_union

from hydrology import (
    _local_utm_epsg,
    _polygon_ring_coordinates,
    _resample_continuous_grid,
    _validate_request,
)

THREEDEP_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)
THREEDEP_IDENTIFY_URL = THREEDEP_EXPORT_URL.rsplit("/", 1)[0] + "/identify"

MAX_DEM_PIXELS = 50_000

CACHE_DIR = Path(os.environ.get("FLOODSIGHT_CACHE_DIR") or Path(__file__).parent / ".cache")
GROUND_SOURCE_PATH = CACHE_DIR / "groundsource" / "groundsource_2026.parquet"
GROUNDSOURCE_BUFFER_DEGREES = 0.015  # ~1.5 km of surrounding historical context
GROUNDSOURCE_FLORIDA_BOUNDS = (-88.0, 24.0, -79.5, 31.5)
_GROUNDSOURCE_INDEX: dict[tuple[str, int, int], tuple[Any, np.ndarray]] = {}
_GROUNDSOURCE_INDEX_LOCK = threading.Lock()

NOAA_VDATUM_URL = "https://vdatum.noaa.gov/vdatumweb/api/convert"
NHD_MAPSERVER_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
OSM_API_URL = "https://api.openstreetmap.org/api/0.6"
MRLC_WMS_URL = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
MRLC_NLCD_LAYER = "NLCD_2021_Land_Cover_L48"
SSURGO_QUERY_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
SSURGO_SAMPLE_GRID_SIZE = 4
FALLBACK_HYDROLOGIC_SOIL_GROUP = "C"

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

_SPATIAL_CACHE: dict[tuple[Any, ...], tuple[float, Any]] = {}
_SPATIAL_CACHE_LOCK = threading.Lock()

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

def _groundsource_florida_events(parquet_path: Path) -> tuple[Any, np.ndarray]:
    """Return an STRtree over Florida flood-event footprints, built once per archive."""
    stat = parquet_path.stat()
    key = (str(parquet_path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _GROUNDSOURCE_INDEX_LOCK:
        cached = _GROUNDSOURCE_INDEX.get(key)
        if cached is not None:
            return cached
        sidecar = parquet_path.with_name(
            f"{parquet_path.stem}_florida_{stat.st_size}_{stat.st_mtime_ns}.pkl"
        )
        if sidecar.is_file():
            with sidecar.open("rb") as handle:
                wkb_values = pickle.load(handle)
        else:
            florida = box(*GROUNDSOURCE_FLORIDA_BOUNDS)
            wkb_values = []
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(
                batch_size=65_536, columns=["geometry"]
            ):
                column = batch.column("geometry").to_pylist()
                geometries = shapely.from_wkb(
                    np.array(column, dtype=object), on_invalid="ignore"
                )
                bounds = shapely.bounds(geometries)
                hit = (
                    (bounds[:, 0] <= florida.bounds[2])
                    & (bounds[:, 2] >= florida.bounds[0])
                    & (bounds[:, 1] <= florida.bounds[3])
                    & (bounds[:, 3] >= florida.bounds[1])
                )
                wkb_values.extend(
                    shapely.to_wkb(geometry)
                    for geometry in geometries[np.nan_to_num(hit, nan=0).astype(bool)]
                )
            tmp = sidecar.with_suffix(".tmp")
            with tmp.open("wb") as handle:
                pickle.dump(wkb_values, handle)
            tmp.replace(sidecar)
        geometries = shapely.from_wkb(np.array(wkb_values, dtype=object))
        geometries = np.array(
            [
                geometry if geometry.is_valid else shapely.make_valid(geometry)
                for geometry in geometries
            ],
            dtype=object,
        )
        index = (STRtree(geometries), geometries)
        _GROUNDSOURCE_INDEX[key] = index
        return index


def _groundsource_event_count(
    bbox_wgs84: tuple[float, float, float, float],
    parquet_path: Path,
) -> dict[str, Any]:
    """Count Groundsource event footprints overlapping the box (positive-area overlap)."""
    if not parquet_path.is_file():
        return {"events_intersecting_area": 0}
    tree, geometries = _groundsource_florida_events(parquet_path)
    query = box(*bbox_wgs84)
    candidates = tree.query(query, predicate="intersects")
    count = sum(
        1
        for position in candidates
        if shapely.intersection(geometries[position], query).area > 0
    )
    return {"events_intersecting_area": count}


def _buffered_bbox(
    bbox: tuple[float, float, float, float],
    buffer_degrees: float,
) -> tuple[float, float, float, float]:
    return (
        bbox[0] - buffer_degrees,
        bbox[1] - buffer_degrees,
        bbox[2] + buffer_degrees,
        bbox[3] + buffer_degrees,
    )


def _groundsource_job(
    bbox_wgs84: tuple[float, float, float, float],
    parquet_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = perf_counter()
    try:
        result = _groundsource_event_count(bbox_wgs84, parquet_path)
        error = None
    except Exception as exc:
        result = {"events_intersecting_area": 0}
        error = str(exc)
    timing = {
        "elapsed_seconds": perf_counter() - started,
        "cache": "index",
        "source": "Groundsource",
    }
    if error:
        timing["error"] = error
    return result, timing

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
    except (tifffile.TiffFileError, ValueError, IndexError, AttributeError) as exc:
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


def _datum_and_offset(
    session: requests.Session,
    latitude: float,
    longitude: float,
    metric_crs: int,
) -> tuple[tuple[str | None, str | None], tuple[float, dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        # 3DEP is NAVD88 for Florida, so convert speculatively while the datum
        # metadata is inspected; the result is only used if the datum agrees.
        speculative = executor.submit(
            _navd88_to_lmsl_offset, session, latitude, longitude, "NAVD88"
        )
        datum = _dem_vertical_datum(session, latitude, longitude, metric_crs)
        offset = speculative.result()
    if datum[0] is None and str(datum[1]).startswith("Could not inspect"):
        # Metadata service timed out; USGS 3DEP elevations are NAVD88 across Florida.
        note = f"{datum[1]} NAVD88 was assumed for the 3DEP DEM."
        return ("North American Vertical Datum of 1988 (NAVD 88)", note), offset
    if datum[0] and "NAVD88" in datum[0].upper().replace(" ", ""):
        return datum, offset
    return datum, _navd88_to_lmsl_offset(session, latitude, longitude, datum[0])
