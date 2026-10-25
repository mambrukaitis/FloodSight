"""Geometry helpers shared by every stage: bounding boxes, the local UTM projection,
the terrain-context analysis domain, grid resampling, bilinear point sampling and
request validation (point inside a Florida-sized bbox).
"""

from __future__ import annotations

import math

import numpy as np
from pyproj import Transformer
from shapely import transform
from shapely.geometry import box, Polygon


DEFAULT_BBOX_SIDE_METERS = 100.0
MAX_BBOX_EXTENT_DEGREES = 7.0
TERRAIN_CONTEXT_BUFFER_METERS = 250.0
MAX_CONTOUR_VERTICES = 15


def bbox_around_point(
    latitude: float,
    longitude: float,
    side_meters: float = DEFAULT_BBOX_SIDE_METERS,
) -> tuple[float, float, float, float]:
    """Return a WGS84 (west, south, east, north) box of the given side length centred
    on a point.
    """
    if not math.isfinite(side_meters) or side_meters <= 0:
        raise ValueError("Bounding-box side length must be positive and finite.")
    metric_crs = local_utm_epsg(latitude, longitude)
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


def analysis_domain(
    bbox_wgs84: tuple[float, float, float, float],
    metric_crs: int,
    buffer_meters: float = TERRAIN_CONTEXT_BUFFER_METERS,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Expand the requested bbox by the terrain-context buffer; returns the expanded
    box in WGS84 and in metric (UTM) coordinates.
    """
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


def resample_continuous_grid(
    values: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Bilinearly resample a 2-D grid to a new shape so coarse source data (soil, land
    cover) matches the DEM grid.
    """
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


def local_utm_epsg(latitude: float, longitude: float) -> int:
    """Return the EPSG code of the UTM zone containing the given latitude/longitude."""
    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    return (32600 if latitude >= 0 else 32700) + zone


def validate_request(
    latitude: float,
    longitude: float,
    bbox_wgs84: tuple[float, float, float, float],
) -> None:
    """Raise ValueError unless the point is a valid coordinate inside an ordered,
    positive-area bbox no larger than MAX_BBOX_EXTENT_DEGREES.
    """
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
            f"Bounding-box dimensions cannot exceed {MAX_BBOX_EXTENT_DEGREES}Â°."
        )


def bilinear_sample(
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    longitude: float,
    latitude: float,
    metric_crs: int,
    nodata: float | None,
) -> float:
    """Bilinearly sample a raster at a WGS84 point, ignoring nodata cells; returns a
    float.
    """
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


def polygon_ring_coordinates(
    polygon: Polygon,
    max_vertices: int = MAX_CONTOUR_VERTICES,
) -> list[tuple[float, float]]:
    """Convert a projected polygon to a closed ring of (lat, lng) points simplified to
    at most max_vertices.
    """
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
