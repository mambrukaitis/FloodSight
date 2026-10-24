"""Converts a modeled water-depth grid into the depth-threshold contour polygons (0.1,
0.2, 0.3, 1.0 m) returned to the app, clipped to the requested map extent and
simplified to a few vertices.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely import transform
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

from floodsight.grid_geometry import MAX_CONTOUR_VERTICES, polygon_ring_coordinates


DEPTH_THRESHOLDS_METERS = (0.1, 0.2, 0.3, 1.0)


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
            ring = polygon_ring_coordinates(polygon, max_vertices)
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
