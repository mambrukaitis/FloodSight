"""Geometry helpers and the rainfall-runoff / terrain ponding model (no network access)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely import contains_xy, transform
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


DEFAULT_BBOX_SIDE_METERS = 100.0

DEPTH_THRESHOLDS_METERS = (0.1, 0.2, 0.3, 1.0)

MAX_CONTOUR_VERTICES = 15

MAX_BBOX_EXTENT_DEGREES = 7.0

TERRAIN_CONTEXT_BUFFER_METERS = 250.0

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
            f"Bounding-box dimensions cannot exceed {MAX_BBOX_EXTENT_DEGREES}Â°."
        )

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
