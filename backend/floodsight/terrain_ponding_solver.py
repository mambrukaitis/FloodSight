"""Terrain-based water solvers: D8 flow routing that ponds runoff in interior sinks
(default model), the legacy closed-box retained-rainfall solver, and the mapped-
water raster mask.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely import contains_xy


def solve_d8_catchment_ponding(
    elevation_grid_meters: np.ndarray,
    runoff_depth_grid_meters: np.ndarray,
    pixel_area_m2: float,
    nodata: float | None,
) -> tuple[np.ndarray, tuple[float, float] | None, dict[str, Any]]:
    """Route runoff over the DEM with D8 flow directions and pond it in interior sinks;
    returns (depth grid, basin level range, routing summary).
    """
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


def solve_retained_rainfall_surface(
    elevation_grid_meters: np.ndarray,
    bounds_metric: tuple[float, float, float, float],
    rainfall_inches: float,
    retained_water_fraction: float,
    nodata: float | None,
    water_mask: np.ndarray | None = None,
) -> tuple[float | None, np.ndarray, float]:
    """Legacy solver: fill the DEM box to one water level holding a fixed fraction of
    the rain; returns (water surface, depth grid, pixel area).
    """
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


def waterbody_raster_mask(
    waterbodies: list[dict[str, Any]],
    bounds_metric: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize mapped waterbody polygons to a boolean mask on the DEM grid."""
    rows, columns = grid_shape
    xmin, ymin, xmax, ymax = bounds_metric
    x_coordinates = xmin + (np.arange(columns) + 0.5) * (xmax - xmin) / columns
    y_coordinates = ymax - (np.arange(rows) + 0.5) * (ymax - ymin) / rows
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    mask = np.zeros(grid_shape, dtype=bool)
    for waterbody in waterbodies:
        mask |= contains_xy(waterbody["geometry"], x_grid, y_grid)
    return mask
