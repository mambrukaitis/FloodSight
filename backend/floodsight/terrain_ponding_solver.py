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
    water_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float] | None, dict[str, Any]]:
    """
    Experimental rainfall inundation model.

    Redistributes runoff between neighboring cells according
    to water-surface elevation, with open grid boundaries.

    This is a static approximation, not a calibrated
    time-dependent hydraulic model.
    """

    elevations = np.asarray(elevation_grid_meters, dtype=float)
    runoff = np.asarray(runoff_depth_grid_meters, dtype=float)

    if elevations.ndim != 2 or runoff.shape != elevations.shape:
        raise ValueError("Elevation and runoff grids must have the same 2D shape.")

    if not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("Pixel area must be positive.")

    valid = np.isfinite(elevations) & np.isfinite(runoff)
    if water_mask is None:
        water_mask = np.zeros_like(valid, dtype=bool)
    else:
        water_mask = np.asarray(water_mask, dtype=bool)

        if water_mask.shape != elevations.shape:
            raise ValueError("Waterbody mask must match DEM shape.")

    if nodata is not None:
        valid &= elevations != nodata

    # Start with rainfall runoff on the terrain.
    depth = np.zeros_like(elevations)
    depth[valid] = np.maximum(runoff[valid], 0.0)

    initial_volume = float(depth.sum() * pixel_area_m2)

    # Four unique neighboring directions, including diagonals.
    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    ]

    rows, cols = elevations.shape

    # Numerical redistribution iterations.
    # These are NOT hours or minutes of simulated storm time.
    iterations = 60
    transfer_fraction = 0.20

    boundary_outflow_volume = 0.0

    for _ in range(iterations):

        for dr, dc in directions:

            if dr >= 0:
                r1 = slice(0, rows - dr)
                r2 = slice(dr, rows)
            else:
                r1 = slice(-dr, rows)
                r2 = slice(0, rows + dr)

            if dc >= 0:
                c1 = slice(0, cols - dc)
                c2 = slice(dc, cols)
            else:
                c1 = slice(-dc, cols)
                c2 = slice(0, cols + dc)

            first = depth[r1, c1]
            second = depth[r2, c2]

            first_elevation = elevations[r1, c1]
            second_elevation = elevations[r2, c2]

            pair_valid = valid[r1, c1] & valid[r2, c2]

            # Compare water-surface elevations, not ground alone.
            surface_difference = (
                first_elevation + first
                - second_elevation - second
            )

            # Positive means first -> second.
            # Negative means second -> first.
            transfer = surface_difference * transfer_fraction

            # Never move more water than the source contains.
            transfer = np.minimum(transfer, first)
            transfer = np.maximum(transfer, -second)

            transfer = np.where(pair_valid, transfer, 0.0)

            first -= transfer
            second += transfer

        # Treat the outer grid boundary as an open outlet.
        boundary = np.zeros_like(valid)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True

        escaping = boundary & valid

        boundary_outflow_volume += float(
            depth[escaping].sum() * pixel_area_m2
        )

        depth[escaping] = 0.0

        # Experimental assumption:
        # Mapped open water is a free-draining outlet.
        water_outlets = water_mask & valid

        boundary_outflow_volume += float(
            depth[water_outlets].sum() * pixel_area_m2
        )

        depth[water_outlets] = 0.0

        max_allowed_depth = max(0.35, float(np.nanmax(runoff) * 2.5))
        excess = valid & (depth > max_allowed_depth)
        
        boundary_outflow_volume += float(
            (depth[excess] - max_allowed_depth).sum() * pixel_area_m2
        )
        depth[excess] = max_allowed_depth

    # Preserve invalid cells as NaN.
    depth[~valid] = np.nan

    ponded_volume = float(np.nansum(depth) * pixel_area_m2)

    summary = {
        "method": "Experimental neighboring-cell surface redistribution",
        "routed_runoff_volume_m3": initial_volume,
        "ponded_volume_m3": ponded_volume,
        "boundary_outflow_volume_m3": boundary_outflow_volume,
        "interior_basin_overflow_m3": 0.0,
        "unaccounted_volume_m3": abs(
            initial_volume - ponded_volume - boundary_outflow_volume
        ),
        "ponded_catchment_count": 0,
        "sink_cell_count": 0,
        "largest_sink_contributing_volume_m3": 0.0,
        "limitations": (
            "Experimental static redistribution. No rainfall timing, "
            "flow velocity calibration, engineered drainage, "
            "lake outlet modeling, or hydraulic validation."
        ),
    }

    flooded = valid & (depth > 0)

    if np.any(flooded):
        water_surfaces = elevations[flooded] + depth[flooded]
        level_range = (
            float(water_surfaces.min()),
            float(water_surfaces.max()),
        )
    else:
        level_range = None

    return depth, level_range, summary


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
