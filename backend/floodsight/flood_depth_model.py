"""Turns fetched terrain, land cover, soils and rainfall into a flood-depth grid.

Two modes exist. The default computes SCS curve-number runoff from NLCD land
cover and SSURGO soil groups, scales it by a Groundsource-based drainage
multiplier, and routes it with D8 into per-sink ponding. The legacy mode
(chosen only when `retained_water_fraction` is given) fills a closed DEM box
with a fixed fraction of the rain. `run_depth_model` returns the depth grid and
the summary numbers reported in the diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np

from floodsight.grid_geometry import resample_continuous_grid
from floodsight.scs_runoff import curve_number_grid, scs_runoff_depth_grid
from floodsight.source_fetching import SourceData
from floodsight.terrain_ponding_solver import (
    solve_d8_catchment_ponding,
    solve_retained_rainfall_surface,
)

MAX_DRAINAGE_MULTIPLIER = 1.25
EVENT_SCALAR = 0.0145  # Reaches ~1.13 at 80 events and ~1.25 at 300 events
METERS_TO_INCHES = 39.3700787402
INCHES_TO_METERS = 0.0254


@dataclass
class DepthModelResult:
    """Depth grid plus the scalar summaries the payload and diagnostics need."""

    depth_grid: np.ndarray
    water_surface_datum: float | None
    model_method: str
    effective_rainfall_inches: float
    model_runoff_fraction: float
    urban_drainage_multiplier: float
    historical_event_count: int


def _run_scs_d8_model(
    sources: SourceData,
    rainfall_inches: float,
    water_mask: np.ndarray,
    historical_event_count: int,
) -> DepthModelResult:
    """Default model: SCS curve-number runoff scaled by the drainage multiplier, routed
    by D8 into ponding.
    """
    dem_values = sources.dem_values
    dem_bounds = sources.dem_bounds
    soil_groups = sources.soil_groups
    if soil_groups.shape != dem_values.shape:
        soil_groups = resample_continuous_grid(soil_groups, dem_values.shape)
    curve_numbers, _curve_number_summary = curve_number_grid(
        sources.landcover, soil_groups, water_mask
    )
    valid_dem = np.isfinite(dem_values)
    if sources.dem_nodata is not None:
        valid_dem &= dem_values != sources.dem_nodata
    curve_numbers[~valid_dem] = np.nan

    runoff_depth_grid = scs_runoff_depth_grid(rainfall_inches, curve_numbers)
    # Up to +15% runoff depth for areas with chronic historical flood news reports.
    urban_drainage_multiplier = min(
        MAX_DRAINAGE_MULTIPLIER,
        1.0 + (math.sqrt(historical_event_count) * EVENT_SCALAR)
    )
    runoff_depth_grid = runoff_depth_grid * urban_drainage_multiplier

    pixel_area_m2 = (
        (dem_bounds[2] - dem_bounds[0])
        * (dem_bounds[3] - dem_bounds[1])
        / dem_values.size
    )
    depth_grid, _basin_level_range, _routing_summary = solve_d8_catchment_ponding(
        dem_values, runoff_depth_grid, pixel_area_m2, sources.dem_nodata
    )
    land_cell_count = int(np.isfinite(runoff_depth_grid).sum())
    effective_rainfall_inches = (
        float(np.nansum(runoff_depth_grid) / land_cell_count * METERS_TO_INCHES)
        if land_cell_count
        else 0.0
    )
    model_runoff_fraction = (
        float(
            np.nansum(runoff_depth_grid)
            / (rainfall_inches * INCHES_TO_METERS * land_cell_count)
        )
        if rainfall_inches > 0 and land_cell_count
        else 0.0
    )
    return DepthModelResult(
        depth_grid=depth_grid,
        water_surface_datum=None,
        model_method=(
            "SCS curve-number rainfall excess from MRLC NLCD land cover and "
            "USDA SSURGO hydrologic soil groups, routed by D8 into per-sink "
            "catchment ponding over an expanded DEM context domain"
        ),
        effective_rainfall_inches=effective_rainfall_inches,
        model_runoff_fraction=model_runoff_fraction,
        urban_drainage_multiplier=urban_drainage_multiplier,
        historical_event_count=historical_event_count,
    )


def _run_retained_rainfall_model(
    sources: SourceData,
    rainfall_inches: float,
    retained_water_fraction: float,
    water_mask: np.ndarray,
    historical_event_count: int,
) -> DepthModelResult:
    """Legacy model: closed-box DEM filled with a fixed fraction of the rainfall."""
    water_surface_datum, depth_grid, _pixel_area_m2 = solve_retained_rainfall_surface(
        sources.dem_values,
        sources.dem_bounds,
        rainfall_inches,
        retained_water_fraction,
        sources.dem_nodata,
        water_mask,
    )
    return DepthModelResult(
        depth_grid=depth_grid,
        water_surface_datum=water_surface_datum,
        model_method=(
            "legacy explicit retained-rainfall-fraction closed-box DEM scenario; "
            "NLCD and SSURGO are diagnostics only"
        ),
        effective_rainfall_inches=rainfall_inches * retained_water_fraction,
        model_runoff_fraction=retained_water_fraction,
        urban_drainage_multiplier=1.0,
        historical_event_count=historical_event_count,
    )


def run_depth_model(
    sources: SourceData,
    rainfall_inches: float,
    retained_water_fraction: float | None,
    water_mask: np.ndarray,
) -> DepthModelResult:
    """Compute the flood-depth grid for the analysis domain with the selected model."""
    historical_event_count = sources.groundsource.get("events_intersecting_area", 0)
    if retained_water_fraction is None:
        return _run_scs_d8_model(
            sources, rainfall_inches, water_mask, historical_event_count
        )
    return _run_retained_rainfall_model(
        sources,
        rainfall_inches,
        retained_water_fraction,
        water_mask,
        historical_event_count,
    )