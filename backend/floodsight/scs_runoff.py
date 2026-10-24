"""SCS (NRCS) curve-number runoff: maps NLCD land-cover classes and SSURGO hydrologic
soil groups to curve numbers, then converts rainfall into runoff depth per grid
cell.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from floodsight.grid_geometry import resample_continuous_grid


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


def curve_number_grid(
    landcover: np.ndarray,
    soil_groups: np.ndarray,
    water_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine land-cover classes and hydrologic soil groups into a per-cell SCS curve-
    number grid (mapped water cells are NaN); returns (grid, summary).
    """
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
        class_fractions[index] = resample_continuous_grid(
            source_fraction,
            soil_grid.shape,
        )
    fraction_sum = class_fractions.sum(axis=0)
    missing_fraction = fraction_sum <= 0
    fallback_index = fraction_classes.index(fallback_code)
    class_fractions[fallback_index, missing_fraction] = 1.0
    fraction_sum[missing_fraction] = 1.0
    class_fractions /= fraction_sum
    unknown_class_fraction = resample_continuous_grid(
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


def scs_runoff_depth_grid(
    rainfall_inches: float,
    curve_numbers: np.ndarray,
) -> np.ndarray:
    """Convert rainfall in inches into SCS runoff depth in metres for each curve-number
    cell.
    """
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
