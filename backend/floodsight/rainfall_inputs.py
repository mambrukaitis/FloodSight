"""Validation and normalisation of the rainfall part of a flood request.

A request may describe rainfall as a total depth, as an intensity plus a
duration, or not at all (a default storm is used). This module turns any valid
combination into a single total in inches and rejects invalid combinations.
"""

from __future__ import annotations

import math

DEFAULT_RAINFALL_INCHES = 10.0


def resolve_rainfall_inches(
    rainfall_inches: float | None,
    rainfall_intensity_inches_per_hour: float | None,
    rainfall_duration_hours: float | None,
) -> float:
    """Return total rainfall in inches from whichever rainfall form was supplied.

    Raises ValueError if intensity and duration are not given together, if
    both a total and an intensity/duration pair are given, or if any value is
    non-finite or out of range.
    """
    has_intensity = rainfall_intensity_inches_per_hour is not None
    has_duration = rainfall_duration_hours is not None
    if has_intensity != has_duration:
        raise ValueError("Rainfall intensity and duration must be supplied together.")
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
        if not math.isfinite(rainfall_duration_hours) or rainfall_duration_hours <= 0:
            raise ValueError("Rainfall duration must be positive and finite.")
        rainfall_inches = rainfall_intensity_inches_per_hour * rainfall_duration_hours
    elif rainfall_inches is None:
        rainfall_inches = DEFAULT_RAINFALL_INCHES
    if not math.isfinite(rainfall_inches) or rainfall_inches < 0:
        raise ValueError("Rainfall must be finite and non-negative.")
    return rainfall_inches


def validate_retained_water_fraction(retained_water_fraction: float | None) -> None:
    """Reject a legacy retained-water fraction that is non-finite or outside 0..1."""
    if retained_water_fraction is not None and (
        not math.isfinite(retained_water_fraction)
        or not 0 <= retained_water_fraction <= 1
    ):
        raise ValueError("Retained-water fraction must be between 0 and 1.")
