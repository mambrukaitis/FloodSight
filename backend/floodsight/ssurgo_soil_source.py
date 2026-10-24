"""USDA SSURGO soil source: samples hydrologic soil groups (A-D) on a coarse grid
through Soil Data Access and interpolates them to the DEM grid, falling back to
group C.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import requests

from floodsight.grid_geometry import resample_continuous_grid


SSURGO_QUERY_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
SSURGO_SAMPLE_GRID_SIZE = 4
FALLBACK_HYDROLOGIC_SOIL_GROUP = "C"


def _parse_ssurgo_hydrologic_group(value: Any) -> int | None:
    """Map an SSURGO hydrologic soil group string (A, B, C, D or dual such as A/D) to
    an index 0-3, or None.
    """
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in {"A", "B", "C", "D"}:
        return ord(normalized) - ord("A")
    groups = [group for group in normalized.split("/") if group in {"A", "B", "C", "D"}]
    if groups:
        return max(ord(group) - ord("A") for group in groups)
    return None


def fetch_ssurgo_hydrologic_soil_grid(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample hydrologic soil groups on a coarse grid via Soil Data Access and
    interpolate to the DEM shape; returns (grid, source metadata).
    """
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

    soil_grid = resample_continuous_grid(sample_groups, grid_shape)
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
