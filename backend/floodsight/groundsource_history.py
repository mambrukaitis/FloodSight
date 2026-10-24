"""Groundsource historical flood-event lookup: builds (once) a spatial index of Florida
event footprints stored beside the 667 MB parquet file, then counts events
overlapping a buffered bbox.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pickle
import threading
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from shapely import STRtree
import shapely
from shapely.geometry import box


LOGGER = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("FLOODSIGHT_CACHE_DIR") or BACKEND_DIR / ".cache")
GROUND_SOURCE_PATH = CACHE_DIR / "groundsource" / "groundsource_2026.parquet"
GROUNDSOURCE_BUFFER_DEGREES = 0.015  # ~1.5 km of surrounding historical context
GROUNDSOURCE_FLORIDA_BOUNDS = (-88.0, 24.0, -79.5, 31.5)
_GROUNDSOURCE_INDEX: dict[tuple[str, int, int], tuple[Any, np.ndarray]] = {}
_GROUNDSOURCE_INDEX_LOCK = threading.Lock()


def groundsource_florida_events(parquet_path: Path) -> tuple[Any, np.ndarray]:
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


def groundsource_event_count(
    bbox_wgs84: tuple[float, float, float, float],
    parquet_path: Path,
) -> dict[str, Any]:
    """Count Groundsource event footprints overlapping the box (positive-area overlap)."""
    if not parquet_path.is_file():
        return {"events_intersecting_area": 0}
    tree, geometries = groundsource_florida_events(parquet_path)
    query = box(*bbox_wgs84)
    candidates = tree.query(query, predicate="intersects")
    count = sum(
        1
        for position in candidates
        if shapely.intersection(geometries[position], query).area > 0
    )
    return {"events_intersecting_area": count}


def buffered_bbox(
    bbox: tuple[float, float, float, float],
    buffer_degrees: float,
) -> tuple[float, float, float, float]:
    """Grow a bbox by a margin in degrees on every side."""
    return (
        bbox[0] - buffer_degrees,
        bbox[1] - buffer_degrees,
        bbox[2] + buffer_degrees,
        bbox[3] + buffer_degrees,
    )


def groundsource_job(
    bbox_wgs84: tuple[float, float, float, float],
    parquet_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the Groundsource lookup and time it; on failure logs a warning and reports
    zero events plus the error.
    """
    started = perf_counter()
    try:
        result = groundsource_event_count(bbox_wgs84, parquet_path)
        error = None
    except Exception as exc:
        LOGGER.warning("Groundsource lookup failed; using 0 events: %s", exc)
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
