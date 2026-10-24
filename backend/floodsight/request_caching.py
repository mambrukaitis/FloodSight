"""In-process time-to-live cache for static spatial downloads, plus helpers that run a
source fetch with a managed HTTP session, time it and report whether it was a cache
hit.
"""

from __future__ import annotations

import copy
import threading
from time import perf_counter
from typing import Any

import requests


SPATIAL_CACHE: dict[tuple[Any, ...], tuple[float, Any]] = {}
SPATIAL_CACHE_LOCK = threading.Lock()


def spatial_cache_key(source: str, *values: Any) -> tuple[Any, ...]:
    """Build a hashable cache key from a source name and its (possibly nested or array-
    like) parameters.
    """
    def normalize(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return tuple(normalize(item) for item in value)
        if isinstance(value, float):
            return round(value, 7)
        return value

    normalized = tuple(normalize(value) for value in values)
    return (source, *normalized)


def cached_spatial_data(
    key: tuple[Any, ...],
    ttl_seconds: float,
    loader: Any,
) -> tuple[Any, bool, float]:
    """Return a copy of the cached value for key (fetching and storing it on a miss or
    after expiry); returns (value, was_hit, age).
    """
    started = perf_counter()
    now = started
    with SPATIAL_CACHE_LOCK:
        cached = SPATIAL_CACHE.get(key)
        if cached is not None and cached[0] > now:
            return copy.deepcopy(cached[1]), True, perf_counter() - started
        if cached is not None:
            del SPATIAL_CACHE[key]
    value = loader()
    with SPATIAL_CACHE_LOCK:
        SPATIAL_CACHE[key] = (perf_counter() + ttl_seconds, copy.deepcopy(value))
    return value, False, perf_counter() - started


def _using_session(session: requests.Session | None, operation: Any) -> Any:
    """Run operation(session), creating and closing a private requests.Session when
    none is supplied.
    """
    if session is not None:
        return operation(session)
    with requests.Session() as local_session:
        return operation(local_session)


def cached_source_job(
    source_name: str,
    key: tuple[Any, ...],
    ttl_seconds: float,
    session: requests.Session | None,
    operation: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fetch one named source through the TTL cache and return (value, timing record
    with elapsed time and hit/miss).
    """
    value, cache_hit, elapsed = cached_spatial_data(
        key,
        ttl_seconds,
        lambda: _using_session(session, operation),
    )
    return value, {
        "elapsed_seconds": elapsed,
        "cache": "hit" if cache_hit else "miss",
        "source": source_name,
    }
