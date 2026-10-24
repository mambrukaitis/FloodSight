"""In-process spatial cache: time-to-live expiry, independent copies of cached values, and cache keys."""

from __future__ import annotations
from __future__ import annotations
import unittest
from floodsight.request_caching import SPATIAL_CACHE
from floodsight.request_caching import SPATIAL_CACHE_LOCK
from floodsight.request_caching import spatial_cache_key
from floodsight.request_caching import cached_spatial_data


class RequestCachingTests(unittest.TestCase):
    def setUp(self) -> None:
        with SPATIAL_CACHE_LOCK:
            SPATIAL_CACHE.clear()

    def test_spatial_data_cache_uses_ttl_and_returns_independent_values(self) -> None:
        calls = 0

        def load() -> dict[str, list[int]]:
            nonlocal calls
            calls += 1
            return {"values": [1, 2]}

        first, first_hit, _ = cached_spatial_data(("test",), 60, load)
        first["values"].append(3)
        second, second_hit, _ = cached_spatial_data(("test",), 60, load)

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(calls, 1)
        self.assertEqual(second, {"values": [1, 2]})

    def test_cache_key_supports_station_id_sequences(self) -> None:
        self.assertEqual(
            spatial_cache_key("daily", ("USGS-0123", "USGS-0456")),
            ("daily", ("USGS-0123", "USGS-0456")),
        )
