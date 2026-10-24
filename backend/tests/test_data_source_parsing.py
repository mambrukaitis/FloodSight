"""Decoding of NLCD land cover, SSURGO soil fallbacks, NOAA VDatum conversion and the waterbody GeoJSON."""

from __future__ import annotations
from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from shapely.geometry import Polygon
from floodsight.nlcd_landcover_source import decode_nlcd_values
from floodsight.request_caching import SPATIAL_CACHE
from floodsight.request_caching import SPATIAL_CACHE_LOCK
from floodsight.scs_runoff import curve_number_grid
from floodsight.vertical_datum_source import navd88_to_lmsl_offset
from floodsight.waterbody_source import waterbody_feature_collection


class DataSourceParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        with SPATIAL_CACHE_LOCK:
            SPATIAL_CACHE.clear()

    def test_nlcd_palette_indices_decode_to_landcover_classes(self) -> None:
        values = np.array([[3, 4, 5, 6, 10, 20]], dtype=np.uint8)
        color_map = np.zeros((3, 256), dtype=np.uint16)
        palette = {
            3: (221, 201, 201),
            4: (216, 147, 130),
            5: (237, 0, 0),
            6: (170, 0, 0),
            10: (28, 99, 48),
            20: (186, 216, 234),
        }
        for index, rgb in palette.items():
            color_map[:, index] = np.asarray(rgb, dtype=np.uint16) * 257
        decoded = decode_nlcd_values(values, color_map)
        np.testing.assert_array_equal(decoded, [[21, 22, 23, 24, 42, 90]])

    def test_nlcd_decoder_rejects_unrecognized_palette_colors(self) -> None:
        values = np.array([[1]], dtype=np.uint8)
        color_map = np.zeros((3, 256), dtype=np.uint16)
        color_map[:, 1] = np.asarray((10, 20, 30), dtype=np.uint16) * 257
        with self.assertRaisesRegex(RuntimeError, "do not match known class colors"):
            decode_nlcd_values(values, color_map)

    def test_unknown_soils_use_average_group_c(self) -> None:
        landcover = np.array([[71]], dtype=np.uint8)
        soil = np.array([[2.0]])
        curve_numbers, _ = curve_number_grid(
            landcover,
            soil,
            np.array([[False]]),
        )
        self.assertEqual(curve_numbers[0, 0], 74)

    def test_vdatum_lmsl_conversion_and_unsupported_coverage(self) -> None:
        session = SimpleNamespace(
            get=lambda *args, **kwargs: SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "region": "CONTIGUOUS",
                    "t_z": "0.125",
                    "uncertainty": "0.04",
                },
            )
        )
        offset, metadata = navd88_to_lmsl_offset(
            session,
            29.64,
            -82.36,
            "North American Vertical Datum of 1988 (NAVD 88)",
        )
        self.assertEqual(offset, 0.125)
        self.assertEqual(metadata["status"], "converted")

        unsupported_session = SimpleNamespace(
            get=lambda *args, **kwargs: SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"region": "CONTIGUOUS", "t_z": "-999999"},
            )
        )
        offset, metadata = navd88_to_lmsl_offset(
            unsupported_session,
            29.64,
            -82.36,
            "NAVD88",
        )
        self.assertEqual(offset, 0.0)
        self.assertEqual(metadata["status"], "unsupported_location")

    def test_waterbody_geojson_is_separate_and_typed(self) -> None:
        coordinates = [
            (500 + 10 * np.cos(angle), 3200 + 10 * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, 40, endpoint=False)
        ]
        circle = Polygon(coordinates)
        bbox = (-82.35, 29.64, -82.34, 29.65)
        with patch(
            "floodsight.waterbody_source.fetch_waterbodies",
            return_value=(
                [{
                    "geometry": circle,
                    "name": "Test pond",
                    "source": "test",
                    "feature_id": "pond-1",
                }],
                [],
            ),
        ):
            collection = waterbody_feature_collection(object(), bbox)
        feature = collection["features"][0]
        self.assertEqual(feature["properties"]["category"], "waterbody")
        self.assertEqual(feature["properties"]["name"], "Test pond")
        self.assertLessEqual(len(feature["geometry"]["coordinates"][0]), 10)
        self.assertEqual(
            collection["metadata"]["category"],
            "mapped_surface_water_not_flood_depth",
        )
