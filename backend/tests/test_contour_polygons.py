"""Depth-threshold contour polygons: thresholds, clipping, layering and invalid input."""

from __future__ import annotations
from __future__ import annotations
import unittest
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry import box
from shapely.geometry import shape
from floodsight.contour_polygons import vectorize_depth_thresholds


class ContourPolygonTests(unittest.TestCase):
    def test_default_contours_include_shallow_flood_depths(self) -> None:
        contours = vectorize_depth_thresholds(
            np.array([[0.0, 0.15]]),
            (0, 0, 2, 1),
            32617,
            water_surface_meters=0.2,
        )
        self.assertEqual(
            [contour["depth_meters"] for contour in contours],
            [0.1, 0.2],
        )

    def test_contours_are_clipped_to_requested_map_extent(self) -> None:
        contours = vectorize_depth_thresholds(
            np.zeros((10, 10)),
            (0, 0, 10, 10),
            32617,
            water_surface_meters=0.2,
            clip_bounds_metric=(2, 2, 8, 8),
        )
        self.assertEqual(
            [contour["depth_meters"] for contour in contours],
            [0.1, 0.2],
        )

    def test_contours_can_be_clipped_to_polygon_geometry(self) -> None:
        clip_geometry = box(500_002, 3_200_002, 500_008, 3_200_008)
        inverse = Transformer.from_crs(
            "EPSG:32617",
            "EPSG:4326",
            always_xy=True,
        )
        forward = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32617",
            always_xy=True,
        )
        contours = vectorize_depth_thresholds(
            np.zeros((10, 10)),
            (500_000, 3_200_000, 500_010, 3_200_010),
            32617,
            water_surface_meters=0.2,
            clip_geometry_metric=clip_geometry,
        )
        self.assertEqual(
            [contour["depth_meters"] for contour in contours],
            [0.1, 0.2],
        )
        for contour in contours:
            for coordinate in contour["coordinates"]:
                point_x, point_y = forward.transform(
                    coordinate["lng"],
                    coordinate["lat"],
                )
                self.assertTrue(
                    clip_geometry.buffer(1e-6).covers(Point(point_x, point_y))
                )
        coordinates = contours[0]["coordinates"]
        self.assertLessEqual(len(coordinates), 15)

    def test_depth_thresholds_create_closed_layered_polygons(self) -> None:
        elevations = np.array(
            [
                [0.0, 0.2],
                [0.8, 1.2],
            ]
        )
        result = vectorize_depth_thresholds(
            elevations,
            (500_000, 3_200_000, 500_002, 3_200_002),
            32617,
            water_surface_meters=1.2,
            thresholds_meters=(0.3, 1.0),
        )
        self.assertEqual([item["depth_meters"] for item in result], [0.3, 1.0])
        for contour in result:
            coordinates = contour["coordinates"]
            self.assertEqual(coordinates[0], coordinates[-1])
            self.assertLessEqual(len(coordinates), 15)
            self.assertTrue(all(set(point) == {"lat", "lng"} for point in coordinates))
        less_deep = shape(
            {
                "type": "Polygon",
                "coordinates": [
                    [(point["lng"], point["lat"]) for point in result[0]["coordinates"]]
                ],
            }
        )
        deeper = shape(
            {
                "type": "Polygon",
                "coordinates": [
                    [(point["lng"], point["lat"]) for point in result[1]["coordinates"]]
                ],
            }
        )
        self.assertTrue(less_deep.buffer(1e-8).covers(deeper))

    def test_contour_rejects_nonfinite_surface(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            vectorize_depth_thresholds(
                np.ones((1, 1)),
                (0, 0, 1, 1),
                32617,
                water_surface_meters=float("nan"),
            )
