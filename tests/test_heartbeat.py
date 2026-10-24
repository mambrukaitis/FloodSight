from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from requests import Response
from shapely.geometry import LineString, Point, mapping
from shapely.wkb import dumps

from flood_heartbeat.common import checked_json
from flood_heartbeat.fema import FemaResult, FemaZone, classify_sfha, compare_events
from flood_heartbeat.geo import StudyArea
from flood_heartbeat.groundsource import (
    _date_text,
    _decode_geometry,
    process_parquet,
)
from reitz_terrain_screen import (
    PROJECT,
    _depth_grid_html,
    bathtub_depths,
    screen_low_points,
)
from reitz_2x2_demo import (
    load_fema_cells,
    load_groundsource_scan,
    make_cells,
    make_result,
    render_html,
    scan_groundsource_box,
    summarize_elevation,
)


class GeoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.area = StudyArea()

    def test_distance_and_circle_filter(self) -> None:
        self.assertAlmostEqual(
            self.area.distance_km(self.area.latitude, self.area.longitude), 0
        )
        self.assertLessEqual(
            self.area.distance_km(self.area.latitude + 0.1, self.area.longitude), 30
        )
        self.assertGreater(
            self.area.distance_km(self.area.latitude + 1, self.area.longitude), 30
        )
        min_lon, min_lat, max_lon, max_lat = self.area.bbox_geographic
        self.assertLess(min_lon, self.area.longitude)
        self.assertGreater(max_lon, self.area.longitude)
        self.assertLess(min_lat, self.area.latitude)
        self.assertGreater(max_lat, self.area.latitude)


class GroundsourceTests(unittest.TestCase):
    def test_decode_wkb_and_handle_malformed_wkb(self) -> None:
        point = Point(-82.32, 29.65)
        self.assertEqual(_decode_geometry(dumps(point)), point)
        with self.assertRaises(Exception):
            _decode_geometry(b"not-wkb")

    def test_date_handling(self) -> None:
        self.assertEqual(_date_text(date(2020, 1, 2)), "2020-01-02")
        self.assertEqual(
            _date_text(datetime(2020, 1, 2, tzinfo=timezone.utc)),
            "2020-01-02T00:00:00+00:00",
        )
        self.assertEqual(_date_text(None), "")

    def test_parquet_spatial_filter_and_empty_dataset(self) -> None:
        schema = pa.schema(
            [
                ("uuid", pa.string()),
                ("area_km2", pa.float64()),
                ("geometry", pa.binary()),
                ("start_date", pa.string()),
                ("end_date", pa.string()),
            ]
        )
        near = Point(-82.3248, 29.6516).buffer(0.01)
        far = Point(-81.0, 30.5).buffer(0.01)
        with tempfile.TemporaryDirectory() as directory:
            populated = Path(directory) / "events.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "uuid": "near",
                            "area_km2": 1.2,
                            "geometry": dumps(near),
                            "start_date": "2020-01-01",
                            "end_date": "2020-01-02",
                        },
                        {
                            "uuid": "far",
                            "area_km2": 1.0,
                            "geometry": dumps(far),
                            "start_date": "2021-01-01",
                            "end_date": "2021-01-02",
                        },
                    ],
                    schema=schema,
                ),
                populated,
            )
            result = process_parquet(populated, StudyArea())
            self.assertEqual(result.scanned, 2)
            self.assertEqual(result.retained, 1)
            self.assertEqual(result.events[0]["event_uuid"], "near")
            self.assertGreater(result.events[0]["intersection_area_km2"], 0)

            empty = Path(directory) / "empty.parquet"
            pq.write_table(pa.Table.from_pylist([], schema=schema), empty)
            empty_result = process_parquet(empty, StudyArea())
            self.assertEqual(empty_result.status, "success_empty")
            self.assertEqual(empty_result.scanned, 0)
            self.assertEqual(empty_result.retained, 0)


class FemaTests(unittest.TestCase):
    def test_sfha_classification(self) -> None:
        self.assertIs(classify_sfha("Y"), True)
        self.assertIs(classify_sfha("n"), False)
        self.assertIsNone(classify_sfha(None))
        self.assertIsNone(classify_sfha("not classified"))

    def test_event_zone_comparison_preserves_unknown(self) -> None:
        event_geometry = StudyArea().project(Point(-82.3248, 29.6516).buffer(0.01))
        zone = FemaZone(
            {"FLD_ZONE": "A", "SFHA_TF": None},
            event_geometry.buffer(10),
        )
        event = {
            "event_uuid": "test",
            "geometry_metric": event_geometry,
            "centroid_lat": 29.6516,
            "centroid_lon": -82.3248,
        }
        result = FemaResult(status="success", zones=[zone])
        comparison = compare_events([event], result)[0]
        self.assertTrue(comparison["intersects_fema_zone"])
        self.assertEqual(comparison["intersects_sfha"], "unknown")
        self.assertEqual(comparison["intersecting_fld_zones"], "A")


class ApiErrorTests(unittest.TestCase):
    def test_arcgis_style_api_error_is_not_a_success_response(self) -> None:
        response = Response()
        response.status_code = 200
        response.url = "https://example.test/query"
        response._content = b'{"error":{"code":400,"message":"bad query"}}'
        with self.assertRaisesRegex(RuntimeError, "bad query"):
            checked_json(response)


class TerrainScreenTests(unittest.TestCase):
    def test_low_points_are_screening_elevations_not_depths(self) -> None:
        values = np.full((100, 100), 10.0, dtype=np.float32)
        values[49, 50] = 2.0
        road = LineString([(50, 50), (51, 50)])
        points, stats, count, samples, terrain_cells = screen_low_points(
            values,
            (0.0, 0.0, 100.0, 100.0),
            None,
            road,
            road_width_m=5,
            count=1,
        )
        self.assertGreater(count, 0)
        self.assertEqual(points[0]["elevation_m"], 2.0)
        self.assertEqual(stats["minimum"], 2.0)
        self.assertEqual(stats["median"], 10.0)
        self.assertEqual(len(terrain_cells), 10_000)
        self.assertEqual(len(samples), count)

    def test_bathtub_model_conserves_rain_volume(self) -> None:
        samples = [
            {"elevation_m": 0.0},
            {"elevation_m": 0.0},
            {"elevation_m": 1.0},
        ]
        _, depths = bathtub_depths(samples, rainfall_inches=1)
        self.assertAlmostEqual(sum(depths), 3 * 0.0254, places=9)
        self.assertGreater(depths[0], 0)
        self.assertEqual(depths[2], 0)

    def test_depth_grid_labels_the_road_result_as_illustrative(self) -> None:
        page = _depth_grid_html(
            [
                {
                    "row": 0,
                    "column": 0,
                    "latitude": 29.0,
                    "longitude": -82.0,
                    "elevation_m": 1.0,
                    "on_mapped_road": True,
                }
            ],
            road_width_m=5,
        )
        self.assertIn("road-result", page)
        self.assertIn("This is not evidence the real road stays dry.", page)
        self.assertIn("Not a real-road flood estimate.", page)


class ReitzGridDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        center_x, center_y = PROJECT.transform(
            -82.34834, 29.64489
        )
        self.bounds = (
            center_x - 50,
            center_y - 50,
            center_x + 50,
            center_y + 50,
        )

    def test_grid_is_four_equal_non_overlapping_polygons(self) -> None:
        cells = make_cells(self.bounds)
        self.assertEqual([cell["id"] for cell in cells], ["NW", "NE", "SW", "SE"])
        self.assertTrue(all(cell["geometry"].is_valid for cell in cells))
        self.assertAlmostEqual(
            sum(cell["metric_geometry"].area for cell in cells),
            10_000,
        )
        for index, cell in enumerate(cells):
            for other in cells[index + 1 :]:
                self.assertAlmostEqual(
                    cell["metric_geometry"].intersection(
                        other["metric_geometry"]
                    ).area,
                    0,
                )

    def test_elevation_summary_uses_each_50_by_50_quadrant(self) -> None:
        values = np.empty((100, 100), dtype=np.float32)
        values[:50, :50] = 1
        values[:50, 50:] = 2
        values[50:, :50] = 3
        values[50:, 50:] = 4
        summary = summarize_elevation(values, no_data=None)
        self.assertEqual(summary["NW"]["minimum_elevation_m"], 1)
        self.assertEqual(summary["NE"]["median_elevation_m"], 2)
        self.assertEqual(summary["SW"]["valid_pixel_count"], 2500)
        self.assertEqual(summary["SE"]["minimum_elevation_m"], 4)

    def test_partial_fema_sample_only_marks_intersecting_cell(self) -> None:
        cells = make_cells(self.bounds)
        nw_geometry = mapping(cells[0]["geometry"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fema.geojson"
            path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": nw_geometry,
                                "properties": {"FLD_ZONE": "X"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            status, summary, error = load_fema_cells(cells, path)
        self.assertEqual(status, "partial_cached_sample")
        self.assertIsNone(error)
        self.assertEqual(summary[0]["flood_zones"], ["X"])
        self.assertEqual(summary[1]["count"], 0)

    def test_groundsource_scan_loads_historical_footprints_per_cell(self) -> None:
        cells = make_cells(self.bounds)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.json"
            path.write_text(
                json.dumps(
                    {
                        "records_scanned": 2_646_302,
                        "events": [
                            {
                                "uuid": "historic-event",
                                "start_date": "2024-01-01",
                                "end_date": "2024-01-02",
                                "area_km2": 1.0,
                                "geometry": mapping(cells[0]["geometry"]),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary, counts, features = load_groundsource_scan(path, cells)
        self.assertEqual(summary["records_scanned"], 2_646_302)
        self.assertEqual(summary["events_intersecting_area"], 1)
        self.assertEqual(counts, {"NW": 1, "NE": 0, "SW": 0, "SE": 0})
        self.assertEqual(len(features), 1)
        self.assertIsNone(features[0]["properties"]["depth_estimate_inches"])

    def test_groundsource_box_scan_checks_geometry_and_dates(self) -> None:
        schema = pa.schema(
            [
                ("uuid", pa.string()),
                ("area_km2", pa.float64()),
                ("geometry", pa.binary()),
                ("start_date", pa.string()),
                ("end_date", pa.string()),
            ],
            metadata={
                b"geo": json.dumps(
                    {
                        "primary_column": "geometry",
                        "columns": {
                            "geometry": {
                                "encoding": "WKB",
                                "crs": {"id": {"authority": "EPSG", "code": 4326}},
                            }
                        },
                    }
                ).encode("utf-8")
            },
        )
        cells = make_cells(self.bounds)
        with tempfile.TemporaryDirectory() as directory:
            parquet_path = Path(directory) / "events.parquet"
            report_path = Path(directory) / "scan.json"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "uuid": "near",
                            "area_km2": 1.0,
                            "geometry": dumps(cells[0]["geometry"]),
                            "start_date": "2020-01-01",
                            "end_date": "2020-01-02",
                        },
                        {
                            "uuid": "far",
                            "area_km2": 2.0,
                            "geometry": dumps(Point(-81, 30).buffer(0.01)),
                            "start_date": "2021-01-01",
                            "end_date": "2021-01-02",
                        },
                    ],
                    schema=schema,
                ),
                parquet_path,
            )
            report = scan_groundsource_box(parquet_path, cells, report_path)
            report_written = report_path.is_file()
        self.assertEqual(report["records_scanned"], 2)
        self.assertEqual(report["malformed_records"], 0)
        self.assertEqual(report["events_intersecting_area"], 1)
        self.assertEqual(report["events"][0]["uuid"], "near")
        self.assertTrue(report_written)

    def test_output_keeps_depth_unknown_when_groundsource_is_unscanned(self) -> None:
        elevation = {
            cell: {
                "valid_pixel_count": 2500,
                "minimum_elevation_m": 37.0,
                "median_elevation_m": 38.0,
            }
            for cell in ("NW", "NE", "SW", "SE")
        }
        fema = [{"count": 0, "flood_zones": []} for _ in range(4)]
        result = make_result(
            3,
            self.bounds,
            elevation,
            "partial_cached_sample",
            fema,
            None,
            [],
        )
        self.assertEqual(len(result["features"]), 4)
        self.assertIsNone(result["depth_estimate_inches"])
        for feature in result["features"]:
            self.assertIsNone(
                feature["properties"]["groundsource_historical_event_count"]
            )
            self.assertEqual(feature["properties"]["groundsource_status"], "not_scanned")
            self.assertIsNone(feature["properties"]["depth_estimate_inches"])
        page = render_html(result)
        self.assertIn("Groundsource local scan output not found", page)
        self.assertIn("Not a flood forecast", page)
        self.assertIn('class="map"', page)
        self.assertIn('fill="white"', page)
        self.assertEqual(page.count("<path "), 4)
        self.assertIn("NW", page)
        self.assertIn("SE", page)
        with self.assertRaises(ValueError):
            make_result(
                -1,
                self.bounds,
                elevation,
                "partial_cached_sample",
                fema,
                None,
                [],
            )


if __name__ == "__main__":
    unittest.main()
