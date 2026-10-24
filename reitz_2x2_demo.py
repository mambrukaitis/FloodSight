from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
import requests
from shapely import transform
from shapely.errors import GEOSException
from shapely import from_wkb
from shapely.geometry import LineString, mapping, shape, box
from shapely.validation import make_valid

from reitz_terrain_screen import (
    OUTPUT_DIR,
    PROJECT,
    REITZ_UNION,
    UNPROJECT,
    get_elevations,
)

SIDE_METERS = 100.0
HALF_SIDE_METERS = SIDE_METERS / 2
GROUND_SOURCE_URL = "https://zenodo.org/records/18647054"
GROUND_SOURCE_FIELDS = {"uuid", "area_km2", "geometry", "start_date", "end_date"}


def make_cells(bounds: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    west, south, east, north = bounds
    mid_x = (west + east) / 2
    mid_y = (south + north) / 2
    definitions = [
        ("NW", west, mid_y, mid_x, north),
        ("NE", mid_x, mid_y, east, north),
        ("SW", west, south, mid_x, mid_y),
        ("SE", mid_x, south, east, mid_y),
    ]
    cells = []
    for name, x1, y1, x2, y2 in definitions:
        metric_geometry = box(x1, y1, x2, y2)
        geographic_geometry = transform(
            metric_geometry,
            UNPROJECT.transform,
            interleaved=False,
        )
        cells.append(
            {
                "id": name,
                "metric_geometry": metric_geometry,
                "geometry": geographic_geometry,
            }
        )
    return cells


def summarize_elevation(
    values: np.ndarray,
    no_data: float | None,
) -> dict[str, dict[str, float | int | None]]:
    if values.shape != (100, 100):
        raise ValueError(f"Expected a 100x100 elevation raster, got {values.shape}.")
    summaries: dict[str, dict[str, float | int | None]] = {}
    for cell, row_slice, column_slice in (
        ("NW", slice(0, 50), slice(0, 50)),
        ("NE", slice(0, 50), slice(50, 100)),
        ("SW", slice(50, 100), slice(0, 50)),
        ("SE", slice(50, 100), slice(50, 100)),
    ):
        subset = values[row_slice, column_slice].astype(float)
        valid = np.isfinite(subset)
        if no_data is not None:
            valid &= subset != no_data
        elevations = subset[valid]
        summaries[cell] = {
            "valid_pixel_count": int(elevations.size),
            "minimum_elevation_m": (
                float(np.min(elevations)) if elevations.size else None
            ),
            "median_elevation_m": (
                float(np.median(elevations)) if elevations.size else None
            ),
        }
    return summaries


def scan_groundsource_box(
    parquet_path: Path,
    cells: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    parquet_file = parquet.ParquetFile(parquet_path)
    fields = set(parquet_file.schema_arrow.names)
    missing = GROUND_SOURCE_FIELDS - fields
    if missing:
        raise ValueError(
            f"Groundsource Parquet is missing fields: {', '.join(sorted(missing))}"
        )
    metadata = parquet_file.schema_arrow.metadata or {}
    geo_metadata = metadata.get(b"geo")
    if not geo_metadata:
        raise ValueError("Groundsource Parquet has no GeoParquet CRS metadata.")
    geo = json.loads(geo_metadata)
    geometry_column = geo.get("columns", {}).get(geo.get("primary_column"), {})
    crs_text = json.dumps(geometry_column.get("crs") or {}).lower()
    if geometry_column.get("encoding") != "WKB" or not (
        "4326" in crs_text or "wgs 84" in crs_text
    ):
        raise ValueError("Groundsource geometry must be WKB in WGS84 coordinates.")

    west = min(cell["geometry"].bounds[0] for cell in cells)
    south = min(cell["geometry"].bounds[1] for cell in cells)
    east = max(cell["geometry"].bounds[2] for cell in cells)
    north = max(cell["geometry"].bounds[3] for cell in cells)
    query_geometry = box(west, south, east, north)
    report: dict[str, Any] = {
        "source": GROUND_SOURCE_URL,
        "parquet_path": str(parquet_path),
        "query_bounds_wgs84": [west, south, east, north],
        "records_scanned": 0,
        "malformed_records": 0,
        "events_intersecting_area": 0,
        "events": [],
    }
    for batch in parquet_file.iter_batches(
        batch_size=4096,
        columns=["uuid", "area_km2", "geometry", "start_date", "end_date"],
    ):
        for record in batch.to_pylist():
            report["records_scanned"] += 1
            try:
                raw_geometry = record["geometry"]
                if isinstance(raw_geometry, memoryview):
                    raw_geometry = raw_geometry.tobytes()
                geometry = from_wkb(raw_geometry)
                if geometry.is_empty:
                    raise ValueError("empty geometry")
                if not geometry.is_valid:
                    geometry = make_valid(geometry)
                clipped = geometry.intersection(query_geometry)
                if clipped.is_empty or clipped.area <= 0:
                    continue
                report["events"].append(
                    {
                        "uuid": str(record["uuid"]),
                        "start_date": record["start_date"],
                        "end_date": record["end_date"],
                        "area_km2": record["area_km2"],
                        "geometry": mapping(clipped),
                    }
                )
            except (GEOSException, TypeError, ValueError, KeyError):
                report["malformed_records"] += 1
    report["events_intersecting_area"] = len(report["events"])
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_groundsource_scan(
    scan_path: Path,
    cells: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int | None], list[dict[str, Any]]]:
    empty_counts = {cell["id"]: None for cell in cells}
    if not scan_path.is_file():
        return (
            {
                "status": "not_scanned",
                "records_scanned": 0,
                "events_intersecting_area": None,
                "note": "Groundsource local scan output not found.",
            },
            empty_counts,
            [],
        )
    try:
        data = json.loads(scan_path.read_text(encoding="utf-8"))
        events = data.get("events", [])
        per_cell_counts = {cell["id"]: 0 for cell in cells}
        features = []
        dates = []
        source_areas = []
        for event in events:
            geometry = shape(event["geometry"])
            for cell in cells:
                if geometry.intersection(cell["geometry"]).area > 0:
                    per_cell_counts[cell["id"]] += 1
            features.append(
                {
                    "type": "Feature",
                    "geometry": event["geometry"],
                    "properties": {
                        "event_uuid": event["uuid"],
                        "start_date": event["start_date"],
                        "end_date": event["end_date"],
                        "source_area_km2": event["area_km2"],
                        "source": "Groundsource historical flood event",
                        "depth_estimate_inches": None,
                    },
                }
            )
            dates.extend([event["start_date"], event["end_date"]])
            if event.get("area_km2") is not None:
                source_areas.append(float(event["area_km2"]))
        summary = {
            "status": "success" if events else "success_empty",
            "records_scanned": data.get("records_scanned", 0),
            "events_intersecting_area": len(events),
            "first_event_date": min(dates) if dates else None,
            "last_event_date": max(dates) if dates else None,
            "source_area_km2_min": min(source_areas) if source_areas else None,
            "source_area_km2_max": max(source_areas) if source_areas else None,
            "note": (
                "Historical event-footprint overlap only; the source has no "
                "water-depth field and the footprints are much larger than this site."
            ),
        }
        return summary, per_cell_counts, features
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        GEOSException,
    ) as exc:
        return (
            {
                "status": "unavailable",
                "records_scanned": None,
                "events_intersecting_area": None,
                "note": f"Could not read Groundsource scan output: {exc}",
            },
            empty_counts,
            [],
        )


def load_fema_cells(
    cells: list[dict[str, Any]],
    geojson_path: Path,
) -> tuple[str, list[dict[str, Any]], str | None]:
    if not geojson_path.is_file():
        return (
            "unavailable",
            [{"count": None, "flood_zones": []} for _ in cells],
            f"Cached FEMA GeoJSON not found: {geojson_path}",
        )
    try:
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
        features = data.get("features", [])
        summaries = []
        for cell in cells:
            overlaps = []
            for feature in features:
                if not feature.get("geometry"):
                    continue
                geometry = shape(feature["geometry"])
                if not geometry.is_valid:
                    geometry = make_valid(geometry)
                if geometry.intersection(cell["geometry"]).area > 0:
                    overlaps.append(feature.get("properties", {}))
            summaries.append(
                {
                    "count": len(overlaps),
                    "flood_zones": sorted(
                        {
                            str(item.get("FLD_ZONE"))
                            for item in overlaps
                            if item.get("FLD_ZONE") not in (None, "")
                        }
                    ),
                }
            )
        return "partial_cached_sample", summaries, None
    except (OSError, json.JSONDecodeError, TypeError, ValueError, GEOSException) as exc:
        return (
            "unavailable",
            [{"count": None, "flood_zones": []} for _ in cells],
            f"Could not read cached FEMA GeoJSON: {exc}",
        )


def load_nearby_usgs(
    station_path: Path,
    observation_path: Path,
    maximum_distance_km: float = 2.0,
) -> list[dict[str, Any]]:
    if not station_path.is_file():
        return []
    center_x, center_y = PROJECT.transform(
        REITZ_UNION["longitude"], REITZ_UNION["latitude"]
    )
    observations_by_station: dict[str, list[dict[str, str]]] = {}
    if observation_path.is_file():
        with observation_path.open(encoding="utf-8", newline="") as stream:
            for observation in csv.DictReader(stream):
                key = observation.get("station_id", "")
                observations_by_station.setdefault(key, []).append(observation)

    stations = []
    with station_path.open(encoding="utf-8", newline="") as stream:
        for station in csv.DictReader(stream):
            try:
                longitude = float(station["longitude"])
                latitude = float(station["latitude"])
                x, y = PROJECT.transform(longitude, latitude)
            except (KeyError, TypeError, ValueError):
                continue
            distance_km = float(np.hypot(x - center_x, y - center_y) / 1000)
            if distance_km > maximum_distance_km:
                continue
            station_id = station.get("station_id", "")
            stations.append(
                {
                    "station_id": station_id,
                    "name": station.get("name", ""),
                    "site_type": station.get("site_type", ""),
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance_km": distance_km,
                    "latest_observations": observations_by_station.get(
                        station_id, []
                    ),
                }
            )
    return sorted(stations, key=lambda item: item["distance_km"])


def make_result(
    rainfall_inches: float,
    bounds: tuple[float, float, float, float],
    elevation_summary: dict[str, dict[str, float | int | None]],
    fema_status: str,
    fema_summary: list[dict[str, Any]],
    fema_error: str | None,
    usgs_stations: list[dict[str, Any]],
    groundsource_summary: dict[str, Any] | None = None,
    groundsource_cell_counts: dict[str, int | None] | None = None,
    groundsource_features: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if rainfall_inches < 0:
        raise ValueError("Rainfall inches cannot be negative.")
    cells = make_cells(bounds)
    groundsource_summary = groundsource_summary or {
        "status": "not_scanned",
        "events_intersecting_area": None,
        "note": "Groundsource local scan output not found.",
    }
    groundsource_cell_counts = groundsource_cell_counts or {
        cell["id"]: None for cell in cells
    }
    features = []
    for cell, fema in zip(cells, fema_summary):
        terrain = elevation_summary[cell["id"]]
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(cell["geometry"]),
                "properties": {
                    "cell": cell["id"],
                    "rainfall_scenario_inches": rainfall_inches,
                    **terrain,
                    "fema_feature_count": fema["count"],
                    "fema_flood_zones": fema["flood_zones"],
                    "groundsource_historical_event_count": (
                        groundsource_cell_counts.get(cell["id"])
                    ),
                    "groundsource_status": groundsource_summary["status"],
                    "depth_estimate_inches": None,
                    "interpretation": "screening cell, not predicted inundation",
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "title": "Reitz Union 100 m campus screening grid",
        "center": REITZ_UNION,
        "rainfall_scenario_inches": rainfall_inches,
        "grid_size": "2x2; 50 m cells",
        "elevation_source": "USGS 3DEP; 1 m pixels",
        "fema_source_status": fema_status,
        "fema_note": (
            fema_error
            or "Uses the cached FEMA response, which is known to be a partial sample."
        ),
        "usgs_note": (
            "Nearby USGS readings are contextual stream-gage observations, "
            "not local street-water measurements."
        ),
        "groundsource": groundsource_summary,
        "depth_estimate_inches": None,
        "depth_note": (
            "Not calculated: rainfall total and these data do not establish "
            "drainage, runoff, storm duration, or water depth."
        ),
        "nearby_usgs_stations": usgs_stations,
        "features": features,
        "groundsource_event_footprints": {
            "type": "FeatureCollection",
            "features": groundsource_features or [],
        },
    }


def render_html(result: dict[str, Any]) -> str:
    map_size = 500
    padding = 24
    features = result["features"]
    all_bounds = [shape(feature["geometry"]).bounds for feature in features]
    west = min(bounds[0] for bounds in all_bounds)
    south = min(bounds[1] for bounds in all_bounds)
    east = max(bounds[2] for bounds in all_bounds)
    north = max(bounds[3] for bounds in all_bounds)

    def svg_point(longitude: float, latitude: float) -> tuple[float, float]:
        x = padding + (longitude - west) / (east - west) * (map_size - 2 * padding)
        y = padding + (north - latitude) / (north - south) * (map_size - 2 * padding)
        return x, y

    def svg_path(geometry: Any) -> str:
        polygons = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
        commands = []
        for polygon in polygons:
            points = [svg_point(x, y) for x, y in polygon.exterior.coords]
            commands.append(
                "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points) + " Z"
            )
        return " ".join(commands)

    polygon_markup = []
    for feature in features:
        geometry = shape(feature["geometry"])
        properties = feature["properties"]
        center_x, center_y = svg_point(*geometry.centroid.coords[0])
        cell_name = html.escape(properties["cell"])
        polygon_markup.append(
            f'<path d="{svg_path(geometry)}" fill="white" stroke="#172033" '
            'stroke-width="2" vector-effect="non-scaling-stroke"/>'
            f'<text x="{center_x:.2f}" y="{center_y - 7:.2f}" '
            'text-anchor="middle" class="cell-label">'
            f"{cell_name}</text>"
            f'<text x="{center_x:.2f}" y="{center_y + 14:.2f}" '
            'text-anchor="middle" class="cell-data">'
            f'min {properties["minimum_elevation_m"]:.1f} m'
            "</text>"
        )
    map_svg = (
        f'<svg class="map" viewBox="0 0 {map_size} {map_size}" '
        'role="img" aria-label="White map box with four outlined Reitz Union grid polygons">'
        f'<rect x="1" y="1" width="{map_size - 2}" height="{map_size - 2}" '
        'fill="white" stroke="#172033" stroke-width="2"/>'
        f'{"".join(polygon_markup)}</svg>'
    )
    cell_details = []
    for feature in features:
        properties = feature["properties"]
        zone_text = (
            ", ".join(properties["fema_flood_zones"])
            if properties["fema_flood_zones"]
            else "None in cached sample"
        )
        median = properties["median_elevation_m"]
        cell_details.append(
            f'<li><strong>{html.escape(properties["cell"])}</strong>: '
            f'min {properties["minimum_elevation_m"]:.1f} m, '
            f'median {median:.1f} m; FEMA zones: {html.escape(zone_text)}; '
            f'Groundsource historical events: '
            f'{properties["groundsource_historical_event_count"] if properties["groundsource_historical_event_count"] is not None else "not checked"}</li>'
        )
    groundsource = result["groundsource"]
    groundsource_count = groundsource.get("events_intersecting_area")
    groundsource_note = (
        f'{groundsource_count} historical Groundsource footprints overlap this '
        "100 m area; they do not contain water depth and are too broad to resolve "
        "local pooling."
        if groundsource_count is not None
        else groundsource["note"]
    )
    station_lines = []
    for station in result["nearby_usgs_stations"][:5]:
        observations = station["latest_observations"]
        latest = (
            "; ".join(
                f'{item.get("parameter_description") or item.get("parameter_code")}: '
                f'{item.get("value")} {item.get("unit")} '
                f'({item.get("timestamp") or "time unknown"})'
                for item in observations[:2]
            )
            if observations
            else "no cached reading for this station"
        )
        station_lines.append(
            f'<li>{html.escape(station["name"])} '
            f'({station["distance_km"]:.2f} km): {html.escape(latest)}</li>'
        )
    station_html = (
        "<ul>" + "".join(station_lines) + "</ul>"
        if station_lines
        else "<p>No cached USGS stations within 2 km.</p>"
    )
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Reitz Union 2x2 screening demo</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:700px;margin:2rem auto;padding:0 1rem;color:#172033}}
.map-box{{width:min(100%,400px);margin:1.5rem auto;border:1px solid #cbd5e1;background:white;padding:10px;box-sizing:border-box}}
.map{{display:block;width:100%;height:auto;background:white}}
.cell-label{{font:bold 22px system-ui,sans-serif;fill:#172033}}
.cell-data{{font:15px system-ui,sans-serif;fill:#334155}}
.note{{background:#f3f4f6;padding:1rem;border-radius:8px}}
</style>
<h1>Reitz Union: 2×2 data-screening demo</h1>
<p>Scenario rainfall: {result["rainfall_scenario_inches"]} inches; each square is about 50×50 m.</p>
<div class="map-box">{map_svg}</div>
<h2>Cell data</h2><ul>{''.join(cell_details)}</ul>
<p>Polygons show the four equal-area map cells. Elevation labels are terrain measurements, not water depth.</p>
<p>{html.escape(groundsource_note)}</p>
<h2>Nearby cached USGS stations</h2>{station_html}
<p class="note"><strong>Not a flood forecast.</strong> Elevation and FEMA zones are context;
Groundsource historical footprints are not depth measurements and FEMA data is a partial cached sample. No water depth is
calculated. The rainfall total lacks duration and drainage/runoff inputs. Gage records, if
listed, are historic observations with their own timestamps.</p>
<p>Center: {REITZ_UNION["latitude"]}, {REITZ_UNION["longitude"]}. See the accompanying
GeoJSON for cell boundaries and source fields.</p></html>"""


def run_demo(
    rainfall_inches: float,
    output_dir: Path,
    groundsource_path: Path,
) -> tuple[Path, Path]:
    if rainfall_inches < 0:
        raise ValueError("Rainfall inches cannot be negative.")
    output_dir.mkdir(parents=True, exist_ok=True)
    center_x, center_y = PROJECT.transform(
        REITZ_UNION["longitude"], REITZ_UNION["latitude"]
    )
    bounds = (
        center_x - HALF_SIDE_METERS,
        center_y - HALF_SIDE_METERS,
        center_x + HALF_SIDE_METERS,
        center_y + HALF_SIDE_METERS,
    )
    centerline = LineString(
        [(center_x - 0.5, center_y), (center_x + 0.5, center_y)]
    )
    values, raster_bounds, no_data = get_elevations(requests.Session(), centerline)
    if any(abs(a - b) > 0.1 for a, b in zip(bounds, raster_bounds)):
        raise RuntimeError("USGS 3DEP raster bounds do not match the 100 m demo grid.")

    cells = make_cells(bounds)
    groundsource_scan_path = output_dir / "reitz_groundsource_full_scan.json"
    if groundsource_path.is_file():
        print(
            "Scanning all Groundsource archive records for historical Reitz-area "
            "event footprints..."
        )
        scan_groundsource_box(
            groundsource_path,
            cells,
            groundsource_scan_path,
        )
        groundsource_summary, groundsource_cell_counts, groundsource_features = (
            load_groundsource_scan(groundsource_scan_path, cells)
        )
    else:
        groundsource_summary = {
            "status": "unavailable",
            "events_intersecting_area": None,
            "note": f"Groundsource Parquet file not found: {groundsource_path}",
        }
        groundsource_cell_counts = {cell["id"]: None for cell in cells}
        groundsource_features = []
    fema_status, fema_summary, fema_error = load_fema_cells(
        cells,
        output_dir / "fema_flood_zones.geojson",
    )
    usgs_stations = load_nearby_usgs(
        output_dir / "usgs_stations.csv",
        output_dir / "usgs_latest_observations.csv",
    )
    result = make_result(
        rainfall_inches,
        bounds,
        summarize_elevation(values, no_data),
        fema_status,
        fema_summary,
        fema_error,
        usgs_stations,
        groundsource_summary,
        groundsource_cell_counts,
        groundsource_features,
    )
    geojson_path = output_dir / "reitz_2x2_screening.geojson"
    html_path = output_dir / "reitz_2x2_screening.html"
    geojson_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    html_path.write_text(render_html(result), encoding="utf-8")
    return geojson_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small 2x2 source-screening map around Reitz Union."
    )
    parser.add_argument(
        "--rainfall-inches",
        type=float,
        help="Rainfall scenario total; if omitted, prompt interactively.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--groundsource-parquet",
        type=Path,
        default=Path(".cache/groundsource/groundsource_2026.parquet"),
        help="Cached Groundsource Parquet file; the complete file is scanned.",
    )
    args = parser.parse_args()
    rainfall = args.rainfall_inches
    if rainfall is None:
        rainfall = float(input("Rainfall scenario (inches): "))
    geojson_path, html_path = run_demo(
        rainfall,
        args.output_dir,
        args.groundsource_parquet,
    )
    print(f"Wrote GeoJSON: {geojson_path}")
    print(f"Wrote 2x2 map: {html_path}")
    if args.groundsource_parquet.is_file():
        print(f"Wrote Groundsource scan: {args.output_dir / 'reitz_groundsource_full_scan.json'}")
    print("Water depth is unavailable; this demo maps source evidence only.")


if __name__ == "__main__":
    main()
