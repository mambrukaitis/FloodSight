from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from requests import Session
from shapely.geometry import mapping, shape
from shapely.validation import make_valid

from .common import get_json, make_session
from .geo import StudyArea

FEMA_BASE = (
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
)
FEMA_FIELDS = ("FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "DFIRM_ID", "VERSION_ID")
COMPARISON_FIELDS = [
    "event_uuid",
    "start_date",
    "end_date",
    "area_km2",
    "centroid_lat",
    "centroid_lon",
    "distance_from_center_km",
    "intersects_fema_zone",
    "intersecting_fld_zones",
    "intersects_sfha",
    "centroid_fld_zones",
]


@dataclass
class FemaZone:
    attributes: dict[str, Any]
    geometry_metric: Any


@dataclass
class FemaResult:
    status: str = "unavailable"
    endpoint: str = FEMA_BASE
    layer_id: int | None = None
    layer_name: str = ""
    fetched: int = 0
    retained: int = 0
    fields: list[str] = field(default_factory=list)
    zones: list[FemaZone] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _esri_bounds(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    return {
        "xmin": bbox[0],
        "ymin": bbox[1],
        "xmax": bbox[2],
        "ymax": bbox[3],
        "spatialReference": {"wkid": 4326},
    }


def run_fema(study_area: StudyArea, session: Session | None = None) -> FemaResult:
    session = session or make_session()
    result = FemaResult()
    try:
        service = get_json(session, f"{FEMA_BASE}/layers", {"f": "json"})
        layer = next(
            (
                item
                for item in service.get("layers", [])
                if "flood hazard zones" in item.get("name", "").lower()
            ),
            None,
        )
        if layer is None:
            raise RuntimeError("NFHL service did not list a Flood Hazard Zones layer.")
        result.layer_id = int(layer["id"])
        result.layer_name = layer["name"]
        layer_url = f"{FEMA_BASE}/{result.layer_id}"
        metadata = get_json(session, layer_url, {"f": "json"})
        available_fields = {
            field["name"] for field in metadata.get("fields", []) if "name" in field
        }
        result.fields = [field for field in FEMA_FIELDS if field in available_fields]

        query_url = f"{layer_url}/query"
        base_params = {
            "where": "1=1",
            "geometry": json.dumps(_esri_bounds(study_area.bbox_geographic)),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
        }
        count_response = get_json(
            session,
            query_url,
            {**base_params, "f": "json", "returnCountOnly": "true"},
        )
        total = int(count_response.get("count", 0))
        page_size = max(1, min(int(metadata.get("maxRecordCount", 1000)), 100))
        offset = 0
        while offset < total:
            try:
                page = get_json(
                    session,
                    query_url,
                    {
                        **base_params,
                        "f": "geojson",
                        "outFields": ",".join(result.fields) if result.fields else "*",
                        "outSR": 4326,
                        "returnGeometry": "true",
                        "resultOffset": offset,
                        "resultRecordCount": page_size,
                    },
                )
            except Exception as exc:
                result.errors.append(
                    f"FEMA feature page at offset {offset} of {total} failed: {exc}"
                )
                break
            features = page.get("features", [])
            if not features:
                result.errors.append(
                    f"FEMA query returned no features at offset {offset}, although the count endpoint reported {total}."
                )
                break
            result.fetched += len(features)
            for feature in features:
                try:
                    geometry = shape(feature["geometry"])
                    if geometry.is_empty:
                        continue
                    if not geometry.is_valid:
                        geometry = make_valid(geometry)
                    projected = study_area.project(geometry)
                    if not projected.intersects(study_area.circle_metric):
                        continue
                    result.zones.append(
                        FemaZone(
                            attributes=feature.get("properties", {}),
                            geometry_metric=projected,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    result.errors.append(f"Skipped malformed FEMA feature: {exc}")

            offset += len(features)
            if len(features) < page_size and not page.get("exceededTransferLimit"):
                if offset < total:
                    result.errors.append(
                        f"FEMA pagination stopped at {offset} of {total} candidate features."
                    )
                break

        result.retained = len(result.zones)
        result.status = (
            "partial"
            if result.errors
            else "success"
            if result.retained
            else "success_empty"
        )
    except Exception as exc:
        result.status = "unavailable"
        result.errors.append(str(exc))
    return result


def classify_sfha(value: Any) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in {"Y", "YES", "TRUE", "T", "1"}:
        return True
    if normalized in {"N", "NO", "FALSE", "F", "0"}:
        return False
    return None


def compare_events(events: list[dict[str, Any]], fema: FemaResult) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        geometry = event["geometry_metric"]
        centroid = geometry.centroid
        overlaps = [
            zone
            for zone in fema.zones
            if geometry.intersects(zone.geometry_metric)
        ]
        centroid_zones = [
            zone
            for zone in fema.zones
            if zone.geometry_metric.covers(centroid)
        ]
        fld_zones = sorted(
            {
                str(zone.attributes.get("FLD_ZONE"))
                for zone in overlaps
                if zone.attributes.get("FLD_ZONE") not in (None, "")
            }
        )
        centroid_fld_zones = sorted(
            {
                str(zone.attributes.get("FLD_ZONE"))
                for zone in centroid_zones
                if zone.attributes.get("FLD_ZONE") not in (None, "")
            }
        )
        classifications = [
            classify_sfha(zone.attributes.get("SFHA_TF")) for zone in overlaps
        ]
        complete_fema_result = fema.status in {"success", "success_empty"}
        if any(value is True for value in classifications):
            intersects_sfha: bool | None = True
        elif overlaps and complete_fema_result and all(
            value is False for value in classifications
        ):
            intersects_sfha = False
        elif not overlaps and complete_fema_result:
            intersects_sfha = False
        else:
            intersects_sfha = None
        intersects_fema: bool | str = (
            True
            if overlaps
            else False
            if complete_fema_result
            else "unknown"
        )
        rows.append(
            {
                key: event.get(key, "")
                for key in (
                    "event_uuid",
                    "start_date",
                    "end_date",
                    "area_km2",
                    "centroid_lat",
                    "centroid_lon",
                    "distance_from_center_km",
                )
            }
            | {
                "intersects_fema_zone": intersects_fema,
                "intersecting_fld_zones": ";".join(fld_zones),
                "intersects_sfha": "unknown"
                if intersects_sfha is None
                else str(intersects_sfha).lower(),
                "centroid_fld_zones": ";".join(centroid_fld_zones),
            }
        )
    return rows


def write_fema_outputs(
    result: FemaResult,
    study_area: StudyArea,
    events: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    features = []
    for zone in result.zones:
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(study_area.unproject(zone.geometry_metric)),
                "properties": zone.attributes,
            }
        )
    (output_dir / "fema_flood_zones.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    rows = compare_events(events, result)
    with (output_dir / "event_fema_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
