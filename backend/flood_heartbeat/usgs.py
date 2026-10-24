from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from requests import Session

from .common import get_json, make_session
from .geo import StudyArea

USGS_BASE = "https://api.waterdata.usgs.gov/ogcapi/v1"
COLLECTIONS_URL = f"{USGS_BASE}/collections"
PARAMETERS = {
    "00065": "Gage height",
    "00060": "Discharge",
}
OBSERVATION_FIELDS = [
    "station_id",
    "station_number",
    "station_name",
    "parameter_code",
    "parameter_description",
    "value",
    "unit",
    "timestamp",
    "approval_status",
    "qualifier",
    "time_series_id",
    "source_endpoint",
]
STATION_FIELDS = [
    "station_id",
    "station_number",
    "name",
    "site_type_code",
    "site_type",
    "county",
    "latitude",
    "longitude",
    "distance_from_center_km",
    "metadata_json",
]


@dataclass
class UsgsResult:
    status: str = "unavailable"
    endpoint: str = f"{COLLECTIONS_URL}/monitoring-locations/items"
    fetched: int = 0
    retained: int = 0
    stations: list[dict[str, Any]] = field(default_factory=list)
    latest: list[dict[str, Any]] = field(default_factory=list)
    recent: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recent_window_start: str = ""
    recent_window_end: str = ""


def _fetch_pages(
    session: Session,
    url: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    data = get_json(session, url, params)
    features = list(data.get("features", []))
    next_link = next(
        (link.get("href") for link in data.get("links", []) if link.get("rel") == "next"),
        None,
    )
    visited = {url}
    while next_link:
        page_url = urljoin(url, next_link)
        if page_url in visited:
            raise RuntimeError(f"USGS API repeated a pagination link: {page_url}")
        visited.add(page_url)
        data = get_json(session, page_url)
        features.extend(data.get("features", []))
        next_link = next(
            (
                link.get("href")
                for link in data.get("links", [])
                if link.get("rel") == "next"
            ),
            None,
        )
    return features


def _observation(feature: dict[str, Any], source: str) -> dict[str, Any]:
    properties = feature.get("properties", {})
    code = str(properties.get("parameter_code") or "")
    return {
        "station_id": properties.get("monitoring_location_id", ""),
        "station_number": properties.get("monitoring_location_number", ""),
        "station_name": properties.get("monitoring_location_name", ""),
        "parameter_code": code,
        "parameter_description": PARAMETERS.get(code, ""),
        "value": properties.get("value", ""),
        "unit": properties.get("unit_of_measure", ""),
        "timestamp": properties.get("time", ""),
        "approval_status": properties.get("approval_status", ""),
        "qualifier": properties.get("qualifier", ""),
        "time_series_id": properties.get("time_series_id", ""),
        "source_endpoint": source,
    }


def run_usgs(study_area: StudyArea, session: Session | None = None) -> UsgsResult:
    session = session or make_session()
    result = UsgsResult()
    bbox = study_area.bbox_geographic
    try:
        location_queryables_url = (
            f"{COLLECTIONS_URL}/monitoring-locations/queryables?f=json"
        )
        location_queryables = get_json(session, location_queryables_url)
        if not {"geometry", "id"}.issubset(location_queryables.get("properties", {})):
            raise RuntimeError("USGS monitoring-location queryables are missing geometry/id.")

        locations_url = f"{COLLECTIONS_URL}/monitoring-locations/items"
        location_features = _fetch_pages(
            session,
            locations_url,
            {
                "f": "json",
                "limit": 1000,
                "bbox": ",".join(f"{value:.8f}" for value in bbox),
            },
        )
        result.fetched = len(location_features)
        for feature in location_features:
            coordinates = feature.get("geometry", {}).get("coordinates", [])
            if len(coordinates) < 2:
                continue
            longitude, latitude = coordinates[:2]
            distance = study_area.distance_km(latitude, longitude)
            if distance > study_area.radius_km:
                continue
            properties = feature.get("properties", {})
            result.stations.append(
                {
                    "station_id": properties.get("id", feature.get("id", "")),
                    "station_number": properties.get(
                        "monitoring_location_number", ""
                    ),
                    "name": properties.get("monitoring_location_name", ""),
                    "site_type_code": properties.get("site_type_code", ""),
                    "site_type": properties.get("site_type", ""),
                    "county": properties.get("county_name", ""),
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance_from_center_km": distance,
                    "metadata_json": properties,
                }
            )
        result.stations.sort(key=lambda station: station["distance_from_center_km"])
        result.retained = len(result.stations)

        latest_queryables = get_json(
            session,
            f"{COLLECTIONS_URL}/latest-continuous/queryables?f=json",
        )
        continuous_queryables = get_json(
            session,
            f"{COLLECTIONS_URL}/continuous/queryables?f=json",
        )
        for label, queryables in (
            ("latest-continuous", latest_queryables),
            ("continuous", continuous_queryables),
        ):
            required = {"monitoring_location_id", "parameter_code"}
            if not required.issubset(queryables.get("properties", {})):
                raise RuntimeError(
                    f"USGS {label} queryables do not support station and parameter filters."
                )

        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)
        result.recent_window_start = start.isoformat()
        result.recent_window_end = end.isoformat()
        surface_stations = [
            station
            for station in result.stations
            if station["site_type_code"] == "ST"
            or "stream" in str(station["site_type"]).lower()
            or "river" in str(station["site_type"]).lower()
        ][:5]

        latest_url = f"{COLLECTIONS_URL}/latest-continuous/items"
        station_ids = {station["station_id"] for station in result.stations}
        try:
            latest_features = _fetch_pages(
                session,
                latest_url,
                {
                    "f": "json",
                    "limit": 1000,
                    "bbox": ",".join(f"{value:.8f}" for value in bbox),
                },
            )
            result.fetched += len(latest_features)
            station_lookup = {
                station["station_id"]: station for station in result.stations
            }
            for feature in latest_features:
                properties = feature.get("properties", {})
                station_id = properties.get("monitoring_location_id")
                if station_id not in station_ids or str(
                    properties.get("parameter_code", "")
                ) not in PARAMETERS:
                    continue
                observation = _observation(feature, latest_url)
                station = station_lookup[station_id]
                observation["station_number"] = (
                    observation["station_number"] or station["station_number"]
                )
                observation["station_name"] = (
                    observation["station_name"] or station["name"]
                )
                result.latest.append(observation)
        except Exception as exc:
            result.errors.append(f"Nearby latest-continuous query: {exc}")

        station_parameters = {
            (row["station_id"], row["parameter_code"]) for row in result.latest
        }
        station_lookup = {station["station_id"]: station for station in surface_stations}
        for station_id, parameter_code in station_parameters:
            station = station_lookup.get(station_id)
            if station is None:
                continue
            continuous_url = f"{COLLECTIONS_URL}/continuous/items"
            continuous_params = {
                "f": "json",
                "limit": 1000,
                "monitoring_location_id": station_id,
                "parameter_code": parameter_code,
                "datetime": f"{start.isoformat()}/{end.isoformat()}",
            }
            try:
                features = _fetch_pages(session, continuous_url, continuous_params)
                result.recent.extend(
                    _observation(feature, continuous_url) for feature in features
                )
            except Exception as exc:
                result.errors.append(
                    f"Recent {parameter_code} for {station_id}: {exc}"
                )

        result.status = (
            "partial"
            if result.errors
            else "success"
            if result.stations or result.latest or result.recent
            else "success_empty"
        )
    except Exception as exc:
        result.errors.append(str(exc))
        result.status = "unavailable"
    return result


def write_usgs_outputs(result: UsgsResult, output_dir: Path) -> None:
    _write_csv(output_dir / "usgs_stations.csv", STATION_FIELDS, result.stations)
    _write_csv(
        output_dir / "usgs_latest_observations.csv",
        OBSERVATION_FIELDS,
        result.latest,
    )
    _write_csv(
        output_dir / "usgs_recent_observations.csv",
        OBSERVATION_FIELDS,
        result.recent,
    )


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        __import__("json").dumps(row.get(field), ensure_ascii=False)
                        if field == "metadata_json"
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )
