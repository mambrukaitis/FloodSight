from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import __version__
from .fema import FemaResult
from .geo import StudyArea
from .groundsource import GroundsourceResult
from .usgs import UsgsResult


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _source_record(
    name: str,
    endpoint: str,
    status: str,
    acquisition: str,
    rows_fetched: int | None = None,
    rows_retained: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_name": name,
        "endpoint_or_dataset_url": endpoint,
        "status": status,
        "request_timestamp": datetime.now(timezone.utc).isoformat(),
        "rows_or_features_obtained": rows_fetched,
        "rows_or_features_retained_locally": rows_retained,
        "error": error,
        "acquisition": acquisition,
    }
    if extra:
        result.update(extra)
    return result


def source_status_document(
    groundsource: GroundsourceResult,
    usgs: UsgsResult,
    fema: FemaResult,
) -> dict[str, Any]:
    return {
        "groundsource": _source_record(
            "Groundsource historical flood events",
            groundsource.endpoint,
            groundsource.status,
            groundsource.acquisition,
            groundsource.scanned,
            groundsource.retained,
            groundsource.error,
            {
                "file_size_bytes": groundsource.file_size_bytes,
                "malformed_records_skipped": groundsource.malformed,
                "available_fields": groundsource.available_fields,
            },
        ),
        "usgs": _source_record(
            "USGS Water Data for the Nation v1 OGC API",
            usgs.endpoint,
            usgs.status,
            "live",
            usgs.fetched,
            usgs.retained,
            "; ".join(usgs.errors) or None,
            {
                "station_count": len(usgs.stations),
                "latest_observation_count": len(usgs.latest),
                "recent_observation_count": len(usgs.recent),
                "recent_window_start": usgs.recent_window_start,
                "recent_window_end": usgs.recent_window_end,
            },
        ),
        "fema": _source_record(
            "FEMA National Flood Hazard Layer",
            fema.endpoint,
            fema.status,
            "live",
            fema.fetched,
            fema.retained,
            "; ".join(fema.errors) or None,
            {
                "layer_id": fema.layer_id,
                "layer_name": fema.layer_name,
                "available_fields": fema.fields,
            },
        ),
    }


def write_run_metadata(
    output_dir: Path,
    study_area: StudyArea,
    cache_dir: Path,
    run_at: str,
    arguments: dict[str, Any],
) -> None:
    write_json(
        output_dir / "run_metadata.json",
        {
            "run_timestamp": run_at,
            "configuration": arguments,
            "study_area": {
                "center_latitude": study_area.latitude,
                "center_longitude": study_area.longitude,
                "radius_km": study_area.radius_km,
                "metric_crs": "EPSG:32617",
                "geographic_crs": "EPSG:4326",
                "candidate_bbox_wgs84": study_area.bbox_geographic,
            },
            "source_urls": {
                "groundsource_record": "https://zenodo.org/records/18647054",
                "groundsource_download": "https://zenodo.org/records/18647054/files/groundsource_2026.parquet?download=1",
                "usgs_api": "https://api.waterdata.usgs.gov/ogcapi/v1/",
                "usgs_documentation": "https://api.waterdata.usgs.gov/docs/ogcapi/",
                "fema_nfhl": "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer",
            },
            "software": {
                "python": __import__("platform").python_version(),
                "flood_heartbeat": __version__,
                "requests": _package_version("requests"),
                "pyarrow": _package_version("pyarrow"),
                "shapely": _package_version("shapely"),
                "pyproj": _package_version("pyproj"),
            },
            "cache_directory": str(cache_dir.resolve()),
        },
    )


def write_summary(
    output_dir: Path,
    study_area: StudyArea,
    run_at: str,
    groundsource: GroundsourceResult,
    usgs: UsgsResult,
    fema: FemaResult,
    comparisons: list[dict[str, Any]],
) -> None:
    event_count = groundsource.retained
    start_dates = sorted(
        event["start_date"] for event in groundsource.events if event["start_date"]
    )
    end_dates = sorted(
        event["end_date"] for event in groundsource.events if event["end_date"]
    )
    latest_by_station: dict[str, dict[str, Any]] = {}
    now = datetime.now(timezone.utc)
    stale: list[str] = []
    for observation in usgs.latest:
        station_id = str(observation.get("station_id", ""))
        previous = latest_by_station.get(station_id)
        if previous is None or observation.get("timestamp", "") > previous.get(
            "timestamp", ""
        ):
            latest_by_station[station_id] = observation
    for station_id, observation in latest_by_station.items():
        timestamp = observation.get("timestamp")
        try:
            observed_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)
            age_hours = (now - observed_at.astimezone(timezone.utc)).total_seconds() / 3600
            if age_hours > 24:
                stale.append(f"{station_id}: {age_hours / 24:.1f} days old")
        except (TypeError, ValueError):
            stale.append(f"{station_id}: timestamp could not be parsed ({timestamp})")

    station_types = Counter(
        station.get("site_type", "unknown") or "unknown"
        for station in usgs.stations
    )
    comparisons_with_overlap = [
        row for row in comparisons if row["intersects_fema_zone"] is True
    ]
    sfha_known = sum(row["intersects_sfha"] == "true" for row in comparisons)
    sfha_unknown = sum(
        row["intersects_fema_zone"] is True and row["intersects_sfha"] == "unknown"
        for row in comparisons
    )
    local_zone_types = Counter(
        str(zone.attributes.get("FLD_ZONE", "unknown"))
        for zone in fema.zones
        if zone.attributes.get("FLD_ZONE") not in (None, "")
    )

    lines = [
        "# Gainesville Flood Detective heartbeat",
        "",
        f"- Run time (UTC): {run_at}",
        f"- Study center: {study_area.latitude:.4f}, {study_area.longitude:.4f}",
        f"- Radius: {study_area.radius_km:g} km (circular area; metric CRS EPSG:32617)",
        "",
        "## Source status",
        "",
        "| Source | Status | Result |",
        "|---|---|---|",
        f"| Groundsource | {groundsource.status} | {event_count} local events |",
        f"| USGS WDFN v1 | {usgs.status} | {len(usgs.stations)} local stations; {len(usgs.latest)} latest readings; {len(usgs.recent)} recent readings |",
        f"| FEMA NFHL | {fema.status} | {fema.retained} local hazard-zone features |",
        "",
        "## Groundsource",
        "",
    ]
    if groundsource.file_size_bytes is not None:
        lines.append(
            f"- Dataset size: {groundsource.file_size_bytes:,} bytes ({groundsource.file_size_bytes / 1e6:.1f} MB); acquisition: {groundsource.acquisition}."
        )
    lines.extend(
        [
            f"- Records scanned: {groundsource.scanned:,}; malformed/skipped: {groundsource.malformed:,}; retained in study circle: {event_count:,}.",
            f"- Date coverage in local events: {start_dates[0] if start_dates else 'none'} through {end_dates[-1] if end_dates else 'none'}.",
            f"- Local event volume: {'zero' if event_count == 0 else 'few' if event_count < 10 else 'many'} usable records.",
            f"- Available dataset fields: {', '.join(groundsource.available_fields) if groundsource.available_fields else 'not inspected'}.",
            "- News article text, URLs, confidence scores, and citations are not inferred; only fields present in the Parquet schema are reported.",
        ]
    )
    if groundsource.error:
        lines.append(f"- Groundsource error: {groundsource.error}")
    lines.extend(["", "| Nearest event | Start | End | Area (km²) | Center distance (km) |", "|---|---|---|---:|---:|"])
    for event in sorted(
        groundsource.events, key=lambda item: item["distance_from_center_km"]
    )[:5]:
        lines.append(
            "| {event_uuid} | {start_date} | {end_date} | {area_km2} | {distance:.2f} |".format(
                **event,
                distance=event["distance_from_center_km"],
            )
        )
    if not groundsource.events:
        lines.append("| No local events available | | | | |")

    lines.extend(
        [
            "",
            "## USGS monitoring locations and observations",
            "",
            f"- Nearby locations: {len(usgs.stations)} ({'; '.join(f'{count} {name}' for name, count in sorted(station_types.items())) or 'no types returned'}).",
            f"- Nearby surface-water locations: {sum(station['site_type_code'] == 'ST' or 'stream' in str(station['site_type']).lower() or 'river' in str(station['site_type']).lower() for station in usgs.stations)}; history was queried for up to five closest locations with relevant latest readings. Stations with latest readings: {len({row['station_id'] for row in usgs.latest})}; stations with readings in the recent window: {len({row['station_id'] for row in usgs.recent})}; window: {usgs.recent_window_start or 'not queried'} to {usgs.recent_window_end or 'not queried'}.",
            "",
            "| Station | Parameter | Value | Unit | Observed at (UTC) | Approval |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for observation in sorted(
        usgs.latest, key=lambda item: item.get("timestamp", ""), reverse=True
    )[:10]:
        lines.append(
            "| {station_name} ({station_id}) | {parameter_description} ({parameter_code}) | {value} | {unit} | {timestamp} | {approval_status} |".format(
                **{key: observation.get(key, "") for key in (
                    "station_name", "station_id", "parameter_description",
                    "parameter_code", "value", "unit", "timestamp", "approval_status"
                )}
            )
        )
    if not usgs.latest:
        lines.append("| No gage-height or discharge readings returned | | | | | |")
    if stale:
        lines.append("")
        lines.append(f"- Readings older than 24 hours or with unparseable timestamps: {', '.join(stale)}.")
    if usgs.errors:
        lines.extend(["", "- USGS request issues: " + "; ".join(usgs.errors)])

    lines.extend(
        [
            "",
            "## FEMA zones and event comparison",
            "",
            f"- FEMA layer: {fema.layer_name or 'not found'} (ID {fema.layer_id if fema.layer_id is not None else 'unknown'}); local features retained: {fema.retained}.",
            f"- Local FLD_ZONE classifications: {', '.join(f'{zone}: {count}' for zone, count in sorted(local_zone_types.items())) or 'none available'}.",
            f"- Groundsource event geometries intersecting a FEMA zone: {len(comparisons_with_overlap)} of {len(comparisons)}; known SFHA overlaps: {sfha_known}; intersecting with unknown SFHA classification: {sfha_unknown}.",
            "- An event outside a mapped zone is only a candidate mismatch for manual review, not proof that a FEMA map is wrong.",
        ]
    )
    if fema.errors:
        lines.append(f"- FEMA request/geometry issues: {'; '.join(fema.errors)}.")

    lines.extend(
        [
            "",
            "## Feasibility assessment and limitations",
            "",
            _assessment(groundsource, usgs, fema),
            "",
            "This heartbeat checks data availability and spatial co-occurrence only. It does not predict floods, assess an emergency, or establish map accuracy.",
            "",
            "## Generated data files",
            "",
            *[
                f"- [{filename}]({filename})"
                for filename in (
                    "source_status.json",
                    "run_metadata.json",
                    "groundsource_local_events.csv",
                    "groundsource_local_events.geojson",
                    "usgs_stations.csv",
                    "usgs_latest_observations.csv",
                    "usgs_recent_observations.csv",
                    "fema_flood_zones.geojson",
                    "event_fema_comparison.csv",
                )
            ],
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _assessment(
    groundsource: GroundsourceResult,
    usgs: UsgsResult,
    fema: FemaResult,
) -> str:
    findings = []
    if groundsource.retained:
        findings.append(
            f"Groundsource yielded {groundsource.retained} local event(s), enabling a first-pass historical footprint comparison."
        )
    elif groundsource.status == "success_empty":
        findings.append("Groundsource processed successfully but found no events in the study circle.")
    else:
        findings.append(
            "Groundsource could not be evaluated locally; resolve its download or file issue before judging event coverage."
        )
    if usgs.stations and usgs.latest:
        findings.append(
            f"USGS returned {len(usgs.stations)} local stations and {len(usgs.latest)} latest gage-height/discharge reading(s); verify age and station context before interpretation."
        )
    elif usgs.stations:
        findings.append(
            f"USGS found {len(usgs.stations)} stations but no requested latest gage-height/discharge readings."
        )
    else:
        findings.append("USGS did not provide usable local station records.")
    if fema.retained:
        findings.append(
            f"FEMA returned {fema.retained} local hazard polygons for comparison."
        )
    else:
        findings.append(
            "FEMA returned no locally usable hazard polygons."
            if fema.status != "partial"
            else f"FEMA returned {fema.fetched} candidate features before a later request failed; local coverage is partial."
        )
    findings.append(
        "Next, inspect event dates, geometry quality, station freshness, and zone attributes manually; do not treat overlap as a causal or predictive result."
    )
    return " ".join(findings)
