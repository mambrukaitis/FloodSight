from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
import requests
from shapely import from_wkb
from shapely.geometry import mapping
from shapely.errors import GEOSException

from .common import REQUEST_TIMEOUT, checked_json, make_session
from .geo import StudyArea, is_geographic

ZENODO_RECORD_URL = "https://zenodo.org/api/records/18647054"
ZENODO_FILE_URL = (
    "https://zenodo.org/records/18647054/files/groundsource_2026.parquet?download=1"
)
ZENODO_FILE_KEY = "groundsource_2026.parquet"
EXPECTED_MD5 = "cd1b5de6508f7aad8e1d1d0dd4cecea6"
REQUIRED_COLUMNS = {"uuid", "area_km2", "geometry", "start_date", "end_date"}
EVENT_CSV_FIELDS = [
    "event_uuid",
    "start_date",
    "end_date",
    "area_km2",
    "centroid_lat",
    "centroid_lon",
    "distance_from_center_km",
    "intersection_area_km2",
]


@dataclass
class GroundsourceResult:
    status: str = "unavailable"
    endpoint: str = ZENODO_RECORD_URL
    acquisition: str = "not_started"
    file_size_bytes: int | None = None
    scanned: int = 0
    malformed: int = 0
    retained: int = 0
    available_fields: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class CoordinateSystemError(RuntimeError):
    pass


def _download_dataset(cache_dir: Path) -> tuple[Path, str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / ZENODO_FILE_KEY
    session = make_session()

    if target.exists():
        digest = _md5_file(target)
        if digest == EXPECTED_MD5:
            return target, "cache", target.stat().st_size
        target.unlink()

    metadata = checked_json(
        session.get(f"{ZENODO_RECORD_URL}/files", timeout=REQUEST_TIMEOUT)
    )
    entry = next(
        (item for item in metadata.get("entries", []) if item.get("key") == ZENODO_FILE_KEY),
        None,
    )
    if entry is None:
        raise RuntimeError(f"Zenodo record does not list {ZENODO_FILE_KEY}.")
    expected_size = entry.get("size")
    expected_checksum = entry.get("checksum", "").removeprefix("md5:")
    if expected_checksum and expected_checksum != EXPECTED_MD5:
        raise RuntimeError(
            f"Zenodo MD5 changed: expected {EXPECTED_MD5}, received {expected_checksum}."
        )

    fd, temp_name = tempfile.mkstemp(
        prefix=f"{ZENODO_FILE_KEY}.", suffix=".part", dir=cache_dir
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with session.get(
            ZENODO_FILE_URL,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            response.raise_for_status()
            downloaded = 0
            with temp_path.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (20 * 1024 * 1024) < len(chunk):
                        total = expected_size or 0
                        if total:
                            print(
                                f"\rGroundsource download: {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\rGroundsource download: {downloaded / 1e6:.0f} MB",
                                end="",
                                flush=True,
                            )
        if downloaded == 0 or (expected_size and downloaded != expected_size):
            raise RuntimeError(
                f"Incomplete Groundsource download: received {downloaded} bytes; expected {expected_size}."
            )
        digest = _md5_file(temp_path)
        if digest != EXPECTED_MD5:
            raise RuntimeError(
                f"Groundsource checksum mismatch: expected {EXPECTED_MD5}, received {digest}."
            )
        temp_path.replace(target)
        print()
        return target, "downloaded", downloaded
    except Exception:
        temp_path.unlink(missing_ok=True)
        print()
        raise


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_geometry(value: Any):
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, str):
        value = bytes.fromhex(value)
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("geometry is not WKB bytes or hexadecimal text")
    geometry = from_wkb(value)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("geometry is empty or invalid")
    return geometry


def _date_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def process_parquet(
    parquet_path: Path,
    study_area: StudyArea,
    batch_size: int = 2048,
) -> GroundsourceResult:
    result = GroundsourceResult(
        status="success",
        endpoint=str(parquet_path),
        acquisition="local_file",
    )
    parquet_file = parquet.ParquetFile(parquet_path)
    schema_names = set(parquet_file.schema_arrow.names)
    result.available_fields = sorted(schema_names)
    missing = REQUIRED_COLUMNS - schema_names
    if missing:
        raise ValueError(
            f"Groundsource Parquet is missing required columns: {', '.join(sorted(missing))}"
        )

    sample_checked = 0
    transformer = study_area.to_metric
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["uuid", "area_km2", "geometry", "start_date", "end_date"],
    ):
        for row in batch.to_pylist():
            result.scanned += 1
            if result.scanned % 100_000 == 0:
                print(
                    f"\rGroundsource records scanned: {result.scanned:,} / "
                    f"{parquet_file.metadata.num_rows:,}",
                    end="",
                    flush=True,
                )
            try:
                geometry = _decode_geometry(row["geometry"])
                if sample_checked < 10:
                    if not is_geographic(geometry):
                        raise CoordinateSystemError(
                            "sample geometry coordinates are outside WGS84 longitude/latitude bounds"
                        )
                    sample_checked += 1
                if not is_geographic(geometry):
                    raise CoordinateSystemError(
                        "geometry coordinates are outside WGS84 longitude/latitude bounds"
                    )
                projected = study_area.project(geometry)
                if not projected.intersects(study_area.circle_metric):
                    continue
                clipped = projected.intersection(study_area.circle_metric)
                if clipped.is_empty:
                    continue

                centroid = projected.centroid
                lon, lat = study_area.to_geographic.transform(
                    centroid.x, centroid.y
                )
                start_date = _date_text(row["start_date"])
                end_date = _date_text(row["end_date"])
                result.dates.extend(value for value in (start_date, end_date) if value)
                result.events.append(
                    {
                        "event_uuid": str(row["uuid"]),
                        "start_date": start_date,
                        "end_date": end_date,
                        "area_km2": row["area_km2"],
                        "centroid_lat": lat,
                        "centroid_lon": lon,
                        "distance_from_center_km": study_area.distance_km(lat, lon),
                        "intersection_area_km2": clipped.area / 1_000_000,
                        "geometry_metric": projected,
                        "geometry_clipped_geojson": mapping(
                            study_area.unproject(clipped)
                        ),
                    }
                )
            except (ValueError, TypeError, OverflowError, GEOSException) as exc:
                result.malformed += 1
                if result.malformed <= 5:
                    print(f"Skipping malformed Groundsource record: {exc}")

    result.retained = len(result.events)
    if result.scanned:
        print()
    if result.retained == 0:
        result.status = "success_empty"
    return result


def run_groundsource(
    study_area: StudyArea,
    cache_dir: Path,
    local_path: Path | None = None,
    skip_download: bool = False,
) -> GroundsourceResult:
    try:
        if local_path is not None:
            if not local_path.is_file():
                raise FileNotFoundError(f"Local Parquet file not found: {local_path}")
            parquet_path = local_path
            acquisition = "local_file"
        elif skip_download:
            parquet_path = cache_dir / ZENODO_FILE_KEY
            if not parquet_path.is_file():
                raise FileNotFoundError(
                    f"Groundsource download skipped and cache is missing: {parquet_path}"
                )
            acquisition = "cache"
        else:
            parquet_path, acquisition, _ = _download_dataset(cache_dir)

        result = process_parquet(parquet_path, study_area)
        result.endpoint = str(parquet_path)
        result.acquisition = acquisition
        result.file_size_bytes = parquet_path.stat().st_size
        if acquisition != "local_file":
            result.endpoint = ZENODO_FILE_URL
        return result
    except Exception as exc:
        return GroundsourceResult(
            status="unavailable",
            acquisition="failed",
            error=str(exc),
            endpoint=str(local_path or ZENODO_RECORD_URL),
        )


def write_groundsource_outputs(result: GroundsourceResult, output_dir: Path) -> None:
    csv_path = output_dir / "groundsource_local_events.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENT_CSV_FIELDS)
        writer.writeheader()
        for event in result.events:
            writer.writerow({key: event.get(key, "") for key in EVENT_CSV_FIELDS})

    features = [
        {
            "type": "Feature",
            "geometry": event["geometry_clipped_geojson"],
            "properties": {
                key: event.get(key)
                for key in EVENT_CSV_FIELDS
                if key != "geometry_clipped_geojson"
            },
        }
        for event in result.events
    ]
    (output_dir / "groundsource_local_events.geojson").write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
