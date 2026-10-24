from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from flood_heartbeat.fema import (
    FemaResult,
    run_fema,
    write_fema_outputs,
)
from flood_heartbeat.geo import StudyArea
from flood_heartbeat.groundsource import (
    GroundsourceResult,
    run_groundsource,
    write_groundsource_outputs,
)
from flood_heartbeat.reporting import (
    source_status_document,
    write_json,
    write_run_metadata,
    write_summary,
)
from flood_heartbeat.usgs import UsgsResult, run_usgs, write_usgs_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether public flood and water datasets support a Gainesville study."
    )
    parser.add_argument("--latitude", type=float, default=29.6516)
    parser.add_argument("--longitude", type=float, default=-82.3248)
    parser.add_argument("--radius-km", type=float, default=30.0)
    parser.add_argument(
        "--groundsource-parquet",
        type=Path,
        help="Use a supplied Groundsource Parquet file instead of downloading it.",
    )
    parser.add_argument(
        "--skip-groundsource-download",
        action="store_true",
        help="Only process an existing cache; do not start a Groundsource download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gainesville_heartbeat"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/groundsource"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        study_area = StudyArea(args.latitude, args.longitude, args.radius_km)
    except ValueError as exc:
        print(f"Invalid study area: {exc}")
        return 2

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc).isoformat()

    source_action = (
        "using supplied Parquet file"
        if args.groundsource_parquet
        else "using cache only"
        if args.skip_groundsource_download
        else "this may download approximately 667 MB"
    )
    print(f"Checking Groundsource ({source_action})...")
    groundsource: GroundsourceResult = run_groundsource(
        study_area,
        cache_dir,
        args.groundsource_parquet.resolve() if args.groundsource_parquet else None,
        args.skip_groundsource_download,
    )
    write_groundsource_outputs(groundsource, output_dir)
    print(
        f"Groundsource: {groundsource.status}; {groundsource.retained} local events"
        + (f"; {groundsource.error}" if groundsource.error else "")
    )

    print("Checking USGS Water Data for the Nation v1...")
    usgs: UsgsResult = run_usgs(study_area)
    write_usgs_outputs(usgs, output_dir)
    print(
        f"USGS: {usgs.status}; {len(usgs.stations)} local stations; "
        f"{len(usgs.latest)} latest readings; {len(usgs.recent)} recent readings"
    )

    print("Checking FEMA National Flood Hazard Layer...")
    fema: FemaResult = run_fema(study_area)
    comparisons = write_fema_outputs(
        fema, study_area, groundsource.events, output_dir
    )
    print(f"FEMA: {fema.status}; {fema.retained} local hazard-zone features")

    status = source_status_document(groundsource, usgs, fema)
    write_json(output_dir / "source_status.json", status)
    write_run_metadata(
        output_dir,
        study_area,
        cache_dir,
        run_at,
        {
            "latitude": args.latitude,
            "longitude": args.longitude,
            "radius_km": args.radius_km,
            "groundsource_parquet": (
                str(args.groundsource_parquet.resolve())
                if args.groundsource_parquet
                else None
            ),
            "skip_groundsource_download": args.skip_groundsource_download,
            "output_dir": str(output_dir),
        },
    )
    write_summary(
        output_dir,
        study_area,
        run_at,
        groundsource,
        usgs,
        fema,
        comparisons,
    )
    print(f"Report: {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
