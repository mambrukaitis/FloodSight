"""Vertical datum handling: detects the 3DEP elevation datum and asks NOAA VDatum for
the offset that converts NAVD88 elevations to local mean sea level (LMSL).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from typing import Any

from pyproj import Transformer
import requests

from floodsight.dem_elevation_source import THREEDEP_IDENTIFY_URL


NOAA_VDATUM_URL = "https://vdatum.noaa.gov/vdatumweb/api/convert"


def dem_vertical_datum(
    session: requests.Session,
    latitude: float,
    longitude: float,
    metric_crs: int,
) -> tuple[str | None, str | None]:
    """Ask the 3DEP identify service for the DEM vertical datum; returns (datum, None),
    or (None, explanatory note) when unknown or on failure.
    """
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    x, y = forward.transform(longitude, latitude)
    try:
        response = session.get(
            THREEDEP_IDENTIFY_URL,
            params={
                "geometry": json.dumps(
                    {"x": x, "y": y, "spatialReference": {"wkid": metric_crs}}
                ),
                "geometryType": "esriGeometryPoint",
                "sr": metric_crs,
                "returnGeometry": "false",
                "returnCatalogItems": "true",
                "f": "json",
            },
            timeout=(2, 2),
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return None, f"Could not inspect 3DEP vertical-datum metadata: {exc}"
    catalog = data.get("catalogItems", {}).get("features", [])
    datums = {
        str(feature.get("attributes", {}).get("VerticalDatum")).strip()
        for feature in catalog
        if feature.get("attributes", {}).get("VerticalDatum")
    }
    if len(datums) == 1:
        return next(iter(datums)), None
    if not datums:
        return None, "3DEP source catalog reports no vertical datum for the sampled tile."
    return None, f"3DEP source tiles report inconsistent vertical datums: {sorted(datums)}."


def navd88_to_lmsl_offset(
    session: requests.Session,
    latitude: float,
    longitude: float,
    source_datum: str | None,
) -> tuple[float, dict[str, Any]]:
    """Ask NOAA VDatum for the NAVD88 to local mean sea level offset at a point;
    returns (offset metres, conversion record).
    """
    if not source_datum or "NAVD88" not in source_datum.upper().replace(" ", ""):
        return 0.0, {
            "status": "unsupported_source_datum",
            "source_datum": source_datum or "unknown",
            "target_datum": "LMSL",
            "applied_offset_meters": 0.0,
        }
    try:
        response = session.get(
            NOAA_VDATUM_URL,
            params={
                "s_x": longitude,
                "s_y": latitude,
                "s_z": 0,
                "s_h_frame": "NAD83_2011",
                "s_coor": "geo",
                "s_v_frame": "NAVD88",
                "s_v_unit": "m",
                "s_v_elevation": "height",
                "t_v_frame": "LMSL",
                "t_v_unit": "m",
            },
            timeout=(10, 30),
        )
        response.raise_for_status()
        result = response.json()
        offset = float(result["t_z"])
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return 0.0, {
            "status": "conversion_service_error",
            "source_datum": source_datum,
            "target_datum": "LMSL",
            "applied_offset_meters": 0.0,
            "error": str(exc),
        }
    if not math.isfinite(offset) or abs(offset) >= 999_000:
        return 0.0, {
            "status": "unsupported_location",
            "source_datum": source_datum,
            "target_datum": "LMSL",
            "applied_offset_meters": 0.0,
            "response_region": result.get("region"),
            "error": "NOAA VDatum reports no valid transformation at this location.",
        }
    return offset, {
        "status": "converted",
        "source_datum": source_datum,
        "target_datum": "LMSL",
        "applied_offset_meters": offset,
        "uncertainty_meters": (
            float(result["uncertainty"])
            if result.get("uncertainty") not in (None, "")
            else None
        ),
        "source": NOAA_VDATUM_URL,
    }


def datum_and_offset(
    session: requests.Session,
    latitude: float,
    longitude: float,
    metric_crs: int,
) -> tuple[tuple[str | None, str | None], tuple[float, dict[str, Any]]]:
    """Resolve the DEM datum and the LMSL offset together, starting the VDatum call
    speculatively; assumes NAVD88 when the metadata call times out.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        # 3DEP is NAVD88 for Florida, so convert speculatively while the datum
        # metadata is inspected; the result is only used if the datum agrees.
        speculative = executor.submit(
            navd88_to_lmsl_offset, session, latitude, longitude, "NAVD88"
        )
        datum = dem_vertical_datum(session, latitude, longitude, metric_crs)
        offset = speculative.result()
    if datum[0] is None and str(datum[1]).startswith("Could not inspect"):
        # Metadata service timed out; USGS 3DEP elevations are NAVD88 across Florida.
        note = f"{datum[1]} NAVD88 was assumed for the 3DEP DEM."
        return ("North American Vertical Datum of 1988 (NAVD 88)", note), offset
    if datum[0] and "NAVD88" in datum[0].upper().replace(" ", ""):
        return datum, offset
    return datum, navd88_to_lmsl_offset(session, latitude, longitude, datum[0])
