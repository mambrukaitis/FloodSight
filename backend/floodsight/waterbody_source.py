"""Mapped surface-water polygons (lakes, ponds, rivers) from OpenStreetMap and the USGS
NHD, merged into projected polygons for the model and into GeoJSON for the
/waterbodies endpoint.
"""

from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET

from pyproj import Transformer
import requests
from shapely import transform
from shapely.geometry import box, LineString, Polygon, shape
from shapely.ops import polygonize, unary_union

from floodsight.grid_geometry import (
    local_utm_epsg,
    polygon_ring_coordinates,
    validate_request,
)


NHD_MAPSERVER_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
OSM_API_URL = "https://api.openstreetmap.org/api/0.6"


def _is_osm_waterbody(tags: dict[str, str]) -> bool:
    """True when OpenStreetMap tags describe a water feature (lake, pond, reservoir,
    river area, ...).
    """
    return (
        tags.get("natural") == "water"
        or "water" in tags
        or tags.get("landuse") == "reservoir"
    )


def _osm_relation_polygon(
    session: requests.Session,
    relation_id: str,
) -> Any:
    """Assemble the polygon of an OpenStreetMap multipolygon relation from its member
    ways.
    """
    response = session.get(
        f"{OSM_API_URL}/relation/{relation_id}/full.json",
        timeout=(5, 20),
    )
    response.raise_for_status()
    elements = response.json().get("elements", [])
    nodes = {
        element["id"]: (element["lon"], element["lat"])
        for element in elements
        if element.get("type") == "node"
    }
    ways = {
        element["id"]: element
        for element in elements
        if element.get("type") == "way"
    }
    relation = next(
        (
            element
            for element in elements
            if element.get("type") == "relation"
            and str(element.get("id")) == relation_id
        ),
        None,
    )
    if relation is None:
        raise RuntimeError(f"OSM relation {relation_id} was not returned.")

    member_lines: dict[str, list[LineString]] = {"outer": [], "inner": []}
    for member in relation.get("members", []):
        way = ways.get(member.get("ref"))
        role = member.get("role")
        if way is None or role not in member_lines:
            continue
        coordinates = [
            nodes[node_id]
            for node_id in way.get("nodes", [])
            if node_id in nodes
        ]
        if len(coordinates) >= 4 and coordinates[0] == coordinates[-1]:
            member_lines[role].append(LineString(coordinates))
    if not member_lines["outer"]:
        raise RuntimeError(f"OSM waterbody relation {relation_id} has no outer ring.")
    outer = unary_union(list(polygonize(unary_union(member_lines["outer"]))))
    inner = unary_union(list(polygonize(unary_union(member_lines["inner"]))))
    return outer.difference(inner) if not inner.is_empty else outer


def _fetch_osm_waterbodies(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch OpenStreetMap waterbody polygons for the bbox; returns (waterbody records,
    notes).
    """
    west, south, east, north = bbox_wgs84
    response = session.get(
        f"{OSM_API_URL}/map",
        params={"bbox": f"{west},{south},{east},{north}"},
        timeout=(5, 25),
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    nodes = {
        node.get("id"): (float(node.get("lon", "nan")), float(node.get("lat", "nan")))
        for node in root.findall("node")
    }
    features: list[dict[str, Any]] = []
    errors: list[str] = []
    relation_ids: set[str] = set()
    for relation in root.findall("relation"):
        tags = {tag.get("k", ""): tag.get("v", "") for tag in relation.findall("tag")}
        if _is_osm_waterbody(tags):
            relation_ids.add(relation.get("id", ""))
    for relation_id in sorted(relation_ids):
        if not relation_id:
            continue
        try:
            geometry = _osm_relation_polygon(session, relation_id)
            if not geometry.is_empty:
                features.append(
                    {
                        "geometry": geometry,
                        "name": next(
                            (
                                tag.get("v")
                                for tag in root.findall(
                                    f".//relation[@id='{relation_id}']/tag"
                                )
                                if tag.get("k") == "name"
                            ),
                            None,
                        ),
                        "source": "OpenStreetMap",
                        "feature_id": f"relation/{relation_id}",
                    }
                )
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
            errors.append(f"OSM waterbody relation {relation_id}: {exc}")
    for way in root.findall("way"):
        tags = {tag.get("k", ""): tag.get("v", "") for tag in way.findall("tag")}
        if not _is_osm_waterbody(tags):
            continue
        coordinates = [
            nodes.get(node.get("ref", ""))
            for node in way.findall("nd")
        ]
        coordinates = [coordinate for coordinate in coordinates if coordinate]
        if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
            continue
        geometry = Polygon(coordinates)
        if geometry.is_valid and not geometry.is_empty:
            features.append(
                {
                    "geometry": geometry,
                    "name": tags.get("name"),
                    "source": "OpenStreetMap",
                    "feature_id": f"way/{way.get('id')}",
                }
            )
    return features, errors


def _fetch_nhd_waterbodies(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch USGS NHD waterbody and area polygons for the bbox; returns (waterbody
    records, notes).
    """
    west, south, east, north = bbox_wgs84
    layer_id = 10 if max(east - west, north - south) > 0.5 else 12
    response = session.get(
        f"{NHD_MAPSERVER_URL}/{layer_id}/query",
        params={
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
            "resultRecordCount": 500,
            "f": "geojson",
        },
        timeout=(10, 45),
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"USGS NHD query failed: {data['error']}")
    features = [
        {
            "geometry": shape(feature["geometry"]),
            "name": feature.get("properties", {}).get("GNIS_NAME"),
            "source": "USGS National Hydrography Dataset",
            "feature_id": str(
                feature.get("properties", {}).get("OBJECTID", "unknown")
            ),
        }
        for feature in data.get("features", [])
        if feature.get("geometry")
    ]
    notes = (
        ["USGS NHD response was truncated at its 500-feature request limit."]
        if data.get("exceededTransferLimit")
        else []
    )
    return features, notes


def fetch_waterbodies(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
    metric_crs: int,
    bounds_metric: tuple[float, float, float, float],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Merge OSM and NHD waterbodies into projected polygons clipped to the analysis
    bounds; returns (records, notes).
    """
    west, south, east, north = bbox_wgs84
    geographic_area = (east - west) * (north - south)
    notes: list[str] = []
    candidates: list[dict[str, Any]] = []
    local_source_queried = False
    if geographic_area <= 0.25:
        try:
            candidates, osm_errors = _fetch_osm_waterbodies(session, bbox_wgs84)
            notes.extend(osm_errors)
            local_source_queried = not osm_errors
            if not candidates and local_source_queried:
                notes.append("OpenStreetMap returned no mapped waterbody in the box.")
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            notes.append(f"OSM local waterbody lookup failed: {exc}")
    if not candidates and not local_source_queried:
        try:
            candidates, nhd_notes = _fetch_nhd_waterbodies(session, bbox_wgs84)
            notes.extend(nhd_notes)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            notes.append(f"USGS NHD waterbody lookup failed: {exc}")

    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    region = box(*bounds_metric)
    waterbodies: list[dict[str, Any]] = []
    for feature in candidates:
        projected = transform(
            feature["geometry"],
            forward.transform,
            interleaved=False,
        ).intersection(region)
        if projected.is_empty or projected.area <= 0:
            continue
        waterbodies.append({**feature, "geometry": projected})
    return waterbodies, notes


def waterbody_feature_collection(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Return the bbox's mapped waterbodies as a GeoJSON FeatureCollection in WGS84 for
    GET /waterbodies.
    """
    west, south, east, north = bbox_wgs84
    center_latitude = (south + north) / 2
    center_longitude = (west + east) / 2
    validate_request(center_latitude, center_longitude, bbox_wgs84)
    metric_crs = local_utm_epsg(center_latitude, center_longitude)
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    inverse = Transformer.from_crs(f"EPSG:{metric_crs}", "EPSG:4326", always_xy=True)
    projected_bbox = transform(
        box(*bbox_wgs84),
        forward.transform,
        interleaved=False,
    )
    waterbodies, notes = fetch_waterbodies(
        session,
        bbox_wgs84,
        metric_crs,
        projected_bbox.bounds,
    )
    features = []
    for waterbody in waterbodies:
        geometry = waterbody["geometry"]
        polygons = (
            [geometry]
            if geometry.geom_type == "Polygon"
            else [part for part in geometry.geoms if part.geom_type == "Polygon"]
        )
        geographic_polygons = []
        for polygon in polygons:
            ring = polygon_ring_coordinates(polygon)
            geographic = transform(
                Polygon(ring),
                inverse.transform,
                interleaved=False,
            )
            geographic_polygons.append(
                [
                    [
                        [longitude, latitude]
                        for longitude, latitude in geographic.exterior.coords
                    ]
                ]
            )
        if not geographic_polygons:
            continue
        geometry_type = (
            "Polygon" if len(geographic_polygons) == 1 else "MultiPolygon"
        )
        coordinates = (
            geographic_polygons[0]
            if geometry_type == "Polygon"
            else geographic_polygons
        )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "category": "waterbody",
                    "name": waterbody["name"],
                    "source": waterbody["source"],
                    "source_id": waterbody["feature_id"],
                },
                "geometry": {
                    "type": geometry_type,
                    "coordinates": coordinates,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "bbox_wgs84": list(bbox_wgs84),
            "category": "mapped_surface_water_not_flood_depth",
            "notes": notes,
        },
    }
