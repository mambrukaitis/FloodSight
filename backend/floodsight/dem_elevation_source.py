"""USGS 3DEP elevation source: predicts the DEM grid size for an area and downloads the
elevation raster (GeoTIFF) from the 3DEP ImageServer.
"""

from __future__ import annotations

import io
import math

import numpy as np
from pyproj import Transformer
import requests
from shapely import transform
from shapely.geometry import box
import tifffile

from floodsight.grid_geometry import local_utm_epsg


THREEDEP_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)
THREEDEP_IDENTIFY_URL = THREEDEP_EXPORT_URL.rsplit("/", 1)[0] + "/identify"
MAX_DEM_PIXELS = 20_000


def dem_grid_shape(
    bbox_wgs84: tuple[float, float, float, float],
    metric_crs: int,
) -> tuple[int, int]:
    """Predict the (rows, cols) of the DEM raster that `fetch_dem` will request for
    this area, capped at MAX_DEM_PIXELS.
    """
    projected = transform(
        box(*bbox_wgs84),
        Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True).transform,
        interleaved=False,
    )
    xmin, ymin, xmax, ymax = projected.bounds
    width_m, height_m = xmax - xmin, ymax - ymin
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Bounding box has no area after projection.")
    pixel_size_m = max(1.0, math.sqrt(width_m * height_m / MAX_DEM_PIXELS))
    columns = max(1, math.ceil(width_m / pixel_size_m))
    rows = max(1, math.ceil(height_m / pixel_size_m))
    while rows * columns > MAX_DEM_PIXELS:
        pixel_size_m *= math.sqrt(rows * columns / MAX_DEM_PIXELS)
        columns = max(1, math.ceil(width_m / pixel_size_m))
        rows = max(1, math.ceil(height_m / pixel_size_m))
    return rows, columns


def fetch_dem(
    session: requests.Session,
    latitude: float,
    longitude: float,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float], float | None, int]:
    """Download the 3DEP elevation raster for the bbox; returns (values, projected
    bounds, nodata value, metric CRS).
    """
    metric_crs = local_utm_epsg(latitude, longitude)
    forward = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    region_metric = transform(
        box(*bbox_wgs84),
        forward.transform,
        interleaved=False,
    )
    xmin, ymin, xmax, ymax = region_metric.bounds
    width_m = xmax - xmin
    height_m = ymax - ymin
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Bounding box has no area after projection.")
    pixel_size_m = max(1.0, math.sqrt(width_m * height_m / MAX_DEM_PIXELS))
    columns = max(1, math.ceil(width_m / pixel_size_m))
    rows = max(1, math.ceil(height_m / pixel_size_m))
    while rows * columns > MAX_DEM_PIXELS:
        pixel_size_m *= math.sqrt(rows * columns / MAX_DEM_PIXELS)
        columns = max(1, math.ceil(width_m / pixel_size_m))
        rows = max(1, math.ceil(height_m / pixel_size_m))

    response = session.get(
        THREEDEP_EXPORT_URL,
        params={
            "bbox": ",".join(f"{value:.8f}" for value in (xmin, ymin, xmax, ymax)),
            "bboxSR": metric_crs,
            "size": f"{columns},{rows}",
            "imageSR": metric_crs,
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_NearestNeighbor",
            "adjustAspectRatio": "false",
            "f": "image",
        },
        timeout=(10, 90),
    )
    response.raise_for_status()
    with tifffile.TiffFile(io.BytesIO(response.content)) as raster:
        page = raster.pages[0]
        values = page.asarray().astype(float)
        if values.shape != (rows, columns):
            raise RuntimeError(
                f"3DEP returned raster shape {values.shape}; expected {(rows, columns)}."
            )
        scale_tag = page.tags.get("ModelPixelScaleTag")
        tiepoint_tag = page.tags.get("ModelTiepointTag")
        if scale_tag is None or tiepoint_tag is None:
            raise RuntimeError("3DEP TIFF is missing georeferencing tags.")
        scale_x, scale_y = map(float, scale_tag.value[:2])
        i0, j0, _, x0, y0, _ = map(float, tiepoint_tag.value[:6])
        if scale_x <= 0 or scale_y <= 0:
            raise RuntimeError("3DEP TIFF has invalid pixel scale.")
        raster_bounds = (
            x0 - i0 * scale_x,
            y0 - (j0 + rows) * scale_y,
            x0 + (columns - i0) * scale_x,
            y0 - j0 * scale_y,
        )
        nodata_tag = page.tags.get("GDAL_NODATA")
        nodata = float(nodata_tag.value) if nodata_tag else None
        geo_keys = page.tags.get("GeoKeyDirectoryTag")
        if geo_keys is None or metric_crs not in geo_keys.value:
            raise RuntimeError("3DEP raster projection does not match the requested UTM CRS.")
        return values, raster_bounds, nodata, metric_crs
