"""MRLC NLCD 2021 land-cover source: requests a colour-coded WMS image and decodes the
palette colours back into NLCD class codes on the DEM grid.
"""

from __future__ import annotations

import io
import math

import numpy as np
import requests
import tifffile


MRLC_WMS_URL = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
MRLC_NLCD_LAYER = "NLCD_2021_Land_Cover_L48"
NLCD_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    11: (70, 107, 159),
    12: (209, 222, 248),
    21: (222, 197, 197),
    22: (217, 146, 130),
    23: (235, 0, 0),
    24: (171, 0, 0),
    31: (179, 172, 159),
    41: (104, 171, 95),
    42: (28, 95, 44),
    43: (181, 197, 143),
    51: (175, 150, 60),
    52: (204, 184, 121),
    71: (223, 223, 194),
    72: (209, 209, 130),
    73: (163, 204, 81),
    74: (130, 186, 65),
    81: (220, 217, 57),
    82: (171, 108, 40),
    90: (184, 217, 235),
    95: (108, 159, 184),
}
NLCD_COLOR_MATCH_TOLERANCE = 30


def fetch_nlcd_landcover(
    session: requests.Session,
    bounds_metric: tuple[float, float, float, float],
    metric_crs: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Request the NLCD WMS image for the area and return NLCD class codes on the DEM
    grid shape.
    """
    if len(grid_shape) != 2 or min(grid_shape) < 1:
        raise ValueError("Target land-cover grid dimensions must be positive.")
    xmin, ymin, xmax, ymax = bounds_metric
    rows = max(1, math.ceil((ymax - ymin) / 30))
    columns = max(1, math.ceil((xmax - xmin) / 30))
    response = session.get(
        MRLC_WMS_URL,
        params={
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetMap",
            "layers": MRLC_NLCD_LAYER,
            "styles": "",
            "srs": f"EPSG:{metric_crs}",
            "bbox": f"{xmin},{ymin},{xmax},{ymax}",
            "width": columns,
            "height": rows,
            "format": "image/geotiff",
            "transparent": "false",
        },
        timeout=(10, 90),
    )
    response.raise_for_status()
    try:
        with tifffile.TiffFile(io.BytesIO(response.content)) as raster:
            page = raster.pages[0]
            values = page.asarray()
            color_map_tag = page.colormap
    except (tifffile.TiffFileError, ValueError, IndexError, AttributeError) as exc:
        raise RuntimeError(f"Could not decode the MRLC NLCD GeoTIFF: {exc}") from exc
    if values.shape != (rows, columns):
        raise RuntimeError(
            f"MRLC returned land-cover shape {values.shape}; expected {(rows, columns)}."
        )
    if values.ndim != 2:
        raise RuntimeError("MRLC land-cover raster must be a single-band image.")
    return decode_nlcd_values(values, color_map_tag)


def decode_nlcd_values(
    values: np.ndarray,
    color_map: np.ndarray | None,
) -> np.ndarray:
    """Map palette-coloured image pixels to NLCD class codes; raises ValueError on
    unrecognised colours.
    """
    if color_map is None:
        decoded = values.astype(np.int64)
        recognized = np.isin(decoded, list(NLCD_CLASS_COLORS) + [0])
        if not recognized.all():
            unknown = np.unique(decoded[~recognized]).tolist()
            raise RuntimeError(
                f"MRLC returned unrecognized NLCD class values: {unknown[:10]}."
            )
        return decoded.astype(np.uint8)

    palette = np.asarray(color_map)
    if palette.shape[0] != 3:
        raise RuntimeError(
            f"MRLC returned an unsupported NLCD color-map shape {palette.shape}."
        )
    palette_rgb = np.rint(palette.astype(float) / 257.0).astype(np.int16).T
    class_colors = np.asarray(list(NLCD_CLASS_COLORS.values()), dtype=np.int16)
    class_codes = np.asarray(list(NLCD_CLASS_COLORS), dtype=np.uint8)
    decoded = np.zeros(values.shape, dtype=np.uint8)
    unknown_indices: list[int] = []
    for index in np.unique(values):
        index_int = int(index)
        if index_int == 0:
            continue
        if index_int < 0 or index_int >= palette_rgb.shape[0]:
            unknown_indices.append(index_int)
            continue
        color = palette_rgb[index_int]
        distances = np.max(np.abs(class_colors - color), axis=1)
        nearest = int(np.argmin(distances))
        if int(distances[nearest]) > NLCD_COLOR_MATCH_TOLERANCE:
            unknown_indices.append(index_int)
            continue
        decoded[values == index] = class_codes[nearest]
    if unknown_indices:
        raise RuntimeError(
            "MRLC returned NLCD palette entries that do not match known class "
            f"colors: {unknown_indices[:10]}."
        )
    return decoded
