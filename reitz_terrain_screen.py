from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
import tifffile
from pyproj import Transformer
from shapely import transform
from shapely.geometry import LineString, Point

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
THREEDEP_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)
REITZ_UNION = {"latitude": 29.64489, "longitude": -82.34834}
ROAD_QUERY = (
    '[out:json][timeout:25];'
    'way["name"="Reitz Union Drive"]["highway"="service"]'
    f'(around:180,{REITZ_UNION["latitude"]},{REITZ_UNION["longitude"]});out geom;'
)
PROJECT = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
UNPROJECT = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)
OUTPUT_DIR = Path("outputs/gainesville_heartbeat")


def request_road(
    session: requests.Session,
) -> tuple[dict[str, Any], LineString, float]:
    response = session.post(
        OVERPASS_URL,
        data={"data": ROAD_QUERY},
        timeout=(10, 60),
    )
    response.raise_for_status()
    ways = response.json().get("elements", [])
    if not ways:
        raise RuntimeError("OpenStreetMap returned no Reitz Union Drive geometry.")
    way = max(ways, key=lambda item: len(item.get("geometry", [])))
    coordinates = [
        (point["lon"], point["lat"]) for point in way.get("geometry", [])
    ]
    if len(coordinates) < 2:
        raise RuntimeError("OpenStreetMap returned an incomplete road geometry.")
    try:
        width_m = float(way.get("tags", {}).get("width", 5))
    except (TypeError, ValueError):
        width_m = 5.0
    road_metric = transform(
        LineString(coordinates), PROJECT.transform, interleaved=False
    )
    return way, road_metric, width_m


def get_elevations(
    session: requests.Session,
    road_metric: LineString,
) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    xmin, ymin, xmax, ymax = road_metric.bounds
    side_m = 100.0
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    bounds = (
        center_x - side_m / 2,
        center_y - side_m / 2,
        center_x + side_m / 2,
        center_y + side_m / 2,
    )
    response = session.get(
        THREEDEP_EXPORT_URL,
        params={
            "bbox": ",".join(str(value) for value in bounds),
            "bboxSR": 32617,
            "size": "100,100",
            "imageSR": 32617,
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
        if page.shape != (100, 100):
            raise RuntimeError(f"Expected 100x100 DEM pixels, received {page.shape}.")
        scale = page.tags.get("ModelPixelScaleTag")
        if scale is None or abs(float(scale.value[0]) - 1.0) > 0.01:
            raise RuntimeError("Elevation response did not provide a 1 m pixel size.")
        no_data_tag = page.tags.get("GDAL_NODATA")
        no_data = float(no_data_tag.value) if no_data_tag else None
        values = page.asarray()
    return values, bounds, no_data


def screen_low_points(
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    no_data: float | None,
    road_metric: LineString,
    road_width_m: float,
    count: int = 5,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    xmin, _, _, ymax = bounds
    road_corridor = road_metric.buffer(road_width_m / 2)
    samples: list[dict[str, Any]] = []
    terrain_cells: list[dict[str, Any]] = []
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            elevation = float(values[row, column])
            if not np.isfinite(elevation) or (
                no_data is not None and elevation == no_data
            ):
                continue
            x = xmin + column + 0.5
            y = ymax - row - 0.5
            longitude, latitude = UNPROJECT.transform(x, y)
            cell = {
                "longitude": longitude,
                "latitude": latitude,
                "elevation_m": elevation,
                "x": x,
                "y": y,
                "row": row,
                "column": column,
                "on_mapped_road": road_corridor.covers(Point(x, y)),
            }
            terrain_cells.append(cell)
            if cell["on_mapped_road"]:
                samples.append(cell)
    if not samples:
        raise RuntimeError("The DEM returned no valid pixels along the road corridor.")

    elevations = np.asarray([sample["elevation_m"] for sample in samples])
    stats = {
        "minimum": float(np.min(elevations)),
        "p10": float(np.percentile(elevations, 10)),
        "median": float(np.median(elevations)),
        "maximum": float(np.max(elevations)),
    }
    separated: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: item["elevation_m"]):
        if all(
            np.hypot(sample["x"] - chosen["x"], sample["y"] - chosen["y"]) >= 5
            for chosen in separated
        ):
            separated.append(sample)
        if len(separated) == count:
            break
    if len(terrain_cells) != 10_000:
        raise RuntimeError(
            f"Expected 10,000 valid 1 m terrain cells, received {len(terrain_cells)}."
        )
    return separated, stats, len(samples), samples, terrain_cells


def bathtub_depths(
    samples: list[dict[str, Any]],
    rainfall_inches: float,
) -> tuple[float, list[float]]:
    if not samples:
        raise ValueError("At least one terrain sample is required.")
    if rainfall_inches < 0:
        raise ValueError("Rainfall cannot be negative.")
    elevations = [sample["elevation_m"] for sample in samples]
    target_volume_m3 = rainfall_inches * 0.0254 * len(samples)
    low = min(elevations)
    high = max(elevations) + rainfall_inches * 0.0254
    for _ in range(80):
        level = (low + high) / 2
        volume = sum(max(level - elevation, 0.0) for elevation in elevations)
        if volume < target_volume_m3:
            low = level
        else:
            high = level
    water_level = (low + high) / 2
    return water_level, [
        max(water_level - elevation, 0.0) for elevation in elevations
    ]


def write_depth_grid(
    terrain_cells: list[dict[str, Any]],
    road_width_m: float,
) -> None:
    cells = []
    for sample in terrain_cells:
        cells.append(
            {
                "row": sample["row"],
                "column": sample["column"],
                "latitude": sample["latitude"],
                "longitude": sample["longitude"],
                "elevation_m": sample["elevation_m"],
                "on_mapped_road": sample["on_mapped_road"],
            }
        )

    html_path = OUTPUT_DIR / "reitz_water_depth_grid.html"
    html_path.write_text(
        _depth_grid_html(cells, road_width_m),
        encoding="utf-8",
    )
    csv_path = OUTPUT_DIR / "reitz_water_depth_scenarios.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rainfall_inches",
                "row",
                "column",
                "latitude",
                "longitude",
                "ground_elevation_m",
                "modeled_depth_inches",
                "modeled_water_surface_m",
            ]
        )
        for rainfall_inches in range(1, 11):
            water_level, depths = bathtub_depths(terrain_cells, rainfall_inches)
            for cell, depth in zip(cells, depths):
                writer.writerow(
                    [
                        rainfall_inches,
                        cell["row"],
                        cell["column"],
                        f"{cell['latitude']:.7f}",
                        f"{cell['longitude']:.7f}",
                        f"{cell['elevation_m']:.3f}",
                        f"{depth / 0.0254:.3f}",
                        f"{water_level:.3f}",
                    ]
                )
    source_status_path = OUTPUT_DIR / "source_status.json"
    if source_status_path.exists():
        source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
        source_status["illustrative_water_depth_model"] = {
            "status": "illustrative_model",
            "rows_or_features_obtained": len(terrain_cells) * 10,
            "rows_or_features_retained_locally": len(terrain_cells) * 10,
            "grid_cell_count": len(terrain_cells),
            "mapped_road_cell_count": sum(
                cell["on_mapped_road"] for cell in terrain_cells
            ),
            "model": "closed bathtub, rainfall on full 100x100 m terrain grid, instantaneous redistribution, no infiltration, drainage, boundary spill, or external runoff",
            "warning": "Not an observed or validated flood-depth estimate.",
            "outputs": [
                html_path.name,
                csv_path.name,
            ],
        }
        source_status_path.write_text(
            json.dumps(source_status, indent=2) + "\n",
            encoding="utf-8",
        )


def _depth_grid_html(cells: list[dict[str, Any]], road_width_m: float) -> str:
    cells_json = json.dumps(cells, separators=(",", ":"))
    title = html.escape("Reitz Union Drive illustrative water-depth grid")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; font-family: "Segoe UI", Arial, sans-serif; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: #18363c; background: #edf3f2; }}
  main {{ max-width: 1100px; margin: 24px auto; padding: 0 18px; }}
  header {{ display:flex; align-items:end; justify-content:space-between; gap:20px; flex-wrap:wrap; }}
  h1 {{ margin: 4px 0 8px; font-size: clamp(24px, 4vw, 38px); }}
  .eyebrow {{ color:#397779; font-weight:700; letter-spacing:.12em; font-size:12px; }}
  .sub {{ margin:0 0 18px; color:#577074; }}
  .warning {{ padding:14px 16px; border:1px solid #e3bf70; border-radius:12px; background:#fff7df; color:#684b16; line-height:1.45; }}
  .panel {{ margin-top:16px; padding:16px; background:#fff; border:1px solid #d8e4e2; border-radius:16px; box-shadow:0 8px 26px #15343a12; }}
  .controls {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
  input[type=range] {{ accent-color:#087f9e; flex:1; min-width:220px; }}
  output {{ min-width:115px; padding:9px 12px; border-radius:9px; background:#e5f5f6; color:#07546a; font-size:19px; font-weight:800; text-align:center; }}
  canvas {{ display:block; width:100%; margin-top:10px; border-radius:12px; background:linear-gradient(#eaf4f5,#f9fbf8); }}
  .legend {{ display:flex; gap:9px; align-items:center; color:#607679; font-size:13px; }}
  .swatch {{ width:22px; height:13px; border-radius:4px; background:#25a9cf; }}
  #road-result {{ margin:12px 0 0; padding:10px 12px; border-radius:9px; background:#fff7df; color:#684b16; line-height:1.4; }}
  #readout {{ min-height:24px; margin-top:10px; color:#34565c; font-variant-numeric:tabular-nums; }}
  .foot {{ color:#718589; font-size:12px; line-height:1.5; }}
  @media(max-width:600px) {{ main {{ padding:0 10px; }} .panel {{ padding:10px; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div><div class="eyebrow">FLOODSIGHT · TERRAIN TOY MODEL</div>
    <h1>Reitz Union Drive — water depth grid</h1>
    <p class="sub">100 × 100 m DEM sample · {road_width_m:g} m mapped road corridor · 1 m cells</p></div>
  </header>
  <div class="warning"><strong>Illustrative only—not a flood prediction.</strong>
  Assumes rainfall lands on the full 100 × 100 m terrain grid and all water instantly redistributes inside a closed tray.
  No drainage, infiltration, boundary spill, or outside runoff is modeled. Real street depth can be very different.
  A dry result in this toy model does not mean the real road is safe.</div>
  <section class="panel">
    <div class="controls">
      <label for="rain">Hypothetical storm total</label>
      <input id="rain" type="range" min="1" max="10" step="1" value="1" aria-label="Hypothetical rainfall in inches">
      <output id="rainLabel" for="rain">1 inch</output>
      <div class="legend"><span class="swatch"></span> modeled standing-water depth</div>
    </div>
    <p id="road-result" aria-live="polite"></p>
    <canvas id="grid" width="1000" height="650" aria-label="Isometric 3D grid of illustrative water depths across the terrain"></canvas>
    <div id="readout" aria-live="polite">Hover a grid cell to inspect its elevation and modeled water depth. Road-corridor cells are outlined.</div>
    <p class="foot">Blue prisms show modeled water depth above each terrain cell; outlined cells fall within the mapped road corridor. Heights are visually exaggerated. Values are not measured water.</p>
  </section>
</main>
<script>
const cells={cells_json};
const canvas=document.getElementById("grid"),ctx=canvas.getContext("2d");
const slider=document.getElementById("rain"),label=document.getElementById("rainLabel"),readout=document.getElementById("readout"),roadResult=document.getElementById("road-result");
const minElev=Math.min(...cells.map(c=>c.elevation_m));
const maxElev=Math.max(...cells.map(c=>c.elevation_m));
const depthScale=90, groundScale=5, tileX=4.8, tileY=2.7;
const originX=canvas.width/2, originY=55;
let hitBoxes=[];
function scenario(inches) {{
 const rainM=inches*0.0254,target=rainM*cells.length;
 let lo=minElev,hi=maxElev+rainM;
 for(let i=0;i<70;i++) {{
   const level=(lo+hi)/2;
   const volume=cells.reduce((s,c)=>s+Math.max(0,level-c.elevation_m),0);
   if(volume<target)lo=level;else hi=level;
 }}
 const level=(lo+hi)/2;
 return cells.map(c=>({{...c,depth:Math.max(0,level-c.elevation_m),waterLevel:level}}));
}}
function color(depth) {{
 if(depth<=0.0001)return "#82978f";
 const inches=depth/0.0254,t=Math.min(1,inches/10);
 const r=Math.round(93+(18-93)*t),g=Math.round(202+(119-202)*t),b=Math.round(225+(168-225)*t);
 return `rgb(${{r}},${{g}},${{b}})`;
}}
function draw() {{
 const inches=Number(slider.value);label.value=`${{inches}} ${{inches===1?"inch":"inches"}}`;
 const scenarioCells=scenario(inches);
 const roadCells=scenarioCells.filter(c=>c.on_mapped_road);
 const wetRoadCells=roadCells.filter(c=>c.depth>0.001);
 const maxRoadDepth=wetRoadCells.reduce((max,c)=>Math.max(max,c.depth),0)/0.0254;
 roadResult.textContent=wetRoadCells.length
   ? `Toy-model road result: ${{wetRoadCells.length}} of ${{roadCells.length}} mapped road cells show water (maximum ${{maxRoadDepth.toFixed(2)}} in). Not a real-road flood estimate.`
   : `Toy-model road result: 0 of ${{roadCells.length}} mapped road cells show water at ${{inches}} ${{inches===1?"inch":"inches"}}. This is not evidence the real road stays dry.`;
 ctx.clearRect(0,0,canvas.width,canvas.height);
 ctx.fillStyle="#eaf4f5";ctx.fillRect(0,0,canvas.width,canvas.height);
 const floorWidth=480,floorDepth=270;
 // The full elevation grid forms the terrain plane; road cells are outlined.
 ctx.beginPath();ctx.moveTo(originX,originY);ctx.lineTo(originX+floorWidth,originY+floorDepth);ctx.lineTo(originX,originY+floorDepth*2);ctx.lineTo(originX-floorWidth,originY+floorDepth);ctx.closePath();
 ctx.fillStyle="#e1ece9";ctx.fill();ctx.strokeStyle="#bacdc9";ctx.lineWidth=1;ctx.stroke();
 for(let i=1;i<10;i++) {{
   const t=i/10;
   ctx.beginPath();ctx.moveTo(originX+floorWidth*t,originY+floorDepth*t);ctx.lineTo(originX-floorWidth+floorWidth*t,originY+floorDepth+floorDepth*t);ctx.strokeStyle="#cfddda";ctx.stroke();
   ctx.beginPath();ctx.moveTo(originX-floorWidth*t,originY+floorDepth*t);ctx.lineTo(originX+floorWidth-floorWidth*t,originY+floorDepth+floorDepth*t);ctx.stroke();
 }}
 hitBoxes=[];
 const drawn=scenarioCells.sort((a,b)=>(a.row+a.column)-(b.row+b.column));
 for(const c of drawn) {{
   const x=originX+(c.column-c.row)*tileX;
   const y=originY+(c.column+c.row)*tileY-(c.elevation_m-minElev)*groundScale;
   const h=c.depth*depthScale;
   const p=[[x,y-tileY],[x+tileX,y],[x,y+tileY],[x-tileX,y]];
   if(h>0.001) {{
     // Visible front and right faces show the assumed water height.
     ctx.beginPath();ctx.moveTo(x+tileX,y);ctx.lineTo(x,y+tileY);ctx.lineTo(x,y+tileY-h);ctx.lineTo(x+tileX,y-h);ctx.closePath();
     ctx.fillStyle="#087fa0";ctx.fill();
     ctx.beginPath();ctx.moveTo(x-tileX,y);ctx.lineTo(x,y+tileY);ctx.lineTo(x,y+tileY-h);ctx.lineTo(x-tileX,y-h);ctx.closePath();
     ctx.fillStyle="#126582";ctx.fill();
   }}
   ctx.beginPath();ctx.moveTo(...p[0]);for(const q of p.slice(1))ctx.lineTo(...q);ctx.closePath();
   ctx.fillStyle=color(c.depth);ctx.fill();ctx.strokeStyle=c.on_mapped_road?"#203d39":"#eff9f8";ctx.lineWidth=c.on_mapped_road?1.1:.45;ctx.stroke();
   if(h>0.001) {{
     ctx.beginPath();ctx.moveTo(x,y-tileY-h);ctx.lineTo(x+tileX,y-h);ctx.lineTo(x,y+tileY-h);ctx.lineTo(x-tileX,y-h);ctx.closePath();
     ctx.fillStyle=color(c.depth);ctx.fill();ctx.strokeStyle="#f5ffff";ctx.stroke();
   }}
   hitBoxes.push({{x,y,c}});
 }}
 ctx.fillStyle="#35575c";ctx.font="13px Segoe UI,Arial";ctx.textAlign="left";
 ctx.fillText("100 × 100 terrain cells · modeled water columns · dark outlines = mapped road",20,30);
 ctx.fillStyle="#6c8384";ctx.font="12px Segoe UI,Arial";
 ctx.fillText(`Cell height exaggerated · ${{drawn.filter(c=>c.depth>0.001).length}} / ${{cells.length}} terrain cells hold water in this closed-tray scenario`,20,50);
}}
canvas.addEventListener("mousemove",event=>{{
 const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height;
 let closest=null,best=20;
 for(const h of hitBoxes){{const d=Math.hypot(x-h.x,y-h.y);if(d<best){{best=d;closest=h;}}}}
 if(!closest){{readout.textContent="Hover a road cell to inspect its elevation and modeled water depth.";return;}}
 const c=closest.c;
 readout.textContent=`Cell ${{c.row}},${{c.column}} · ground ${{c.elevation_m.toFixed(2)}} m · modeled depth ${{(c.depth/0.0254).toFixed(2)}} in · ${{c.latitude.toFixed(7)}}, ${{c.longitude.toFixed(7)}}`;
}});
slider.addEventListener("input",draw);draw();
</script>
</body>
</html>
"""


def write_outputs(
    way: dict[str, Any],
    road_metric: LineString,
    road_width_m: float,
    points: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    terrain_cells: list[dict[str, Any]],
    stats: dict[str, float],
    sample_count: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "reitz_low_spots.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["rank", "latitude", "longitude", "elevation_m"],
        )
        writer.writeheader()
        for rank, point in enumerate(points, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "latitude": f"{point['latitude']:.7f}",
                    "longitude": f"{point['longitude']:.7f}",
                    "elevation_m": f"{point['elevation_m']:.2f}",
                }
            )

    report = [
        "# Reitz Union Drive terrain screening",
        "",
        "- Purpose: locate relatively low sampled points along the mapped car loop; this is not a flood-depth or rainfall model.",
        "- Area: the longest mapped OpenStreetMap `Reitz Union Drive` service-road segment near the Reitz Union.",
        "- Elevation: USGS 3DEP ImageServer, 1 m raster cells, sampled in UTM zone 17N. Raster resolution is not a guarantee of 1 m vertical or horizontal accuracy.",
        f"- Road mask: {road_width_m:g} m corridor based on the OpenStreetMap road width; geometry is approximate and should be checked against the intended loop.",
        f"- Valid road-corridor elevation pixels: {sample_count}.",
        "",
        "## Elevation range in the road corridor",
        "",
        "| Statistic | Elevation (m) |",
        "|---|---:|",
        f"| Minimum | {stats['minimum']:.2f} |",
        f"| 10th percentile | {stats['p10']:.2f} |",
        f"| Median | {stats['median']:.2f} |",
        f"| Maximum | {stats['maximum']:.2f} |",
        "",
        "## Lowest separated sample points",
        "",
        "| Rank | Latitude | Longitude | Elevation (m) |",
        "|---:|---:|---:|---:|",
    ]
    for rank, point in enumerate(points, start=1):
        report.append(
            f"| {rank} | {point['latitude']:.7f} | {point['longitude']:.7f} | {point['elevation_m']:.2f} |"
        )
    report.extend(
        [
            "",
            "## Why rainfall-to-depth values are not shown",
            "",
            "A 1–10 inch storm total does not determine street-water depth by itself. This terrain sample does not include curb elevations, storm-drain locations/capacity, runoff from the contributing area, or storm duration/intensity. Low terrain points are screening candidates only; pooling locations and inches of water require those additional inputs and a validated hydraulic model.",
            "",
            f"Source: [USGS 3DEP Elevation ImageServer]({THREEDEP_EXPORT_URL}); road geometry from [OpenStreetMap](https://www.openstreetmap.org/way/{way['id']}) (ODbL attribution).",
            "",
            "Elevation values describe terrain relative to the service's elevation reference; they are not water depths.",
            "",
        ]
    )
    (OUTPUT_DIR / "reitz_terrain_screening.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [point["longitude"], point["latitude"]],
                },
                "properties": {
                    "rank": rank,
                    "elevation_m": point["elevation_m"],
                    "interpretation": "low-terrain screening point; not a predicted pool",
                },
            }
            for rank, point in enumerate(points, start=1)
        ],
    }
    (OUTPUT_DIR / "reitz_low_spots.geojson").write_text(
        json.dumps(feature_collection, indent=2) + "\n",
        encoding="utf-8",
    )
    write_depth_grid(terrain_cells, road_width_m)
    source_status_path = OUTPUT_DIR / "source_status.json"
    if source_status_path.exists():
        source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
        source_status["terrain_screening"] = {
            "source_name": "USGS 3DEP elevation with OpenStreetMap road geometry",
            "status": "success",
            "request_timestamp": datetime.now(timezone.utc).isoformat(),
            "rows_or_features_obtained": len(terrain_cells),
            "rows_or_features_retained_locally": sample_count,
            "grid_cell_count": len(terrain_cells),
            "mapped_road_cell_count": len(samples),
            "acquisition": "live",
            "elevation_endpoint": THREEDEP_EXPORT_URL,
            "road_geometry_url": f"https://www.openstreetmap.org/way/{way['id']}",
            "road_width_m": road_width_m,
            "sampled_cell_size_m": 1,
            "warning": "Terrain screening only; does not estimate flood depth.",
        }
        source_status_path.write_text(
            json.dumps(source_status, indent=2) + "\n",
            encoding="utf-8",
        )
    metadata_path = OUTPUT_DIR / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["terrain_screening"] = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "center": REITZ_UNION,
            "road_width_m": road_width_m,
            "sampled_cells": sample_count,
            "elevation_raster_cell_size_m": 1,
            "elevation_endpoint": THREEDEP_EXPORT_URL,
            "road_geometry_url": f"https://www.openstreetmap.org/way/{way['id']}",
            "model": "USGS 3DEP elevation grid with OpenStreetMap road corridor overlay; illustrative water scenario is a closed tray",
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    session = requests.Session()
    session.headers["User-Agent"] = "FloodSight-GatorHacks/0.1 (OpenStreetMap data use)"
    way, road, road_width_m = request_road(session)
    values, bounds, no_data = get_elevations(session, road)
    points, stats, sample_count, samples, terrain_cells = screen_low_points(
        values, bounds, no_data, road, road_width_m
    )
    write_outputs(
        way,
        road,
        road_width_m,
        points,
        samples,
        terrain_cells,
        stats,
        sample_count,
    )
    print(
        f"Sampled {sample_count} 1 m terrain cells along Reitz Union Drive; "
        f"elevation range {stats['minimum']:.2f}-{stats['maximum']:.2f} m."
    )
    print(f"Wrote {OUTPUT_DIR / 'reitz_terrain_screening.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
