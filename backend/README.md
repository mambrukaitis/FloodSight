# FloodSight heartbeat test

This is a small data-feasibility program for comparing historical Groundsource
flood events, nearby USGS monitoring data, and FEMA NFHL hazard zones around
Gainesville, Florida. It reports observed data; it does not predict floods or
prove a hazard-map error.

## Run

From the repository root with Python 3.11 or newer:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python flood_heartbeat.py
```

By default, the program downloads and caches the Groundsource Parquet file
(approximately 667 MB), then writes the local data and report to
`outputs/gainesville_heartbeat/`. The download is checksum-verified and reused
on later runs. To use a supplied Parquet file:

```powershell
python flood_heartbeat.py --groundsource-parquet C:\data\groundsource_2026.parquet
```

To test USGS and FEMA without starting a Groundsource download:

```powershell
python flood_heartbeat.py --skip-groundsource-download
```

The study location can be changed with `--latitude`, `--longitude`, and
`--radius-km`. Use `--output-dir` and `--cache-dir` to change output and cache
locations.

The generated `summary.md`, JSON status/metadata files, CSV tables, and GeoJSON
features are written under the output directory. The `.cache/` and `outputs/`
directories are ignored by Git.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Reitz Union terrain screen

To sample elevation along the mapped Reitz Union Drive loop and list low
terrain points (not flood depths), run:

```powershell
python reitz_terrain_screen.py
```

This uses USGS 3DEP elevation pixels and OpenStreetMap road geometry. Outputs
are `reitz_terrain_screening.md`, `reitz_low_spots.csv`, and
`reitz_low_spots.geojson` in `outputs/gainesville_heartbeat/`. Rainfall totals
cannot be converted into pooling depths without local drainage and storm data.

It also writes an interactive, explicitly illustrative closed-tray water-depth
grid and CSV for 1–10 inch rainfall scenarios. The toy model assumes water
lands on a 100 × 100 m terrain grid and redistributes instantly without drains,
infiltration, or boundary spill. The grid reports how many mapped road cells it
marks wet for each slider setting; a dry result is not evidence that the real
road is safe. Do not treat its displayed depths as real street-flood estimates.

## Small Reitz Union 2×2 demo

To make a small, on-campus source-screening map around Reitz Union, run:

```powershell
python reitz_2x2_demo.py --rainfall-inches 3
```

The script requests a 100 × 100 m USGS 3DEP raster and divides it into four
50 × 50 m cells. It overlays any intersecting FEMA zones from the existing
cached FEMA GeoJSON, lists nearby cached USGS stations/readings, and scans the
cached Groundsource Parquet archive for historical footprints at the site. The
scan reads the full file because it has no spatial index. It writes
`reitz_2x2_screening.geojson`, `reitz_2x2_screening.html`, and
`reitz_groundsource_full_scan.json` under `outputs/gainesville_heartbeat/`.
To use a different local Groundsource file:

```powershell
python reitz_2x2_demo.py --rainfall-inches 3 --groundsource-parquet C:\data\groundsource_2026.parquet
```

This is a data inventory and mapping demo, not a flood forecast. The scan found
25 historical Groundsource footprints intersecting the 100 m area (2014–2024),
but the archive has no water-depth field and the footprints are too broad to
resolve campus pooling. The City of Gainesville Flood Reporter has approximate
depth-in-inches values at submitted point locations, but no reports were found
within 1 km of Reitz Union. FEMA's live campus polygon is Zone X with no usable
base-flood elevation or depth value. Nearby USGS gage-height observations are
point measurements and are not current at this site.

The City stormwater-network layer provides infrastructure features and some
design fields, but this 100 m area has no mapped assets; the nearest lines
within 300 m have missing invert, slope, and size attributes. The City layer's
metadata warns that its infrastructure data is incomplete and may be inaccurate.
Terrain is available at 1 m resolution, while the reviewed land-cover data is
30 m and SSURGO provides map-unit soil classes, not measured site infiltration.
An actual rainfall-to-depth model still needs a storm duration/hyetograph,
usable campus drain and outfall details, surface roughness/runoff parameters,
and local depth observations for validation. The rainfall total alone cannot
determine water depth.
