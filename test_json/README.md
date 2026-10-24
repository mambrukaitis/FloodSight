# Reitz Union flood JSON demo

Run from the FloodSight workspace root:

```powershell
.\.venv\Scripts\python.exe -m test_json.service
```

Override the 10-inch demo rainfall assumption or retained-water fraction:

```powershell
.\.venv\Scripts\python.exe -m test_json.service --rainfall-inches 10 --retained-water-fraction 0.02
```

To run the HTTP microservice locally:

```powershell
.\.venv\Scripts\python.exe -m test_json.service --serve
```

Send a POST to `http://127.0.0.1:8000/flood` with JSON containing
`latitude`, `longitude`, `bbox_wgs84: [west, south, east, north]`, and
optionally `rainfall_inches` and `retained_water_fraction`. Defaults are 10
inches of rain and 0.02 retained. A valid request returns only the iOS payload;
malformed requests return HTTP 400 and failed upstream/data requests return
HTTP 503 with an `error` field.

The service function is `build_flood_payload(latitude, longitude, bbox_wgs84,
rainfall_inches=10, retained_water_fraction=0.02)`.
The bounding box is `(west, south, east, north)` in EPSG:4326. The example uses
the existing Reitz Union Drive coordinate and a 100 m by 100 m projected box.
It requests the local DEM from USGS 3DEP, queries nearby stream observations
from USGS Water Data for the Nation v1 API, and scans/caches the Groundsource
Parquet archive for historical event footprints intersecting the box.

The command writes `reitz_union_flood.json` with the exact two top-level keys
expected by the Swift model, and `reitz_union_diagnostics.json` separately.
The diagnostic file records raw DEM elevation, raster/datum metadata, nearby
gage availability, and historical Groundsource scan results.

Run the service tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest test_json.test_service -v
```

## Interpretation and limitations

The payload is numeric and uses the exact two top-level keys required by the
Swift shape. Its values are an **illustrative scenario**, not observed water
levels or a calibrated forecast:

- 3DEP supplies the elevation raster. This Reitz sample is NAVD 88. The
  payload's MSL-named fields use the DEM's native datum as an explicit local
  approximation; the service does not apply a NAVD 88 to local mean-sea-level
  conversion. Depth arithmetic is internally consistent because it subtracts
  elevations in the same DEM datum, but absolute AR altitude can be offset.
- `retained_water_fraction` defaults to 0.02. With 10 inches of rain, this
  retains the equivalent of 0.2 inches over the box, then redistributes it over
  low DEM cells as a closed-basin “bathtub.” This 2% is a transparent demo
  assumption (matching the illustrative example), **not** a coefficient
  measured by WDFN, Groundsource, soils, or land cover.
- WDFN and Groundsource do not numerically set the runoff fraction: current
  local gage heights and pointwise historical depths are not available. They
  are queried and included in diagnostics, but cannot scientifically calibrate
  this scenario. NLCD and SSURGO are not yet wired into this service's runoff
  calculation.
- USGS WDFN v1 is queried for nearby stations and gage heights. At Reitz, the
  closest streamgage has no current stage reading; its latest daily value is
  historical. A gage-stage observation is not assigned across the bounding
  box because no gage-datum/elevation tie or hydraulic connection is available.
- Groundsource is scanned for local historical event footprints. It contains no
  point depth values, so event overlap is diagnostic context, not calibration.
- No storm duration, infiltration, storm-drain capacity, boundary flow, or
  runoff from outside the box is modeled. Contours are generated from DEM
  raster math at 0.3 m and 1.0 m; the required single-ring coordinate format
  cannot represent holes and therefore emits polygon exteriors only.

Use the output for a demo visualization only, not safety decisions. A
scientifically defensible prediction needs event rainfall timing, local
hydraulic/drainage data, and observed depths for calibration. The output schema
has no confidence/provenance fields, so the returned JSON itself cannot convey
these caveats; see the generated diagnostics JSON and this README.
