# Reitz Union flood JSON demo

Run from the FloodSight workspace root:

```powershell
.\.venv\Scripts\python.exe -m test_json.service
```

To run the HTTP microservice locally:

```powershell
.\.venv\Scripts\python.exe -m test_json.service --serve
```

Send a POST to `http://127.0.0.1:8000/flood` with JSON containing
`latitude`, `longitude`, and `bbox_wgs84: [west, south, east, north]`. A valid
request returns only the iOS payload; malformed requests return HTTP 400 and
failed upstream/data requests return HTTP 503 with an `error` field.

The service function is `build_flood_payload(latitude, longitude, bbox_wgs84)`.
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

**A null value is unknown, not dry.** The Swift model should make the AR values
optional so unavailable measurements are distinct from a confirmed dry result.
On the verified Reitz run, the 3DEP tile
reported NAVD 88 rather than MSL, no WDFN gage-height observation was available
from a stream station within 5 km, and the closest streamgage was 1.465 km away
with no vertical datum in its station metadata. Groundsource contains
historical event footprints without water depths. Consequently the output
leaves MSL elevations, depth, and flooded state null, and emits no contours. It
does not substitute gage stage for a site-wide water-surface elevation or
invent depth contours. WDFN's latest daily gage-height value for that nearby
station is 2.67 ft from 1998-09-29, so it is historical evidence, not current
water level.

`vectorize_depth_thresholds(...)` is provided for later use when a validated
water-surface elevation and DEM share a known vertical datum. It treats water
surface as a constant plane; it is a terrain-screening calculation, not a
hydraulic flood model, and the current payload builder does not call it without
validated inputs.
