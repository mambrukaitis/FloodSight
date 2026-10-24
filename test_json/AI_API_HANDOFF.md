# FloodSight API integration handoff

## Task for the receiving AI

Integrate the existing FloodSight Python HTTP service with the app without
changing its request or response contract. Preserve the current JSON key names,
units, and separate waterbody endpoint. Do not present screening estimates as
observed or safety-grade flood predictions. Confirm the app's actual network
environment before selecting a host; the service currently runs locally and
has no cloud deployment configuration.

## How it is hosted today

The service is implemented in `test_json/service.py` using Python's
`ThreadingHTTPServer`. From the FloodSight project root:

```powershell
.\.venv\Scripts\python.exe -m test_json.service --serve
```

By default it listens on `127.0.0.1:8000`. This is a development/local
endpoint, not a public deployment. For a phone or simulator to reach it,
loopback may not refer to the developer computer; determine the development
setup's reachable host address and configure the app base URL accordingly.
Binding to `0.0.0.0` exposes the unauthenticated, unencrypted service to the
network and must only be used on a trusted development network. Do not deploy
it publicly without adding production hosting, HTTPS, authentication,
request/rate limits, timeouts, and upstream-data/cache management.

The default service also expects
`.cache/groundsource/groundsource_2026.parquet` relative to the process working
directory. The output generators and API call several external data services,
so an endpoint response may take time and may fail when an upstream service or
required local cache is unavailable.

Independent source requests run concurrently. Static spatial responses and
short-lived USGS context are cached in-process; Groundsource has a separate
bbox-keyed disk cache. `build_flood_payload` diagnostics include per-source
timings and cache status, but `/flood` intentionally returns only the
established iOS payload. A cold Lake Alice request was measured at about
4.3 seconds in one live run; cached repeat requests can be substantially
faster, but external-service latency means this is not a guaranteed SLA.
3DEP vertical-datum metadata is best-effort with a short timeout; if unavailable,
the returned datum is explicitly unknown rather than guessed. This affects
absolute elevation labeling, not depth differences.

## Flood endpoint

`POST /flood` accepts a JSON object:

```json
{
  "latitude": 29.64289215,
  "longitude": -82.36122382833753,
  "bbox_wgs84": [-82.3625, 29.6417, -82.3599, 29.6441],
  "rainfall_inches": 10
}
```

Required inputs:

- `latitude`: finite WGS84 decimal degrees, between -90 and 90.
- `longitude`: finite WGS84 decimal degrees, between -180 and 180.
- `bbox_wgs84`: `[west, south, east, north]` in WGS84 decimal degrees. It must
  be ordered, have positive area, contain the requested latitude/longitude,
  and each dimension must be no more than 7 degrees.

Rain input is optional; the service demo default is 10 inches. Supply exactly
one of:

- `rainfall_inches`: non-negative total rainfall depth in inches; or
- both `rainfall_intensity_inches_per_hour` and `rainfall_duration_hours`.
  The service multiplies these values to obtain total rainfall in inches.

Do not send both rainfall forms. `retained_water_fraction` is optional and
defaults to `null`, selecting the SCS Curve Number + D8 catchment screening
model. Passing a number from 0 through 1 explicitly selects the legacy
retained-rainfall model; do not set it unless that behavior is intended.

### Response contract

A successful response is the app payload only. Its top-level keys remain:

```json
{
  "ar_flood_data": {
    "ground_elevation_msl_meters": 20.569,
    "water_surface_msl_meters": 20.569,
    "water_depth_meters": 0.0,
    "is_flooded": false,
    "elevation_datum": "North American Vertical Datum of 1988 (NAVD 88)"
  },
  "map_contours": []
}
```

Interpretation:

- `ar_flood_data` describes the requested coordinate, not every location in
  the bounding box.
- `ground_elevation_msl_meters` and `water_surface_msl_meters` are numeric
  meters. Despite the legacy field names, use `elevation_datum` to determine
  their actual vertical reference. NOAA VDatum converts NAVD 88 to LMSL where
  it has coverage. When it has no conversion, the service preserves the
  source-datum values and reports that datum (Lake Alice is NAVD 88). Do not
  describe a non-LMSL value as mean sea level. `elevation_datum` is additive
  metadata inside `ar_flood_data`; it clarifies the legacy MSL field names.
- `water_depth_meters` is the modeled depth at the requested coordinate in
  meters; it is not a stage measurement or validated forecast.
- `is_flooded` is derived from the point depth. It does not indicate whether
  other parts of the bbox are flooded.
- `map_contours` is an array of threshold polygons. Each object contains
  `depth_meters` (the minimum modeled depth represented by that zone) and
  `coordinates`, an ordered ring of `{ "lat": number, "lng": number }` points.
  Rings are closed, simplified to at most 15 coordinates including closure,
  and clipped to the requested bbox. Thresholds currently are 0.1, 0.2, 0.3,
  and 1.0 m when areas meet them; threshold polygons can overlap because
  deeper areas also meet shallower thresholds.
- Mapped lakes and other existing water features are not zero-depth contour
  polygons and are not represented in this response.

The API currently returns HTTP 400 with an `error` object for invalid requests
and HTTP 503 with an `error` object for unavailable required services/data.
Clients should handle non-2xx responses and malformed/network failures without
assuming a successful-shaped payload. Model diagnostics are produced by the
Python function but are not included in `/flood` responses.

## Waterbody endpoint

`GET /waterbodies?bbox_wgs84=west,south,east,north` returns a GeoJSON
`FeatureCollection`. Mapped features have `properties.category == "waterbody"`
and separate source/name metadata. Treat these as mapped surface water, not
flood depth or bathymetry.

## Modeling and display constraints

The service uses 3DEP terrain, 30 m NLCD land cover, sampled SSURGO soils, and
an SCS runoff estimate. Coarse land-cover classes and sparse soil groups are
interpolated to the finer terrain grid; finer rendering is not finer observed
data. Missing soil data uses hydrologic group C as an explicit middle-category
fallback. D8 routing and static per-sink catchment ponding are screening
approximations; they do not model storm-drain networks, culverts, flow timing,
velocity, downstream routing of basin overflow, or validated street-level
depths. Groundsource
event footprints and USGS observations are contextual and do not train or
calibrate point depths.

Display the output as a scenario visualization with appropriate uncertainty
language, not as a real-time warning or a life-safety decision tool. Keep the
existing payload shape intact unless the app team explicitly approves a
versioned API change.
