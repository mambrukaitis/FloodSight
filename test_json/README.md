# Reitz Union flood JSON demo

Run from the FloodSight workspace root:

```powershell
.\.venv\Scripts\python.exe -m test_json.service
```

Override the 10-inch demo rainfall assumption. The standard model derives
rainfall excess from land cover and soils; an explicit retained-water fraction
selects the legacy demonstration model instead:

```powershell
.\.venv\Scripts\python.exe -m test_json.service --rainfall-inches 10 --retained-water-fraction 0.02
```

Generate the same scenario at the Lake Alice mapped-water representative point:

```powershell
.\.venv\Scripts\python.exe -m test_json.service --location lake-alice
```

To run the HTTP microservice locally:

```powershell
.\.venv\Scripts\python.exe -m test_json.service --serve
```

Send a POST to `http://127.0.0.1:8000/flood` with JSON containing
`latitude`, `longitude`, and `bbox_wgs84: [west, south, east, north]`. Rain can
be supplied as `rainfall_inches`, or as both
`rainfall_intensity_inches_per_hour` and `rainfall_duration_hours` (for example,
5.4 in/hr for 0.5 hours gives 2.7 inches). Do not supply both forms.
`retained_water_fraction` is optional and defaults to `null`, which selects the
SCS curve-number model. Supplying a number from 0 to 1 explicitly selects the
legacy retained-rainfall scenario. If rain input is omitted, the example
default is 10 inches. A valid request returns only the iOS payload; malformed
requests return HTTP 400 and failed required upstream/DEM requests return HTTP
503 with an `error` field. `GET /waterbodies?bbox_wgs84=west,south,east,north`
returns mapped water features as GeoJSON, separately from flood-depth contours.
The server is a local Python `ThreadingHTTPServer`, not a deployed cloud
service; it binds to `127.0.0.1:8000` by default. See
[`AI_API_HANDOFF.md`](./AI_API_HANDOFF.md) for the endpoint contract and an
integration task prompt.

The service function is `build_flood_payload(latitude, longitude, bbox_wgs84,
rainfall_inches=None, rainfall_intensity_inches_per_hour=None,
rainfall_duration_hours=None, retained_water_fraction=None)`.
The bounding box is `(west, south, east, north)` in EPSG:4326. The default
example uses the existing Reitz Union Drive coordinate; `--location lake-alice`
uses a representative point inside the OpenStreetMap Lake Alice polygon
(relation 1630473). Reitz uses a 100 m by 100 m box; Lake Alice uses a
250 m by 250 m neighborhood so the map includes surrounding land, not only the
lake. The model fetches an additional 250 m terrain context around the requested
map area, routes runoff within that wider domain, then clips contours back to
the requested box. Contour zones are emitted at 0.1, 0.2, 0.3, and 1.0 m
where present.
The Lake Alice representative point remains on the lake, so its point-depth
field is zero; inspect the surrounding contours for modeled land flooding.
Requests can cover larger regions up to 7 degrees per dimension; the 3DEP grid
is capped at 50,000 pixels, so larger boxes are sampled more coarsely. The
service requests USGS 3DEP elevation, nearby USGS WDFN observations, cached
Groundsource event footprints, and mapped water-body polygons from
OpenStreetMap for small boxes or USGS NHD at broader scales.

The command writes `reitz_union_flood.json` with the exact two top-level keys
expected by the Swift model, `reitz_union_diagnostics.json` separately, and
`reitz_union_waterbodies.geojson`. The Lake Alice option writes corresponding
`lake_alice_*` files. Diagnostics record the model inputs and source status.

Run the service tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest test_json.test_service -v
```

## Interpretation and limitations

The payload is numeric and uses the exact two top-level keys required by the
Swift shape. Its values are a **screening estimate**, not observed water levels
or a calibrated forecast:

- 3DEP supplies the elevation raster. NOAA VDatum is asked to convert NAVD 88
  heights to local mean sea level (LMSL). If VDatum has no transformation at
  the location, the numeric fields remain in the source DEM datum and
  `ar_flood_data.elevation_datum` explicitly identifies that datum. For
  example, NOAA VDatum returned its no-coverage sentinel at Lake Alice, so the
  generated values there remain NAVD 88 rather than being falsely labeled as
  LMSL. Depths are relative differences in one source datum and are unaffected
  by a constant vertical offset. `elevation_datum` is additive metadata inside
  `ar_flood_data`; the two top-level response keys are unchanged.
- The default runoff estimate uses the NRCS SCS curve-number rainfall-excess
  equation, with 2021 MRLC NLCD land cover and spatially sampled USDA SSURGO
  hydrologic soil groups. The MRLC WMS GeoTIFF is palette-indexed, so palette
  colors are decoded to NLCD class codes; unrecognized palette values cause a
  visible fallback and diagnostic note rather than being treated as raw class
  numbers.
- The estimated runoff is assigned to a D8 steepest-descent terrain network.
  Interior sink catchments receive static volume-balanced ponding, and flow
  reaching the context-domain boundary is counted as outflow. This improves on
  one pooled water surface across the entire box, but it is still not a
  hydraulic simulation: flow is not time-dependent, each cell chooses one
  downstream neighbor, flats use a deterministic tie-break, and catchment
  overflow is counted as outflow rather than routed downstream. Lake exchange,
  storm drains, culverts, road barriers, and external flows are not simulated.
  Generalized curve numbers and basin depths are not
  calibrated to local observed flood depths.
- Supplying `retained_water_fraction` opts into the legacy assumption-based
  alternative. For example, 0.02 with 10 inches of rain retains an equivalent
  0.2 inches over land cells before closed-box redistribution. It is not an
  empirically measured coefficient.
- Missing SSURGO coverage uses hydrologic soil group C, a middle-category
  average assumption, and known 4x4 soil samples are bilinearly interpolated.
  It is an explicit estimate, not a measured soil property. NLCD class
  fractions from 30 m cells are bilinearly averaged to the analysis grid;
  this smooths coarse categories but does not create sub-30 m observations.
- WDFN and Groundsource do not numerically calibrate the model. Gage
  observations and recent historical event footprints are reported as context;
  Groundsource has no pointwise depth labels. NLCD and SSURGO are model inputs,
  not flood-depth training labels.
- USGS WDFN v1 is queried for nearby stations and gage heights. At Reitz, the
  closest streamgage has no current stage reading; its latest daily value is
  historical. A gage-stage observation is not assigned across the bounding
  box because no gage-datum/elevation tie or hydraulic connection is available.
- Groundsource is scanned for local historical event footprints. It contains no
  point depth values, so event overlap is diagnostic context, not calibration.
- Mapped water-body polygons are separate typed GeoJSON features with
  `category: "waterbody"`; they are not zero-depth entries in `map_contours`
  and do not represent bathymetry.
- Contour rings are simplified to at most 15 coordinate dictionaries,
  including the repeated closing coordinate. Holes are omitted by the payload
  schema.
- FEMA NFHL zones, NOAA storm-surge products, stream stage, and Groundsource
  footprints are not converted into local rainfall depths. Their hazard types
  differ; surge needs a coastal boundary condition and compatible vertical
  datum, while FEMA zones and Groundsource footprints are not pointwise depth
  observations.
- Intensity and duration are converted to total rainfall; the simplified model
  does not simulate a time-varying hydrograph. Infiltration, storm-drain
  capacity, boundary flow, and runoff from outside the box are not modeled.

Use the output for a demo visualization only, not safety decisions. A
scientifically defensible prediction needs event rainfall timing, local
hydraulic/drainage data, and observed depths for calibration. The output schema
has no confidence/provenance fields, so the returned JSON itself cannot convey
these caveats; see the generated diagnostics JSON and this README.
