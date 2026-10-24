# FloodSight API Reference

Base URL (local/dev): `http://<host>:8000`. All responses are JSON
(`Content-Type: application/json`). There is no authentication, no versioning
and no CORS (the client is a native iOS app, which does not need it).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness / readiness |
| POST | `/flood` | Modeled flood depth + contours for a point and bbox |
| GET | `/waterbodies` | Mapped surface water in a bbox (GeoJSON) |

## `GET /health`

```json
{ "status": "ok", "groundsource_available": true }
```

`groundsource_available` is `false` if the historical-flood parquet is not
installed; the service still works but skips the historical-event adjustment
(see [Hosting](#hosting)).

## `POST /flood`

Request body (JSON, max 64,000 bytes):

```json
{
  "latitude": 29.6516,
  "longitude": -82.3248,
  "bbox_wgs84": [-82.3262, 29.6506, -82.3234, 29.6526],
  "rainfall_inches": 6
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `latitude` | number | yes | WGS84, -90..90 |
| `longitude` | number | yes | WGS84, -180..180 |
| `bbox_wgs84` | `[west, south, east, north]` | yes | Positive area, must contain the point, each side at most 7° |
| `rainfall_inches` | number ≥ 0 | one of | Total rainfall |
| `rainfall_intensity_inches_per_hour` + `rainfall_duration_hours` | numbers | one of | Multiplied to get total rainfall |
| `retained_water_fraction` | number 0..1 | no | Selects the legacy retained-rainfall model. Leave unset. |

Rain input is optional (defaults to 10 in). Do not send both rainfall forms.
Values must be real JSON numbers; strings and booleans are rejected.

Success (200):

```json
{
  "ar_flood_data": {
    "ground_elevation_msl_meters": 20.569,
    "water_surface_msl_meters": 20.569,
    "water_depth_meters": 0.0,
    "is_flooded": false,
    "elevation_datum": "North American Vertical Datum of 1988 (NAVD 88)"
  },
  "map_contours": [
    {
      "depth_meters": 0.1,
      "coordinates": [{ "lat": 29.6511, "lng": -82.3255 }, "... closed ring ..."]
    }
  ]
}
```

- `ar_flood_data` describes the requested coordinate only.
- Field names say "msl", but the real vertical reference is `elevation_datum`
  (NOAA VDatum converts NAVD 88 to local mean sea level where available;
  otherwise source-datum values are returned and labelled). Depth differences
  are unaffected.
- `water_depth_meters` is a modeled screening depth, not a gauge reading or
  validated forecast. `is_flooded` is derived from it.
- `map_contours`: polygons for depth thresholds 0.1, 0.2, 0.3 and 1.0 m, only
  where met. Each ring is closed, at most 15 points, clipped to the bbox.
  Polygons overlap (deeper areas also meet shallower thresholds). Existing
  water bodies are not included; use `/waterbodies`.

## `GET /waterbodies?bbox_wgs84=west,south,east,north`

Returns a GeoJSON `FeatureCollection`. Each feature has
`properties.category == "waterbody"` plus `name`, `source` and `source_id`.
This is mapped surface water, not flood depth or bathymetry.

## Errors

Every non-2xx response is JSON: `{ "error": "message" }`.

| Status | Meaning |
|---|---|
| 400 | Invalid JSON, missing field, wrong type, out-of-range value, bad bbox |
| 404 | Unknown path |
| 503 | A required upstream data service failed or timed out; retry later |
| 500 | Unexpected server error |

Clients should also handle network failures and timeouts.

## Performance

A cold `/flood` call fetches elevation, land cover, soil and water data from
public services, typically 3 to 8 s (large bboxes in dense areas such as Miami can
take longer). Repeat requests for the same area are cached in memory and
return in under a second. Set client timeouts to at least 30 s. Keep the bbox
small (a few hundred metres to ~2 km) for the fastest responses.

## Modeling caveats

Uses 3DEP terrain, 30 m NLCD land cover, sampled SSURGO soils and an SCS
curve-number runoff estimate with D8 catchment ponding. It does not model
storm drains, culverts, flow timing, velocity or validated street-level depths.
Historical Groundsource flood reports can raise modeled runoff by up to 15%
in chronically flooded areas. Present results as a scenario visualization with
uncertainty language, not a real-time warning or life-safety tool. Keep the
payload shape stable unless the app team approves a versioned change.

## Hosting

### Docker

```bash
cd backend
docker compose up -d --build
curl http://<host>:8000/health
```

The container runs `python service.py --host 0.0.0.0 --port 8000` as a non-root
user, has a health check, and restarts automatically. Port is configurable via
`FLOODSIGHT_PORT`.

### Groundsource data (optional)

Place the Groundsource parquet at
`backend/.cache/groundsource/groundsource_2026.parquet` before starting (the
directory is mounted at `/data/cache`, override with `FLOODSIGHT_CACHE_DIR`).
On first start the server builds a Florida index file beside it (about 15 s,
in the background); later starts reuse it. The cache directory must be
writable by uid 1000 (`chown -R 1000 .cache` on Linux). Without the parquet,
`/health` reports `groundsource_available: false` and the historical
adjustment is skipped.

### Letting other computers on the same network reach it

`docker-compose.yml` publishes port 8000 on all host interfaces, so any device
on the same network can call `http://<host-LAN-IP>:8000`. Find the IP with
`ipconfig` (Windows) or `ip addr` (Linux). If the host OS firewall blocks it,
allow the port, for example (Windows, admin PowerShell):
`New-NetFirewallRule -DisplayName FloodSight -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`.
Some networks (campus Wi-Fi such as eduroam, guest Wi-Fi) isolate clients from
each other; use a personal hotspot or a tunnel (Tailscale, ngrok) there.

### Reaching it from the iOS app

- Simulator on the same machine: `http://localhost:8000`.
- Physical device / remote host: use the host's LAN IP or public DNS name.
- iOS blocks plain `http://` by default (App Transport Security). For
  production put the container behind an HTTPS reverse proxy (Caddy, nginx,
  a cloud load balancer) and use `https://`; for local dev add an ATS
  exception in the app.
- The service has no authentication or rate limiting. Do not expose it
  publicly without a proxy that adds HTTPS and limits.
