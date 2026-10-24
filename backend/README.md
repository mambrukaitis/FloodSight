# FloodSight Backend

Python service that turns a location, bounding box and rainfall amount into a
modeled flood-depth payload (point depth plus depth-contour polygons) for the
iOS app.

| File | Purpose |
|---|---|
| `floodsight/http_api.py`, `server_startup.py` | HTTP endpoints (`/flood`, `/waterbodies`, `/health`) and CLI |
| `floodsight/flood_payload.py` | `build_flood_payload`: orchestrates one estimate |
| `floodsight/rainfall_inputs.py` | Rainfall / retained-fraction input resolution |
| `floodsight/source_fetching.py` | Parallel fetch of all datasets |
| `floodsight/flood_depth_model.py` | SCS+D8 and legacy depth models, Groundsource drainage multiplier |
| `floodsight/scs_runoff.py`, `terrain_ponding_solver.py` | Curve numbers, runoff, D8 ponding |
| `floodsight/grid_geometry.py`, `contour_polygons.py` | Grid/CRS helpers, depth-contour polygons |
| `floodsight/*_source.py`, `groundsource_history.py` | DEM, NLCD, SSURGO, waterbody, VDatum and Groundsource data access |
| `floodsight/request_caching.py` | TTL cache for upstream requests |
| `tests/` | Unit tests, one file per topic |
| `time_flood_estimate.py` | Times live `/flood` calls (cold and warm) at Florida cities |
| `Dockerfile`, `docker-compose.yml` | Container hosting |
| `API.md` | Endpoint reference for frontend integration |

## Run locally

Python 3.11+:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m floodsight --port 8000
```

## Run with Docker

```bash
docker compose up -d --build        # serves on http://localhost:8000
curl http://localhost:8000/health
docker compose logs -f
docker compose down
```

Change the host port with `FLOODSIGHT_PORT=9000 docker compose up -d`.
See [API.md](API.md) for hosting notes (cache volume, security, HTTPS).

## Tests

```powershell
python -m unittest discover -s tests -t . -v
```
