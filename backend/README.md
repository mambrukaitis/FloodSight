# FloodSight Backend

Python service that turns a location, bounding box and rainfall amount into a
modeled flood-depth payload (point depth plus depth-contour polygons) for the
iOS app.

| File | Purpose |
|---|---|
| `service.py` | `build_flood_payload` and the HTTP API (`/flood`, `/waterbodies`, `/health`) |
| `data_sources.py` | Remote data (3DEP, NLCD, SSURGO, NHD/OSM, NOAA VDatum), caches, Groundsource index |
| `hydrology.py` | Geometry helpers, SCS runoff + D8 ponding model, contour vectorization |
| `tests/test_service.py` | Unit tests |
| `test_timer.py` | Times a live Miami `/flood` call (needs the server running) |
| `Dockerfile`, `docker-compose.yml` | Container hosting |
| `API.md` | Endpoint reference for frontend integration |

## Run locally

Python 3.11+:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python service.py --port 8000
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
$env:PYTHONPATH="."; python -m unittest tests.test_service -v
```
