import time
import service
from service import build_flood_payload, bbox_around_point

STAGES = [
    "_fetch_dem", "_dem_vertical_datum", "run_usgs", "_latest_daily_stage",
    "_fetch_waterbodies", "_fetch_nlcd_landcover",
    "_fetch_ssurgo_hydrologic_soil_grid", "_solve_d8_catchment_ponding",
    "vectorize_depth_thresholds",
]
timings: dict[str, float] = {}


def wrap(name):
    fn = getattr(service, name)

    def inner(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            timings[name] = timings.get(name, 0) + time.perf_counter() - t

    setattr(service, name, inner)


for s in STAGES:
    wrap(s)

LOCATIONS = {
    "Gainesville": (29.6516, -82.3248),
    "Miami": (25.7617, -80.1918),
    "Tampa": (27.9506, -82.4572),
    "Jacksonville": (30.3322, -81.6557),
    "Tallahassee": (30.4383, -84.2807),
}
for city, (lat, lon) in LOCATIONS.items():
    timings.clear()
    bbox = bbox_around_point(lat, lon)
    t = time.perf_counter()
    try:
        build_flood_payload(lat, lon, bbox, rainfall_inches=6.0)
        status = "ok"
    except Exception as e:
        status = f"ERR {e!r}"[:80]
    total = time.perf_counter() - t
    print(f"\n{city}: {total:.1f}s {status}")
    for k, v in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"   {k:40s} {v:6.2f}s")
