"""Times `build_flood_payload` (no HTTP server needed) at several Florida locations.

Run from the backend directory:  python time_flood_estimate.py
Each location is run twice: the first call is "cold" (downloads data from the
public services), the second is "warm" (served from the in-process cache).
"""

import time

from floodsight.flood_payload import build_flood_payload

# name: (latitude, longitude); each uses a ~2 km box around the point.
LOCATIONS = {
    "Gainesville": (29.6516, -82.3248),
    "Miami": (25.7617, -80.1918),
    "Tampa": (27.9506, -82.4572),
}
HALF_SIDE_DEGREES = 0.01
RAINFALL_INCHES = 6.0


def main() -> None:
    for name, (lat, lon) in LOCATIONS.items():
        bbox = (
            lon - HALF_SIDE_DEGREES,
            lat - HALF_SIDE_DEGREES,
            lon + HALF_SIDE_DEGREES,
            lat + HALF_SIDE_DEGREES,
        )
        for label in ("cold", "warm"):
            started = time.perf_counter()
            payload, diagnostics = build_flood_payload(
                lat, lon, bbox, rainfall_inches=RAINFALL_INCHES
            )
            elapsed = time.perf_counter() - started
            model = diagnostics["model"]
            print(
                f"{name:<12} {label}: {elapsed:5.2f} s | "
                f"flooded={payload['ar_flood_data']['is_flooded']} | "
                f"groundsource events={model['historical_groundsource_events']} | "
                f"multiplier={model['backend_urban_drainage_calibration_multiplier']}"
            )


if __name__ == "__main__":
    main()
