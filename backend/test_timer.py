"""Times `build_flood_payload` (no HTTP server needed) at several Florida locations.

Run from the backend directory:  python test_timer.py
Each location is run twice: the first call is "cold" (downloads data from the
public services), the second is "warm" (served from the in-process cache).
"""

import time

from floodsight.flood_payload import build_flood_payload

# name: (latitude, longitude, half_side_degrees)
LOCATIONS = {
    "Miami (Big Box)": (25.7617, -80.1918, 0.03),  # 0.06 degree span (~6.5 km across)
}
RAINFALL_INCHES = 6.0


def main() -> None:
    for name, (lat, lon, half_side) in LOCATIONS.items():
        bbox = (
            lon - half_side,
            lat - half_side,
            lon + half_side,
            lat + half_side,
        )
        for label in ("cold", "warm"):
            started = time.perf_counter()
            payload, diagnostics = build_flood_payload(
                lat, lon, bbox, rainfall_inches=RAINFALL_INCHES
            )
            elapsed = time.perf_counter() - started
            model = diagnostics["model"]
            
            max_depth = model.get("maximum_grid_depth_meters", 0.0)
            
            print(
                f"{name:<20} {label}: {elapsed:5.2f} s | "
                f"flooded={payload['ar_flood_data']['is_flooded']} | "
                f"max depth={max_depth:.2f}m | "
                f"events={model['historical_groundsource_events']} | "
                f"multiplier={model['backend_urban_drainage_calibration_multiplier']:.3f}"
            )


if __name__ == "__main__":
    main()