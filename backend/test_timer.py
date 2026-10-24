import time
from service import build_flood_payload

# Miami urban test coordinate (historically high Groundsource event density)
lat, lon = 25.7617, -80.1918
bbox = (-80.20, 25.75, -80.18, 25.77)

print("Starting flood simulation for Miami test case...")
start_time = time.perf_counter()

payload, diagnostics = build_flood_payload(lat, lon, bbox, rainfall_inches=6.0)

end_time = time.perf_counter()
elapsed_seconds = end_time - start_time

print(f"\n--- EXECUTION TIME: {elapsed_seconds:.2f} s ---")
print(f"Is Flooded: {payload['ar_flood_data']['is_flooded']}")
print(f"Historical Groundsource Events: {diagnostics['model']['historical_groundsource_events']}")
print(f"Backend Drainage Multiplier: {diagnostics['model']['backend_urban_drainage_calibration_multiplier']}")