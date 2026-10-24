import time
from service import build_flood_payload

# Test coordinates (Gainesville or Miami)
lat, lon = 29.6516, -82.3248
bbox = (-82.33, 29.64, -82.32, 29.66)

print("Starting flood simulation...")
start_time = time.perf_counter()

payload, diagnostics = build_flood_payload(lat, lon, bbox, rainfall_inches=6.0)

end_time = time.perf_counter()
# Calculate elapsed time directly in seconds
elapsed_seconds = end_time - start_time

print(f"\n--- EXECUTION TIME: {elapsed_seconds:.2f} s ---")
print(f"Is Flooded: {payload['ar_flood_data']['is_flooded']}")
print(f"Historical Groundsource Events: {diagnostics['model']['historical_groundsource_events']}")