"""Process entry point: parses --host/--port, warms the Groundsource index in a
background thread, and runs the threaded HTTP server.
"""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import threading

from floodsight.groundsource_history import (
    GROUND_SOURCE_PATH,
    groundsource_florida_events,
)
from floodsight.http_api import create_request_handler


def warm_up() -> None:
    """Pre-load the Groundsource index so the first real request is not slow."""
    if GROUND_SOURCE_PATH.is_file():
        groundsource_florida_events(GROUND_SOURCE_PATH)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the threaded HTTP server and block until interrupted; warms the
    Groundsource index in the background.
    """
    server = ThreadingHTTPServer((host, port), create_request_handler())
    threading.Thread(target=warm_up, daemon=True).start()
    print(f"FloodSight service listening at http://{host}:{port} (POST /flood, GET /waterbodies, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping FloodSight service.")
    finally:
        server.server_close()


def main() -> None:
    """Parse --host/--port and start the server."""
    parser = argparse.ArgumentParser(
        description="Run the FloodSight Florida-wide flood JSON microservice."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.host, args.port)
