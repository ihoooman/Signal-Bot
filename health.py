"""Minimal HTTP health server for Render deployments."""

import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


def run() -> None:
    """Start an HTTP server bound to the provided PORT environment variable."""
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    server.serve_forever()


if __name__ == "__main__":  # pragma: no cover - manual execution
    run()
