"""Serve the current directory over HTTP with CORS enabled, for Label Studio image URLs.

    cd data/processed/punch-crops && uv run python -m boxing.serve_cors [--port 8000] [--bind 127.0.0.1]

Serves the directory it is run from. Task URLs in ``label_studio_tasks_http.json`` are relative
to the export folder, so run it from that folder (the ``DST`` given to ``boxing.person_crops``).
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # preflight requests
        self.send_response(204)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1", help="address to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    root = Path.cwd()
    handler = partial(CORSRequestHandler, directory=str(root))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f"Serving {root} with CORS at http://{args.bind}:{args.port}/ (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
