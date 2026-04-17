#!/usr/bin/env python3
"""Launch the local Disc Golf Analyzer web UI.

Usage:
    python scripts/serve.py [--port 8765] [--no-browser]

Opens http://localhost:<port>/ in your default browser, then runs a Flask
development server. Localhost only — nothing is exposed to the network.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Make the project root importable so `from web.app import app` works when
# this file is run directly (python scripts/serve.py).
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.app import app  # noqa: E402


def open_browser_soon(url: str, delay_s: float = 0.8) -> None:
    def _open() -> None:
        time.sleep(delay_s)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="TCP port (default: 8765)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Leave at 127.0.0.1 unless you know what you're doing.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open a browser tab on start.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flask debug mode (auto-reload on code changes).",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    print(f"Disc Golf Analyzer — serving at {url}")
    print("Press Ctrl-C to stop.")

    if not args.no_browser:
        open_browser_soon(url)

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()
