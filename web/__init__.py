"""Local web UI for the Disc Golf Analyzer.

Launch with `python scripts/serve.py`.

This package is optional — the CLI pipeline and coaching flows work without it.
Only imported by the launcher.
"""

from web.app import app, create_app  # noqa: F401

__all__ = ["app", "create_app"]
