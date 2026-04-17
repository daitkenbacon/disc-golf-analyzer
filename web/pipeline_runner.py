"""Subprocess wrapper around the analyzer scripts.

Keeps the web layer decoupled from the pipeline implementation: the scripts
are called as processes, not imported. That way the pipeline owns its own
state, logging, and module-level side effects (MediaPipe init, etc.) without
polluting the Flask process.

Usage:
    for line in run_pipeline("cv2", clip_path, overrides={"plant": 230}):
        yield line  # stream to client
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_CV2 = PROJECT_ROOT / "scripts" / "analyze_throw_cv.py"
SCRIPT_POSE = PROJECT_ROOT / "scripts" / "analyze_throw.py"

OVERRIDE_FLAGS = {
    "setup": "--setup",
    "plant": "--plant",
    "reach_back": "--reach-back",
    "power_pocket": "--power-pocket",
    "hit": "--hit",
    "release": "--release",
}

# Which override flags each pipeline understands.
PIPELINE_FLAGS = {
    "cv2": set(OVERRIDE_FLAGS),
    "pose": {"plant", "reach_back", "power_pocket", "hit", "release"},  # no --setup
}


def script_for(pipeline: str) -> Path:
    if pipeline == "cv2":
        return SCRIPT_CV2
    if pipeline == "pose":
        return SCRIPT_POSE
    raise ValueError(f"unknown pipeline: {pipeline!r}")


def build_command(
    pipeline: str,
    clip_path: Path,
    overrides: dict[str, int] | None = None,
    notes: str = "",
) -> list[str]:
    """Build the argv for a single pipeline run."""
    script = script_for(pipeline)
    if not script.exists():
        raise FileNotFoundError(f"pipeline script missing: {script}")

    cmd = [sys.executable, str(script), str(clip_path)]
    if notes:
        cmd += ["--notes", notes]

    if overrides:
        allowed = PIPELINE_FLAGS[pipeline]
        for key, value in overrides.items():
            if value is None or key not in allowed:
                continue
            cmd += [OVERRIDE_FLAGS[key], str(int(value))]

    return cmd


def run_pipeline(
    pipeline: str,
    clip_path: Path,
    overrides: dict[str, int] | None = None,
    notes: str = "",
) -> Iterator[str]:
    """Run a pipeline and yield stdout lines as they arrive.

    stderr is merged into stdout so MediaPipe's chatty GL-context init and
    the pipeline's own `[N/7]` progress lines land in the same stream that
    the web client consumes.
    """
    cmd = build_command(pipeline, clip_path, overrides=overrides, notes=notes)
    yield f"$ {' '.join(cmd)}\n"

    env = os.environ.copy()
    # Force unbuffered stdout so progress lines land in real time.
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None

    for line in proc.stdout:
        yield line

    rc = proc.wait()
    if rc != 0:
        yield f"\n[error] {pipeline} exited with code {rc}\n"
    else:
        yield f"\n[done] {pipeline} finished cleanly\n"
