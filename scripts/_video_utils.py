"""Shared video helpers used by the analyzer scripts.

Kept small and dependency-light. `ffmpeg` and `ffprobe` must be installed
(README calls them out as setup requirements) but we don't require them to
be on PATH — when Flask or a subprocess inherits a minimal environment
(GUI-launched apps, CI, etc.) PATH often doesn't include Homebrew. We fall
back to probing the usual Homebrew + system install locations so the
pipeline's auto-transcode still fires in those cases.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Homebrew + system paths where ffmpeg/ffprobe commonly live when PATH doesn't
# include them (GUI-launched Python, launchd services, fresh shells without
# ~/.zshrc sourced). Checked in order; first hit wins.
_CANDIDATE_DIRS = (
    "/opt/homebrew/bin",   # Apple Silicon Homebrew
    "/usr/local/bin",      # Intel Mac Homebrew, also many Linux installs
    "/usr/bin",            # Debian/Ubuntu/etc.
    "/opt/local/bin",      # MacPorts
)


def _resolve(bin_name: str) -> str | None:
    """Find a binary by checking PATH first, then a handful of known install
    locations. Returns the full path, or None if nothing matches."""
    hit = shutil.which(bin_name)
    if hit:
        return hit
    for d in _CANDIDATE_DIRS:
        candidate = Path(d) / bin_name
        if candidate.is_file():
            return str(candidate)
    return None


def ffmpeg_bin() -> str | None:
    """Return the absolute path to ffmpeg, or None if not installed."""
    return _resolve("ffmpeg")


def ffprobe_bin() -> str | None:
    """Return the absolute path to ffprobe, or None if not installed."""
    return _resolve("ffprobe")


def codec_of(path: Path) -> str | None:
    """Return the video codec name as reported by ffprobe, or None if ffprobe
    is missing / the file has no video stream.

    Examples: 'h264', 'mpeg4', 'hevc'.
    """
    ffprobe = ffprobe_bin()
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def transcode_to_h264(
    path: Path,
    crf: int = 20,
    preset: str = "veryfast",
) -> tuple[bool, str | None]:
    """Re-encode `path` in place to H.264 / yuv420p / +faststart.

    OpenCV's VideoWriter writes MPEG-4 Part 2 (fourcc `mp4v`) by default, which
    HTML5 `<video>` tags widely refuse. This helper normalises the output so
    browsers can play the annotated clip without a separate transcode.

    Returns (ok, detail). On success: (True, None). On failure: (False, msg)
    where `msg` is a short human-readable reason (used by the pipeline logs
    so the user can tell whether ffmpeg was missing vs. errored).
    Idempotent — callers should check `codec_of()` first if they want to skip
    already-H.264 files.
    """
    ffmpeg = ffmpeg_bin()
    if ffmpeg is None:
        return False, "ffmpeg not found (install via `brew install ffmpeg`)"

    src = path
    tmp = path.with_suffix(path.suffix + ".h264.tmp.mp4")

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",  # annotated video has no audio anyway
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        # ffmpeg writes the useful diagnostic to stderr; surface the last line
        stderr = (e.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else f"exit {e.returncode}"
        return False, f"ffmpeg failed: {detail}"
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, "ffmpeg timed out (>600s)"
    except FileNotFoundError:
        tmp.unlink(missing_ok=True)
        return False, f"ffmpeg binary vanished at {ffmpeg}"

    tmp.replace(src)
    return True, None


def ensure_browser_playable(path: Path) -> tuple[str, str | None]:
    """Make sure `path` is H.264 so browsers can play it. Idempotent.

    Returns (status, detail). status is one of:
      "already_h264"   file already uses H.264 — no action taken
      "transcoded"     successfully re-encoded in place
      "skipped"        ffmpeg missing entirely; file left untouched
      "failed"         transcode was attempted but failed
    detail carries a short human-readable reason when status is "skipped"
    or "failed"; None otherwise.
    """
    current = codec_of(path)
    if current == "h264":
        return "already_h264", None

    ok, detail = transcode_to_h264(path)
    if ok:
        return "transcoded", None
    if ffmpeg_bin() is None:
        return "skipped", detail or "ffmpeg not found"
    return "failed", detail
