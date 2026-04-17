"""Shared video helpers used by the analyzer scripts.

Kept small and dependency-light. `ffmpeg` and `ffprobe` must be on PATH
(README already calls them out as setup requirements).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def codec_of(path: Path) -> str | None:
    """Return the video codec name as reported by ffprobe, or None if ffprobe
    is missing / the file has no video stream.

    Examples: 'h264', 'mpeg4', 'hevc'.
    """
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
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


def transcode_to_h264(path: Path, crf: int = 20, preset: str = "veryfast") -> bool:
    """Re-encode `path` in place to H.264 / yuv420p / +faststart.

    OpenCV's VideoWriter writes MPEG-4 Part 2 (fourcc `mp4v`) by default, which
    HTML5 `<video>` tags widely refuse. This helper normalises the output so
    browsers can play the annotated clip without a separate transcode.

    Returns True on success, False on any failure (ffmpeg missing, transcode
    non-zero exit, etc.). Idempotent — callers should check `codec_of()` first
    if they want to skip already-H.264 files.
    """
    if shutil.which("ffmpeg") is None:
        return False

    src = path
    tmp = path.with_suffix(path.suffix + ".h264.tmp.mp4")

    cmd = [
        "ffmpeg",
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
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(src)
    return True


def ensure_browser_playable(path: Path) -> str:
    """Make sure `path` is H.264 so browsers can play it. Idempotent.

    Returns one of:
      "already_h264"   file already uses H.264 — no action taken
      "transcoded"     successfully re-encoded in place
      "skipped"        ffmpeg/ffprobe missing; file left untouched
      "failed"         transcode was attempted but failed
    """
    current = codec_of(path)
    if current == "h264":
        return "already_h264"
    if current is None:
        # ffprobe missing. Try transcode anyway — ffmpeg might still be there
        # and will just re-encode blindly.
        pass

    ok = transcode_to_h264(path)
    if ok:
        return "transcoded"
    # If ffmpeg is missing we can't do anything useful.
    if shutil.which("ffmpeg") is None:
        return "skipped"
    return "failed"
