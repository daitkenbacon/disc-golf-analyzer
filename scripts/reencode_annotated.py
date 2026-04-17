#!/usr/bin/env python3
"""Re-encode existing annotated/*.mp4 files to H.264 for browser playback.

The analyzer pipelines historically wrote MPEG-4 Part 2 (fourcc `mp4v`)
which HTML5 `<video>` tags widely refuse. Newer runs of
`analyze_throw_cv.py` / `analyze_throw.py` post-process to H.264
automatically; this script is for clips that were analyzed before that
change landed.

Idempotent: files already reported as `h264` by ffprobe are skipped.

Usage:
    python scripts/reencode_annotated.py              # re-encode all
    python scripts/reencode_annotated.py throw-1.cv.mp4 other.pose.mp4
    python scripts/reencode_annotated.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _video_utils import codec_of, ensure_browser_playable  # noqa: E402


ROOT = HERE.parent
ANNOTATED_DIR = ROOT / "annotated"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "files",
        nargs="*",
        help="specific filenames inside annotated/ to re-encode "
             "(default: every *.mp4 in the directory)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would happen, but don't touch any files",
    )
    args = ap.parse_args()

    if not ANNOTATED_DIR.exists():
        print(f"no annotated/ directory at {ANNOTATED_DIR}")
        return 0

    if args.files:
        targets = [ANNOTATED_DIR / f for f in args.files]
        missing = [t for t in targets if not t.exists()]
        if missing:
            for m in missing:
                print(f"[skip] not found: {m.name}")
        targets = [t for t in targets if t.exists()]
    else:
        targets = sorted(ANNOTATED_DIR.glob("*.mp4"))

    if not targets:
        print("no annotated/*.mp4 files to process")
        return 0

    print(f"scanning {len(targets)} file(s) in {ANNOTATED_DIR}")
    already = transcoded = failed = skipped = 0

    for path in targets:
        codec = codec_of(path)
        label = codec or "unknown"
        if codec == "h264":
            print(f"  [ok]    {path.name}  ({label})")
            already += 1
            continue

        if args.dry_run:
            print(f"  [would] {path.name}  ({label} → h264)")
            continue

        print(f"  [...]   {path.name}  ({label} → h264)", end="", flush=True)
        result = ensure_browser_playable(path)
        if result == "transcoded":
            print(" ✓")
            transcoded += 1
        elif result == "already_h264":
            # Race — someone else transcoded while we were looking.
            print(" (already h264)")
            already += 1
        elif result == "skipped":
            print(" [skipped — ffmpeg not found on PATH]")
            skipped += 1
        else:
            print(" [FAILED]")
            failed += 1

    print()
    print(f"summary: {already} already-h264, {transcoded} transcoded, "
          f"{failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
