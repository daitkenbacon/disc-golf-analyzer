#!/usr/bin/env python3
"""
Extract N evenly-spaced keyframes from a video as PNGs.

Usage:
    python scripts/extract_keyframes.py clips/my_throw.mp4 [--n 8] [--out-dir keyframes/<stem>]

Used as a fallback for visual-only review when the full pose pipeline hasn't
been run yet, or when a clip won't open with MediaPipe. Requires only cv2
(already in requirements.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--n", type=int, default=8, help="number of keyframes to extract")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    clip: Path = args.clip
    if not clip.exists():
        print(f"clip not found: {clip}", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir or (ROOT / "keyframes" / clip.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        print(
            f"cv2 could not open {clip}. If HEVC/.MOV from iPhone, transcode with:\n"
            f"  ffmpeg -i {clip.name} -c:v libx264 -crf 18 -preset veryfast -c:a aac {clip.stem}.mp4",
            file=sys.stderr,
        )
        return 3

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print("could not determine frame count", file=sys.stderr)
        return 4

    indices = [int(round(i * (frame_count - 1) / (args.n - 1))) for i in range(args.n)]
    written = 0
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        path = out_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(path), img)
        written += 1
        print(f"wrote {path}")

    cap.release()
    print(f"\nextracted {written} keyframes to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
