#!/usr/bin/env python3
"""
Disc Golf Tee Shot Analyzer — cv2-only fallback
================================================

Used when MediaPipe isn't available. Computes motion-envelope, silhouette-
bounding-box, and centroid-trajectory metrics from a side/perpendicular camera
angle.  Detected events and metrics are proxies for the pose-based versions,
but still meaningful for coaching.

What this produces (schema-compatible with analyze_throw.py):

  - annotated/<stem>.cv.mp4         bbox + motion-mag HUD + event labels
  - metrics/<stem>.cv.json          events + metrics
  - metrics/<stem>_frames/*.png     event-frame snapshots
  - history.jsonl                    append row (tagged "backend": "cv")

Usage:
    python scripts/analyze_throw_cv.py clips/my_throw.mov
    python scripts/analyze_throw_cv.py clips/my_throw.mov --notes "..."

Assumptions:
    - Static camera (no pan/zoom)
    - Player is the dominant moving object in frame
    - Camera angle is side / 3/4-side (used for posture proxies)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

# Local helper module. sys.path[0] is this script's directory when run
# directly (python scripts/analyze_throw_cv.py ...), so this import works.
from _video_utils import ensure_browser_playable


# ---------------------------------------------------------------------------
# Tiny scipy-free signal helpers (scipy isn't available in all sandboxes)
# ---------------------------------------------------------------------------


def savgol_filter(a: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Lightweight stand-in for scipy.signal.savgol_filter.

    For simplicity we just apply a centered moving-average of the requested
    window. Adequate for the smoothing precision we need here.
    """
    if window < 3 or len(a) < window:
        return a
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    # Use convolution with 'same' padding, and replicate edges to avoid dips
    pad = window // 2
    padded = np.concatenate([np.full(pad, a[0]), a, np.full(pad, a[-1])])
    conv = np.convolve(padded, kernel, mode="valid")
    return conv[: len(a)]


def find_peaks(
    x: np.ndarray,
    distance: int = 1,
    height: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Minimal find_peaks — local maxima separated by at least `distance`
    samples, above optional `height`. Returns (indices, {}) to mimic scipy's
    (peaks, properties) signature."""
    if len(x) < 3:
        return np.array([], dtype=int), {}
    is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    candidates = np.where(is_peak)[0] + 1
    if height is not None:
        candidates = candidates[x[candidates] >= height]
    if distance > 1 and len(candidates) > 1:
        kept = [int(candidates[0])]
        for c in candidates[1:]:
            if c - kept[-1] >= distance:
                kept.append(int(c))
        candidates = np.array(kept, dtype=int)
    return candidates, {}

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FrameData:
    idx: int
    t_ms: float
    bbox: tuple[int, int, int, int] | None   # x, y, w, h
    centroid_x: float | None
    centroid_y: float | None
    foot_y: float | None                      # lowest silhouette y
    top_y: float | None                       # highest silhouette y
    left_x: float | None
    right_x: float | None
    silhouette_area: int
    motion_mag: float                         # frame-diff magnitude


@dataclass
class Events:
    plant_idx: int | None = None
    reach_back_idx: int | None = None
    power_pocket_idx: int | None = None       # disc at chest, elbow leading, mid-pull
    hit_idx: int | None = None
    release_idx: int | None = None
    setup_idx: int | None = None              # first stable pose before motion
    facing_direction: str = "unknown"         # screen direction player moves


@dataclass
class Metrics:
    # Timing (ms from start)
    setup_t_ms: float | None = None
    plant_t_ms: float | None = None
    reach_back_t_ms: float | None = None
    power_pocket_t_ms: float | None = None
    hit_t_ms: float | None = None
    release_t_ms: float | None = None

    # Tempo
    plant_to_release_ms: float | None = None
    reach_back_to_hit_ms: float | None = None
    reach_back_to_power_pocket_ms: float | None = None
    power_pocket_to_hit_ms: float | None = None
    setup_to_hit_ms: float | None = None

    # Motion envelope — how "explosive" is the hit vs. the buildup?
    peak_motion_mag: float | None = None
    baseline_motion_mag: float | None = None
    hit_to_baseline_ratio: float | None = None     # higher = more snap
    motion_spike_width_ms: float | None = None     # narrower = sharper hit

    # Plant-window stability (±150ms around hit)
    plant_foot_y_stddev_px: float | None = None
    plant_bbox_width_stddev_px: float | None = None

    # Body posture / footprint proxies (from silhouette bbox)
    bbox_width_at_reach_back_px: float | None = None
    bbox_width_at_hit_px: float | None = None
    bbox_aspect_at_reach_back: float | None = None   # w/h
    bbox_aspect_at_hit: float | None = None

    # Centroid travel during throw
    centroid_x_travel_rb_to_hit_px: float | None = None
    centroid_y_travel_rb_to_hit_px: float | None = None

    # x-step cadence proxy — count of bbox-bottom oscillation peaks before hit
    x_step_peaks_before_hit: int | None = None
    x_step_cadence_ms: float | None = None

    flags: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    clip: str
    run_at: str
    backend: str
    fps: float
    frame_count: int
    width: int
    height: int
    notes: str
    events: Events
    metrics: Metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def probe_video(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(
            f"cv2 could not open {path}. If HEVC .MOV, transcode first:\n"
            f"  ffmpeg -i '{path}' -c:v libx264 -crf 18 -preset veryfast '{path.with_suffix('.mp4')}'"
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, w, h


def _estimate_camera_motion(path: Path) -> tuple[list[tuple[float, float]], float]:
    """Return per-frame cumulative (dx, dy) translations aligning every frame
    back to frame-0, plus the maximum absolute cumulative drift in pixels.

    Uses ORB keypoint matches and takes the median of match deltas so the
    player's local motion doesn't dominate the estimate (most keypoints land
    on static background)."""
    cap = cv2.VideoCapture(str(path))
    orb = cv2.ORB_create(500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp = prev_des = None

    # Incremental (frame-to-frame) translations; first frame is (0,0) by def.
    deltas: list[tuple[float, float]] = [(0.0, 0.0)]
    while True:
        ok, img = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(g, None)
        if prev_des is not None and des is not None and len(prev_kp) > 10 and len(kp) > 10:
            matches = bf.match(prev_des, des)
            if len(matches) > 10:
                matches = sorted(matches, key=lambda m: m.distance)[:80]
                pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts_cur = np.float32([kp[m.trainIdx].pt for m in matches])
                d = pts_cur - pts_prev
                deltas.append((float(np.median(d[:, 0])), float(np.median(d[:, 1]))))
            else:
                deltas.append((0.0, 0.0))
        elif prev_des is None:
            pass  # first frame already seeded
        else:
            deltas.append((0.0, 0.0))
        prev_kp, prev_des = kp, des
    cap.release()

    # Cumulative per-frame warp: frame i needs to shift by -sum(deltas[1..i])
    # to align with frame 0.
    cum: list[tuple[float, float]] = [(0.0, 0.0)]
    cx = cy = 0.0
    for dx, dy in deltas[1:]:
        cx += dx
        cy += dy
        cum.append((cx, cy))
    max_drift = max((abs(x) + abs(y)) for x, y in cum) if cum else 0.0
    return cum, max_drift


def _stabilize_to_tempfile(src: Path, cum_shifts: list[tuple[float, float]]) -> Path:
    """Write a camera-stabilized copy of the video by warping each frame by
    the inverse of its cumulative shift. Returns the temp file path."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst = src.with_name(src.stem + ".stabilized.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx < len(cum_shifts):
            tx, ty = cum_shifts[idx]
        else:
            tx, ty = cum_shifts[-1] if cum_shifts else (0.0, 0.0)
        M = np.array([[1.0, 0.0, -tx], [0.0, 1.0, -ty]], dtype=np.float32)
        warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        writer.write(warped)
        idx += 1
    cap.release()
    writer.release()
    return dst


def extract_motion_data(path: Path) -> tuple[list[FrameData], Path]:
    """Run background subtraction + frame differencing to get per-frame:
    silhouette bbox, centroid, motion magnitude.

    If the source clip was shot with a moving camera, we first stabilize it
    (ORB-based global translation estimate + inverse warp) so MOG2 and
    frame-diff have a stable background to work against.

    `motion_mag` is restricted to the silhouette bbox (with a small margin)
    so that background jitter, handheld-camera shake, and lighting wobble
    outside the player do not inflate the baseline. When no silhouette is
    detected yet, motion_mag falls back to full-frame diff."""
    # Detect camera motion and stabilize if needed (threshold: 50px cumulative drift)
    cum_shifts, max_drift = _estimate_camera_motion(path)
    if max_drift > 50.0:
        print(f"      camera motion detected (max drift {max_drift:.0f}px); stabilizing…")
        path = _stabilize_to_tempfile(path, cum_shifts)

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MOG2 background subtractor — handles mild lighting shifts
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    prev_gray: np.ndarray | None = None

    frames: list[FrameData] = []
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Silhouette via MOG2 (computed before motion so we can mask by bbox)
        fg = bg_sub.apply(img)
        # Drop shadow pixels (MOG2 marks them with value 127)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox = None
        centroid_x = centroid_y = None
        foot_y = top_y = left_x = right_x = None
        silhouette_area = 0

        if contours:
            # Take largest contour — assumed to be the player
            cnt = max(contours, key=cv2.contourArea)
            area = int(cv2.contourArea(cnt))
            if area > 2000:  # filter tiny noise
                x, y, w, h = cv2.boundingRect(cnt)
                bbox = (int(x), int(y), int(w), int(h))
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    centroid_x = float(M["m10"] / M["m00"])
                    centroid_y = float(M["m01"] / M["m00"])
                foot_y = float(y + h)
                top_y = float(y)
                left_x = float(x)
                right_x = float(x + w)
                silhouette_area = area

        # Motion magnitude: frame diff restricted to the silhouette bbox
        # (with a 20% margin so arm/disc motion just outside the core
        # silhouette is still counted).
        if prev_gray is None:
            motion_mag = 0.0
        else:
            diff = cv2.absdiff(gray_blur, prev_gray)
            _, thr = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
            if bbox is not None:
                x, y, w, h = bbox
                mx = int(w * 0.20)
                my = int(h * 0.10)
                x0 = max(0, x - mx)
                y0 = max(0, y - my)
                x1 = min(frame_w, x + w + mx)
                y1 = min(frame_h, y + h + my)
                motion_mag = float(np.sum(thr[y0:y1, x0:x1]) / 255.0)
            else:
                motion_mag = float(np.sum(thr) / 255.0)
        prev_gray = gray_blur

        frames.append(
            FrameData(
                idx=idx,
                t_ms=1000.0 * idx / fps,
                bbox=bbox,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                foot_y=foot_y,
                top_y=top_y,
                left_x=left_x,
                right_x=right_x,
                silhouette_area=silhouette_area,
                motion_mag=motion_mag,
            )
        )
        idx += 1

    cap.release()
    return frames, path


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------


def _series(frames: list[FrameData], attr: str) -> np.ndarray:
    out = np.full(len(frames), np.nan)
    for i, f in enumerate(frames):
        v = getattr(f, attr)
        if v is not None:
            out[i] = float(v)
    return _interp_nan(out)


def _interp_nan(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    mask = np.isfinite(a)
    if not mask.any():
        return a
    idxs = np.arange(len(a))
    a[~mask] = np.interp(idxs[~mask], idxs[mask], a[mask])
    return a


def _smooth(a: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    if len(a) < window or window % 2 == 0:
        return a
    try:
        return savgol_filter(a, window, polyorder)
    except Exception:
        return a


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


def detect_events(frames: list[FrameData], fps: float) -> Events:
    events = Events()
    if len(frames) < 10:
        return events

    smooth_win = max(5, int(fps // 6) | 1)
    motion = _smooth(np.array([f.motion_mag for f in frames]), window=smooth_win)
    cx = _smooth(_series(frames, "centroid_x"), window=smooth_win)
    foot_y = _smooth(_series(frames, "foot_y"), window=smooth_win)
    left_x = _smooth(_series(frames, "left_x"), window=smooth_win)
    right_x = _smooth(_series(frames, "right_x"), window=smooth_win)
    bbox_width = right_x - left_x

    # Facing direction from net centroid travel
    early = np.nanmean(cx[: max(1, len(cx) // 5)])
    late = np.nanmean(cx[-max(1, len(cx) // 5):])
    if np.isfinite(early) and np.isfinite(late):
        delta = late - early
        if abs(delta) < 15:
            events.facing_direction = "standstill"
        else:
            events.facing_direction = "right" if delta > 0 else "left"

    # --- Hit = peak motion in the last 60% of the clip ---
    search_start = int(len(motion) * 0.4)
    local = motion[search_start:]
    hit_rel = int(np.argmax(local))
    events.hit_idx = search_start + hit_rel

    # --- Setup = frame before sustained motion begins. Detected early so
    # reach-back and plant can use it as a soft lower bound. ---
    baseline = float(np.nanmedian(motion[: max(3, len(motion) // 10)]))
    thresh = max(baseline * 3.0, baseline + 50.0)
    above = np.where(motion > thresh)[0]
    if len(above) > 0:
        events.setup_idx = max(0, int(above[0]) - 1)

    # Effective search window for pre-hit events. We cap at ~1.5s before the
    # hit regardless of where setup_idx was — on long clips with a multi-step
    # runup, reach-back and plant both live in the final second or so before
    # the hit. If setup_idx is *inside* that 1.5s window we respect it (don't
    # wander into the warmup); otherwise we let the 1.5s cap rule.
    setup_lo = events.setup_idx if events.setup_idx is not None else 0
    window_lo = max(0, events.hit_idx - int(fps * 1.5))
    pre_lo = max(setup_lo, window_lo)
    # Exclude the last ~80ms before the hit itself so the swing-through doesn't
    # masquerade as reach-back.
    pre_hi = max(pre_lo + 3, events.hit_idx - max(1, int(fps * 0.08)))

    # --- Reach-back = frame where the throwing arm is maximally extended.
    # On a lateral clip this shows up as the silhouette's widest frame in the
    # pre-hit window — the extended arm sticks out beyond the body. The catch
    # is that the swing-through arm *also* widens the silhouette, so naive
    # argmax(width) can pick a pull-phase frame instead of the real reach-back.
    #
    # To isolate reach-back we cap the search at the onset of the forward
    # pull: the first frame where the silhouette's TARGET-side edge extends
    # beyond its pre-pull baseline. For facing="left" (target to the left of
    # frame) the target-side edge is left_x; for "right", right_x. Anything
    # wider than that cap is the arm already swinging through.
    #
    # On standstill clips we fall back to the legacy motion-minimum heuristic
    # (there's no reliable lateral signal to exploit). ---
    if events.facing_direction in ("right", "left") and pre_hi - pre_lo >= 4:
        if events.facing_direction == "left":
            # target is to the left — forward-extension grows as cx-lx grows
            forward_asym = cx - left_x
        else:
            # facing == "right" — forward-extension grows as rx-cx grows
            forward_asym = right_x - cx

        # Pre-pull baseline: median of the first half of the pre-hit window.
        pre_mid = pre_lo + max(3, (pre_hi - pre_lo) // 2)
        baseline_pre = forward_asym[pre_lo:pre_mid]
        finite_pre = baseline_pre[np.isfinite(baseline_pre)]
        if finite_pre.size > 0:
            fa_baseline = float(np.nanmedian(finite_pre))
            fa_threshold = max(fa_baseline * 1.6, fa_baseline + 30.0)
            fa_pre = forward_asym[pre_lo:events.hit_idx]
            fa_clean = _interp_nan(fa_pre.copy())
            above_thr = np.where(fa_clean > fa_threshold)[0]
            pull_start = int(pre_lo + above_thr[0]) if len(above_thr) > 0 else int(pre_hi)
        else:
            pull_start = int(pre_hi)

        rb_search_hi = min(pre_hi, pull_start)
        if rb_search_hi - pre_lo >= 4:
            w_seg = bbox_width[pre_lo:rb_search_hi]
            if np.isfinite(w_seg).any():
                events.reach_back_idx = int(pre_lo + int(np.nanargmax(w_seg)))
    if events.reach_back_idx is None:
        # Standstill / width-unavailable fallback: last local min of motion
        rb_window_frames = int(fps * 0.8)
        rb_lo = max(pre_lo, events.hit_idx - rb_window_frames)
        rb_slice = motion[rb_lo:events.hit_idx]
        if len(rb_slice) > 3:
            neg = -rb_slice
            peaks, _ = find_peaks(neg, distance=max(2, int(fps * 0.08)))
            if len(peaks) > 0:
                events.reach_back_idx = int(rb_lo + peaks[-1])
            else:
                events.reach_back_idx = int(rb_lo + int(np.argmin(rb_slice)))

    # --- Release = first local minimum of motion AFTER the hit ---
    rel_search_hi = min(len(motion), events.hit_idx + int(fps * 0.6))
    rel_slice = motion[events.hit_idx:rel_search_hi]
    if len(rel_slice) > 3:
        neg = -rel_slice
        peaks, _ = find_peaks(neg, distance=max(2, int(fps * 0.08)))
        if len(peaks) > 0:
            events.release_idx = int(events.hit_idx + peaks[0])
        else:
            events.release_idx = int(events.hit_idx + int(np.argmin(rel_slice)))

    # --- Plant = earliest frame in [pre_lo, hit] where lead foot y settles at
    # its local max (image y grows downward, so foot on ground = largest y)
    # and stays there for a sustained run of frames. Structurally decoupled
    # from reach_back: plant can happen before OR after reach-back — ideal is
    # simultaneous, but real players vary. ---
    plant_search_lo = pre_lo
    plant_search_hi = events.hit_idx
    if plant_search_hi is not None and plant_search_hi - plant_search_lo >= 3:
        seg = foot_y[plant_search_lo:plant_search_hi]
        if np.isfinite(seg).any() and len(seg) > 0:
            seg_clean = _interp_nan(seg.copy())
            target = np.nanmax(seg_clean)
            tol = 4.0  # pixels — same tolerance as the prior implementation
            within_mask = np.abs(seg_clean - target) < tol
            # Require a sustained run (~100ms of stability) so a single noisy
            # frame at the local max doesn't win.
            min_run = max(3, int(fps * 0.1))
            run_start: int | None = None
            plant_rel: int | None = None
            for i, is_within in enumerate(within_mask):
                if is_within:
                    if run_start is None:
                        run_start = i
                    if i - run_start + 1 >= min_run:
                        plant_rel = run_start
                        break
                else:
                    run_start = None
            if plant_rel is None:
                # No sustained run — take the first frame inside the tolerance
                # band, or (degenerate) the frame at the local max.
                within = np.where(within_mask)[0]
                if len(within) > 0:
                    plant_rel = int(within[0])
                else:
                    plant_rel = int(np.nanargmax(seg_clean))
            events.plant_idx = int(plant_search_lo + plant_rel)

    return events


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    frames: list[FrameData],
    events: Events,
    fps: float,
    cfg: dict,
) -> Metrics:
    m = Metrics()

    def t(idx: int | None) -> float | None:
        return round(frames[idx].t_ms, 1) if idx is not None and 0 <= idx < len(frames) else None

    m.setup_t_ms = t(events.setup_idx)
    m.plant_t_ms = t(events.plant_idx)
    m.reach_back_t_ms = t(events.reach_back_idx)
    m.power_pocket_t_ms = t(events.power_pocket_idx)
    m.hit_t_ms = t(events.hit_idx)
    m.release_t_ms = t(events.release_idx)

    if m.plant_t_ms is not None and m.release_t_ms is not None:
        m.plant_to_release_ms = round(m.release_t_ms - m.plant_t_ms, 1)
    if m.reach_back_t_ms is not None and m.hit_t_ms is not None:
        m.reach_back_to_hit_ms = round(m.hit_t_ms - m.reach_back_t_ms, 1)
    if m.reach_back_t_ms is not None and m.power_pocket_t_ms is not None:
        m.reach_back_to_power_pocket_ms = round(m.power_pocket_t_ms - m.reach_back_t_ms, 1)
    if m.power_pocket_t_ms is not None and m.hit_t_ms is not None:
        m.power_pocket_to_hit_ms = round(m.hit_t_ms - m.power_pocket_t_ms, 1)
    if m.setup_t_ms is not None and m.hit_t_ms is not None:
        m.setup_to_hit_ms = round(m.hit_t_ms - m.setup_t_ms, 1)

    motion = np.array([f.motion_mag for f in frames])

    # Baseline = quietest sustained window anywhere in the clip. Finds the held
    # moment (pre-throw setup, between steps, or post-release stillness) rather
    # than assuming the first 10% is quiet — which is false when the player is
    # already moving in frame 0.
    win = max(3, int(fps * 0.3))  # ~300ms window
    if len(motion) >= win:
        # Rolling median via a simple loop (fine for <2k frames)
        rolling = np.array([
            float(np.nanmedian(motion[i : i + win]))
            for i in range(0, len(motion) - win + 1)
        ])
        m.baseline_motion_mag = round(float(np.nanmin(rolling)), 1)
    else:
        m.baseline_motion_mag = round(float(np.nanmedian(motion)), 1)

    if events.hit_idx is not None:
        m.peak_motion_mag = round(float(motion[events.hit_idx]), 1)
        if m.baseline_motion_mag is not None and m.baseline_motion_mag > 0:
            m.hit_to_baseline_ratio = round(m.peak_motion_mag / m.baseline_motion_mag, 2)

    # Motion spike width: width at half-prominence (baseline + 50% of the peak's
    # rise above baseline). Bounded to a ±0.5s window around the hit so a
    # secondary motion burst (e.g. player walking off the tee after release)
    # doesn't inflate the reading.
    if events.hit_idx is not None and m.peak_motion_mag is not None and m.baseline_motion_mag is not None:
        prom = m.peak_motion_mag - m.baseline_motion_mag
        if prom > 0:
            half = m.baseline_motion_mag + 0.5 * prom
            i = events.hit_idx
            bound_lo = max(0, i - int(fps * 0.75))
            bound_hi = min(len(motion) - 1, i + int(fps * 0.75))
            lo = i
            while lo > bound_lo and motion[lo] > half:
                lo -= 1
            hi = i
            while hi < bound_hi and motion[hi] > half:
                hi += 1
            m.motion_spike_width_ms = round(1000.0 * (hi - lo) / fps, 1)

    # Plant-window stability: ±150ms around hit
    if events.hit_idx is not None:
        half_w = max(1, int(fps * 0.15))
        lo = max(0, events.hit_idx - half_w)
        hi = min(len(frames), events.hit_idx + half_w + 1)
        foot_ys = [f.foot_y for f in frames[lo:hi] if f.foot_y is not None]
        if len(foot_ys) >= 3:
            m.plant_foot_y_stddev_px = round(float(np.std(foot_ys)), 2)
        widths = [f.bbox[2] for f in frames[lo:hi] if f.bbox is not None]
        if len(widths) >= 3:
            m.plant_bbox_width_stddev_px = round(float(np.std(widths)), 2)

    # Body footprint at reach-back / hit
    def snapshot(idx: int | None):
        if idx is None or frames[idx].bbox is None:
            return None
        x, y, w, h = frames[idx].bbox
        return {"w": w, "h": h, "aspect": w / max(1, h)}

    rb = snapshot(events.reach_back_idx)
    hit = snapshot(events.hit_idx)
    if rb:
        m.bbox_width_at_reach_back_px = float(rb["w"])
        m.bbox_aspect_at_reach_back = round(rb["aspect"], 3)
    if hit:
        m.bbox_width_at_hit_px = float(hit["w"])
        m.bbox_aspect_at_hit = round(hit["aspect"], 3)

    # Centroid travel from reach-back to hit
    if events.reach_back_idx is not None and events.hit_idx is not None:
        a, b = events.reach_back_idx, events.hit_idx
        ax, ay = frames[a].centroid_x, frames[a].centroid_y
        bx, by = frames[b].centroid_x, frames[b].centroid_y
        if None not in (ax, ay, bx, by):
            m.centroid_x_travel_rb_to_hit_px = round(float(bx - ax), 2)
            m.centroid_y_travel_rb_to_hit_px = round(float(by - ay), 2)

    # x-step cadence: count motion peaks between setup and reach-back
    if events.setup_idx is not None and events.reach_back_idx is not None:
        seg = motion[events.setup_idx:events.reach_back_idx]
        if len(seg) > 5:
            # peaks at least 150ms apart, height > 30% of max segment motion
            peaks, _ = find_peaks(
                seg,
                distance=max(2, int(fps * 0.15)),
                height=float(0.3 * np.nanmax(seg)) if np.nanmax(seg) > 0 else 0.0,
            )
            m.x_step_peaks_before_hit = int(len(peaks))
            if len(peaks) >= 2:
                diffs = np.diff(peaks)
                m.x_step_cadence_ms = round(float(np.mean(diffs) * 1000 / fps), 1)

    # Flags
    thresholds = cfg["pipeline"]["thresholds"]
    if m.plant_foot_y_stddev_px is not None and m.plant_foot_y_stddev_px > thresholds["plant_ankle_px_stddev_flag"]:
        m.flags.append("weak_plant_foot_vertical")
    if m.hit_to_baseline_ratio is not None and m.hit_to_baseline_ratio < 15.0:
        m.flags.append("soft_hit_motion_envelope")
    if m.motion_spike_width_ms is not None and m.motion_spike_width_ms > 250.0:
        m.flags.append("broad_hit_envelope_possible_arm_throw")

    return m


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def annotate_video(
    src: Path,
    dst: Path,
    frames: list[FrameData],
    events: Events,
    fps: float,
) -> None:
    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))

    # Pre-compute motion max for HUD bar scaling
    motion = np.array([f.motion_mag for f in frames])
    motion_max = float(np.nanmax(motion)) if len(motion) > 0 else 1.0

    event_labels = {
        events.setup_idx: "SETUP",
        events.plant_idx: "PLANT",
        events.reach_back_idx: "REACH-BACK",
        events.power_pocket_idx: "POWER POCKET",
        events.hit_idx: "HIT",
        events.release_idx: "RELEASE",
    }

    for i in range(len(frames)):
        ok, img = cap.read()
        if not ok:
            break

        fd = frames[i]

        # Draw silhouette bbox
        if fd.bbox is not None:
            x, y, bw, bh = fd.bbox
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            if fd.centroid_x is not None and fd.centroid_y is not None:
                cv2.circle(img, (int(fd.centroid_x), int(fd.centroid_y)), 6, (0, 255, 255), -1)

        # HUD — top-left text
        hud = [
            f"t={fd.t_ms:6.0f}ms  f={i}",
            f"motion={fd.motion_mag:6.0f}",
            f"facing={events.facing_direction}",
        ]
        y_text = 28
        for line in hud:
            cv2.putText(img, line, (12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y_text += 28

        # Motion bar on right edge
        bar_h = int((fd.motion_mag / max(1, motion_max)) * (h - 60))
        cv2.rectangle(img, (w - 30, h - 30 - bar_h), (w - 10, h - 30), (0, 200, 255), -1)

        # Event label if this frame matches
        label = event_labels.get(i)
        if label:
            cv2.rectangle(img, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(img, label, (14, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)

        writer.write(img)

    cap.release()
    writer.release()


def save_event_snapshots(src: Path, events: Events, stem: str) -> None:
    event_frames = {
        "setup": events.setup_idx,
        "plant": events.plant_idx,
        "reach_back": events.reach_back_idx,
        "power_pocket": events.power_pocket_idx,
        "hit": events.hit_idx,
        "release": events.release_idx,
    }
    out_dir = ROOT / "metrics" / f"{stem}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe prior *_frame_*.png so a re-run with corrected indices doesn't leave
    # stale snapshots alongside the new ones (list_keyframes only shows one per
    # event, and which one wins is filesystem-order-dependent).
    for old in out_dir.glob("*_frame_*.png"):
        old.unlink()
    cap = cv2.VideoCapture(str(src))
    try:
        for label, idx in event_frames.items():
            if idx is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, img = cap.read()
            if ok:
                cv2.imwrite(str(out_dir / f"{label}_frame_{idx:06d}.png"), img)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    clip: Path,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    events: Events,
    metrics: Metrics,
    notes: str,
) -> AnalysisResult:
    run_at = datetime.now(timezone.utc).isoformat()
    result = AnalysisResult(
        clip=clip.name,
        run_at=run_at,
        backend="cv2",
        fps=round(fps, 3),
        frame_count=frame_count,
        width=width,
        height=height,
        notes=notes,
        events=events,
        metrics=metrics,
    )

    metrics_path = ROOT / "metrics" / f"{clip.stem}.cv.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    history_path = ROOT / "history.jsonl"
    row = {
        "clip": clip.name,
        "run_at": run_at,
        "backend": "cv2",
        "fps": result.fps,
        "metrics": asdict(metrics),
        "events": asdict(events),
        "notes": notes,
    }
    with history_path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument(
        "--no-transcode",
        action="store_true",
        help="skip the H.264 post-process step (the browser UI needs it, but "
             "batch/CI runs may not)",
    )
    # Event overrides — pin any event to a specific frame index, bypassing
    # the auto-detector for that event. Useful when the user ground-truths
    # events from the video after a pipeline misdetection.
    ap.add_argument("--setup", type=int, default=None, help="override setup frame index")
    ap.add_argument("--plant", type=int, default=None, help="override plant frame index")
    ap.add_argument("--reach-back", type=int, default=None, help="override reach-back frame index")
    ap.add_argument("--power-pocket", type=int, default=None, help="override power-pocket frame index")
    ap.add_argument("--hit", type=int, default=None, help="override hit frame index")
    ap.add_argument("--release", type=int, default=None, help="override release frame index")
    args = ap.parse_args()

    clip: Path = args.clip
    if not clip.exists():
        print(f"clip not found: {clip}", file=sys.stderr)
        return 2

    cfg = load_config()
    print(f"[1/6] probing: {clip.name}")
    fps, frame_count, w, h = probe_video(clip)
    print(f"      {w}x{h} @ {fps:.2f}fps, {frame_count} frames")

    print("[2/6] extracting motion + silhouettes (cv2)...")
    frames, analysis_clip = extract_motion_data(clip)
    detected_bbox = sum(1 for f in frames if f.bbox is not None)
    print(f"      silhouette detected on {detected_bbox}/{len(frames)} frames")

    print("[3/6] detecting events...")
    events = detect_events(frames, fps)

    # Apply CLI overrides on top of auto-detection. Any flag explicitly set
    # overrides the detected value; unspecified flags leave auto-detect alone.
    overrides = {
        "setup_idx": args.setup,
        "plant_idx": args.plant,
        "reach_back_idx": args.reach_back,
        "power_pocket_idx": args.power_pocket,
        "hit_idx": args.hit,
        "release_idx": args.release,
    }
    applied = []
    for attr, val in overrides.items():
        if val is not None:
            setattr(events, attr, int(val))
            applied.append(f"{attr}={val}")
    if applied:
        print(f"      overrides applied: {', '.join(applied)}")

    print(f"      setup@{events.setup_idx}  plant@{events.plant_idx}  "
          f"reach_back@{events.reach_back_idx}  power_pocket@{events.power_pocket_idx}  "
          f"hit@{events.hit_idx}  release@{events.release_idx}  "
          f"facing={events.facing_direction}")

    print("[4/6] computing metrics...")
    metrics = compute_metrics(frames, events, fps, cfg)
    print(f"      flags: {', '.join(metrics.flags) or 'none'}")

    if not args.no_annotate:
        dst = ROOT / "annotated" / f"{clip.stem}.cv.mp4"
        dst.parent.mkdir(exist_ok=True)
        print(f"[5/6] writing annotated video: {dst}")
        annotate_video(analysis_clip, dst, frames, events, fps)
        if not args.no_transcode:
            status, detail = ensure_browser_playable(dst)
            if status == "transcoded":
                print("      post-processed to H.264 for browser playback")
            elif status == "failed":
                print(f"      [warn] H.264 post-process failed — video may not play in browser: {detail}")
            elif status == "skipped":
                print(f"      [warn] skipping H.264 post-process: {detail}")
    else:
        print("[5/6] skipping annotation")

    print("[6/6] writing metrics + event snapshots + history...")
    write_outputs(clip, fps, frame_count, w, h, events, metrics, args.notes)
    save_event_snapshots(analysis_clip, events, clip.stem)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
