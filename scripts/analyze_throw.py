#!/usr/bin/env python3
"""
Disc Golf Tee Shot Analyzer
===========================

Takes a side-view video of a disc golf backhand tee shot, runs MediaPipe Pose,
detects key events (plant, reach-back, hit, release), computes metrics, and
writes:

  - annotated/<stem>.mp4       skeleton overlay + event labels
  - metrics/<stem>.json        detected events + computed metrics
  - history.jsonl              append-only trend log (metrics only)

Designed for a side/perpendicular camera angle. Handedness (backhand) is
auto-detected from wrist velocity peaks unless overridden in config.yaml.

Usage:
    python scripts/analyze_throw.py clips/my_throw.mp4
    python scripts/analyze_throw.py clips/my_throw.mp4 --notes "windy, shallow reach-back"

This script is deliberately transparent about what it can and cannot reliably
measure from side view. See README.md and COACH.md for the caveat on camera
angles.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.signal import savgol_filter

import mediapipe as mp

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# MediaPipe landmark indices (subset we care about)
# ---------------------------------------------------------------------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

L = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FramePose:
    """Pose data for a single frame. Pixel coordinates, origin top-left."""

    idx: int
    t_ms: float
    landmarks: dict[str, tuple[float, float, float]]  # name -> (x_px, y_px, visibility)


@dataclass
class Events:
    """Frame indices for detected key events. None if not detected."""

    plant_idx: int | None = None
    reach_back_idx: int | None = None
    hit_idx: int | None = None
    release_idx: int | None = None
    throw_wrist: str = "right_wrist"       # auto-detected
    plant_foot_ankle: str = "left_ankle"   # opposite of throw side for a backhand
    facing_direction: str = "unknown"      # "left" or "right" (screen direction player moves)


@dataclass
class Metrics:
    """Computed side-view metrics. Units are documented per field."""

    # Keyframes (ms from start)
    plant_t_ms: float | None = None
    reach_back_t_ms: float | None = None
    hit_t_ms: float | None = None
    release_t_ms: float | None = None

    # Tempo
    plant_to_release_ms: float | None = None
    reach_back_to_hit_ms: float | None = None

    # Reach-back extent: wrist horizontal distance behind torso center
    # at reach-back peak, normalized by shoulder-to-hip (torso) length.
    reach_back_extent_ratio: float | None = None

    # Plant-foot stability: stddev of front ankle (x,y) in hit window (±150ms),
    # in pixels. Lower = quieter plant foot.
    plant_ankle_x_stddev_px: float | None = None
    plant_ankle_y_stddev_px: float | None = None

    # Release height: wrist y-position at release, normalized by image height.
    # Lower numbers = higher release (image y-axis is inverted).
    release_height_norm: float | None = None

    # Spine lean at hit: angle between torso axis and vertical, degrees.
    spine_lean_deg_at_hit: float | None = None

    # Shoulder-vs-hip opening order (side-view proxy).
    # Negative = shoulders open before hips (bad). Positive = hips lead (good).
    # Measured as (shoulder_opening_time - hip_opening_time) in ms.
    hip_lead_ms: float | None = None

    # Rounding proxy (side view): max sagittal deviation of throwing wrist
    # from the straight line between reach-back and release, in pixels.
    # Larger = more curved pull path. Rough proxy — true rounding is a
    # horizontal (transverse-plane) arc, which side view cannot see directly.
    wrist_path_sagittal_deviation_px: float | None = None

    # Flags derived from thresholds in config.yaml
    flags: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    clip: str
    run_at: str
    fps: float
    frame_count: int
    width: int
    height: int
    notes: str
    events: Events
    metrics: Metrics


# ---------------------------------------------------------------------------
# Video IO
# ---------------------------------------------------------------------------


def probe_video(path: Path) -> tuple[float, int, int, int]:
    """Return (fps, frame_count, width, height)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(
            f"cv2 could not open {path}. If this is an iPhone .MOV, transcode with:\n"
            f"  ffmpeg -i {path.name} -c:v libx264 -crf 18 -preset veryfast -c:a aac {path.stem}.mp4"
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, width, height


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------


def run_pose(path: Path, cfg: dict) -> list[FramePose]:
    pose_cfg = cfg["pipeline"]["pose"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames: list[FramePose] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=pose_cfg["model_complexity"],
        enable_segmentation=False,
        min_detection_confidence=pose_cfg["min_detection_confidence"],
        min_tracking_confidence=pose_cfg["min_tracking_confidence"],
    ) as pose:
        idx = 0
        while True:
            ok, image_bgr = cap.read()
            if not ok:
                break
            h, w = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            landmarks: dict[str, tuple[float, float, float]] = {}
            if results.pose_landmarks:
                for name, li in L.items():
                    lm = results.pose_landmarks.landmark[li]
                    landmarks[name] = (lm.x * w, lm.y * h, lm.visibility)

            frames.append(FramePose(idx=idx, t_ms=1000.0 * idx / fps, landmarks=landmarks))
            idx += 1

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_landmarks(frames: list[FramePose], cfg: dict) -> list[FramePose]:
    """Apply Savitzky-Golay smoothing per landmark over time, forward-filling
    missing frames where the landmark was not detected."""
    window = max(5, cfg["pipeline"]["smoothing"]["window_frames"] | 1)  # must be odd
    min_vis = cfg["pipeline"]["smoothing"]["min_landmark_visibility"]
    if len(frames) < window:
        return frames  # too short to smooth

    names = list(L.keys())
    n = len(frames)

    # Build arrays per landmark of x, y, visibility with NaNs for missing.
    xs = {name: np.full(n, np.nan) for name in names}
    ys = {name: np.full(n, np.nan) for name in names}
    vs = {name: np.full(n, 0.0) for name in names}
    for i, fp in enumerate(frames):
        for name in names:
            if name in fp.landmarks:
                x, y, v = fp.landmarks[name]
                if v >= min_vis:
                    xs[name][i] = x
                    ys[name][i] = y
                    vs[name][i] = v

    # Interpolate NaNs (simple linear), then smooth.
    for name in names:
        xs[name] = _interp_nan(xs[name])
        ys[name] = _interp_nan(ys[name])
        if np.isfinite(xs[name]).all() and n >= window:
            xs[name] = savgol_filter(xs[name], window, polyorder=2)
            ys[name] = savgol_filter(ys[name], window, polyorder=2)

    # Write back.
    out: list[FramePose] = []
    for i, fp in enumerate(frames):
        new_lm: dict[str, tuple[float, float, float]] = {}
        for name in names:
            if np.isfinite(xs[name][i]) and np.isfinite(ys[name][i]):
                new_lm[name] = (float(xs[name][i]), float(ys[name][i]), float(vs[name][i]))
        out.append(FramePose(idx=fp.idx, t_ms=fp.t_ms, landmarks=new_lm))
    return out


def _interp_nan(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    n = len(a)
    mask = np.isfinite(a)
    if not mask.any():
        return a
    idxs = np.arange(n)
    a[~mask] = np.interp(idxs[~mask], idxs[mask], a[mask])
    return a


# ---------------------------------------------------------------------------
# Handedness & facing auto-detection
# ---------------------------------------------------------------------------


def _series(frames: list[FramePose], name: str, axis: int = 0) -> np.ndarray:
    out = np.full(len(frames), np.nan)
    for i, fp in enumerate(frames):
        if name in fp.landmarks:
            out[i] = fp.landmarks[name][axis]
    return _interp_nan(out)


def detect_handedness(frames: list[FramePose], configured: str) -> str:
    """Returns 'right' or 'left' (throwing hand). Uses configured value unless 'auto'."""
    if configured in ("right", "left"):
        return configured
    rx = _series(frames, "right_wrist", 0)
    lx = _series(frames, "left_wrist", 0)
    rv = np.nanmax(np.abs(np.diff(rx))) if len(rx) > 1 else 0.0
    lv = np.nanmax(np.abs(np.diff(lx))) if len(lx) > 1 else 0.0
    return "right" if rv >= lv else "left"


def detect_facing(frames: list[FramePose]) -> str:
    """From side view, detect which screen-direction the thrower moves during run-up.
    Uses net horizontal displacement of the hip midpoint.
    """
    lhx = _series(frames, "left_hip", 0)
    rhx = _series(frames, "right_hip", 0)
    mid = (lhx + rhx) / 2.0
    if len(mid) < 4:
        return "unknown"
    early = np.nanmean(mid[: max(1, len(mid) // 5)])
    late = np.nanmean(mid[-max(1, len(mid) // 5) :])
    if np.isnan(early) or np.isnan(late):
        return "unknown"
    delta = late - early
    if abs(delta) < 8:  # pixels — not much movement, could be a standstill
        return "standstill"
    return "right" if delta > 0 else "left"


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


def detect_events(frames: list[FramePose], fps: float, handedness: str) -> Events:
    throw_wrist = f"{handedness}_wrist"
    plant_foot = "left_ankle" if handedness == "right" else "right_ankle"

    wx = _series(frames, throw_wrist, 0)
    wy = _series(frames, throw_wrist, 1)
    speed = np.hypot(np.gradient(wx), np.gradient(wy))  # px per frame

    events = Events(throw_wrist=throw_wrist, plant_foot_ankle=plant_foot)
    events.facing_direction = detect_facing(frames)

    if len(speed) < 10:
        return events

    # Hit ~ moment of peak wrist speed in the last ~40% of the clip
    search_start = int(len(speed) * 0.4)
    hit_idx = search_start + int(np.nanargmax(speed[search_start:]))
    events.hit_idx = int(hit_idx)

    # Release ~ next local minimum of speed AFTER hit (wrist decelerates and disc is gone)
    rel_idx = hit_idx
    for i in range(hit_idx + 1, min(len(speed) - 1, hit_idx + int(fps * 0.5))):
        if speed[i + 1] > speed[i] and speed[i] < speed[hit_idx] * 0.5:
            rel_idx = i
            break
    events.release_idx = int(rel_idx)

    # Reach-back ~ extreme wrist x behind torso BEFORE hit.
    hip_mid_x = (_series(frames, "left_hip", 0) + _series(frames, "right_hip", 0)) / 2.0
    behind_signed = wx[:hit_idx] - hip_mid_x[:hit_idx]
    if events.facing_direction == "right":
        # Player moves right, so reach-back is wrist FURTHEST LEFT of hips → most negative
        if len(behind_signed) > 0:
            events.reach_back_idx = int(np.nanargmin(behind_signed))
    elif events.facing_direction == "left":
        if len(behind_signed) > 0:
            events.reach_back_idx = int(np.nanargmax(behind_signed))
    else:
        # fallback: use the max |behind| before hit
        if len(behind_signed) > 0:
            events.reach_back_idx = int(np.nanargmax(np.abs(behind_signed)))

    # Plant ~ front-ankle y reaches its low (contact) before hit, and then stays low.
    ay = _series(frames, plant_foot, 1)
    # Plant = earliest frame in last 60% where ankle y is within 8px of its minimum
    if hit_idx > 5:
        window = ay[: hit_idx + 1]
        if np.isfinite(window).any():
            target = np.nanmin(window)
            candidates = np.where(np.abs(window - target) < 8)[0]
            if len(candidates) > 0:
                events.plant_idx = int(candidates[0])

    return events


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    frames: list[FramePose],
    events: Events,
    fps: float,
    height_px: int,
    cfg: dict,
) -> Metrics:
    m = Metrics()
    thresholds = cfg["pipeline"]["thresholds"]
    throw_wrist = events.throw_wrist
    plant_foot = events.plant_foot_ankle

    def t(idx: int | None) -> float | None:
        return frames[idx].t_ms if idx is not None and 0 <= idx < len(frames) else None

    m.plant_t_ms = t(events.plant_idx)
    m.reach_back_t_ms = t(events.reach_back_idx)
    m.hit_t_ms = t(events.hit_idx)
    m.release_t_ms = t(events.release_idx)

    if m.plant_t_ms is not None and m.release_t_ms is not None:
        m.plant_to_release_ms = m.release_t_ms - m.plant_t_ms
    if m.reach_back_t_ms is not None and m.hit_t_ms is not None:
        m.reach_back_to_hit_ms = m.hit_t_ms - m.reach_back_t_ms

    # Reach-back extent
    if events.reach_back_idx is not None:
        fp = frames[events.reach_back_idx]
        if all(k in fp.landmarks for k in (throw_wrist, "left_hip", "right_hip", "left_shoulder", "right_shoulder")):
            wx = fp.landmarks[throw_wrist][0]
            hip_mid_x = (fp.landmarks["left_hip"][0] + fp.landmarks["right_hip"][0]) / 2.0
            shoulder_mid_y = (fp.landmarks["left_shoulder"][1] + fp.landmarks["right_shoulder"][1]) / 2.0
            hip_mid_y = (fp.landmarks["left_hip"][1] + fp.landmarks["right_hip"][1]) / 2.0
            torso_len = max(1.0, abs(shoulder_mid_y - hip_mid_y))
            m.reach_back_extent_ratio = round(abs(wx - hip_mid_x) / torso_len, 3)

    # Plant stability: stddev of plant-foot ankle (x,y) in ±150ms window around hit
    if events.hit_idx is not None:
        half_w = max(1, int(fps * 0.15))
        lo = max(0, events.hit_idx - half_w)
        hi = min(len(frames), events.hit_idx + half_w + 1)
        axs = [frames[i].landmarks[plant_foot][0] for i in range(lo, hi) if plant_foot in frames[i].landmarks]
        ays = [frames[i].landmarks[plant_foot][1] for i in range(lo, hi) if plant_foot in frames[i].landmarks]
        if len(axs) >= 3:
            m.plant_ankle_x_stddev_px = round(float(np.std(axs)), 2)
            m.plant_ankle_y_stddev_px = round(float(np.std(ays)), 2)

    # Release height
    if events.release_idx is not None and throw_wrist in frames[events.release_idx].landmarks:
        wy = frames[events.release_idx].landmarks[throw_wrist][1]
        m.release_height_norm = round(wy / height_px, 3)

    # Spine lean at hit
    if events.hit_idx is not None:
        fp = frames[events.hit_idx]
        if all(k in fp.landmarks for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
            sx = (fp.landmarks["left_shoulder"][0] + fp.landmarks["right_shoulder"][0]) / 2.0
            sy = (fp.landmarks["left_shoulder"][1] + fp.landmarks["right_shoulder"][1]) / 2.0
            hx = (fp.landmarks["left_hip"][0] + fp.landmarks["right_hip"][0]) / 2.0
            hy = (fp.landmarks["left_hip"][1] + fp.landmarks["right_hip"][1]) / 2.0
            # angle between torso vector (hip -> shoulder) and vertical (up)
            vx, vy = sx - hx, sy - hy
            ang = math.degrees(math.atan2(vx, -vy))  # 0 = perfectly upright
            m.spine_lean_deg_at_hit = round(ang, 1)

    # Hip-lead proxy: when does each pair "open"?
    # Use the time at which shoulder_line slope vs. hip_line slope starts changing fast.
    # Side view limitation: magnitudes are projected, but timing differences are still meaningful.
    if events.hit_idx is not None:
        lo = max(0, events.hit_idx - int(fps * 0.5))
        hi = events.hit_idx + 1
        shoulder_ang = _line_angle_series(frames, "left_shoulder", "right_shoulder", lo, hi)
        hip_ang = _line_angle_series(frames, "left_hip", "right_hip", lo, hi)
        if shoulder_ang is not None and hip_ang is not None:
            s_peak = lo + int(np.nanargmax(np.abs(np.gradient(shoulder_ang))))
            h_peak = lo + int(np.nanargmax(np.abs(np.gradient(hip_ang))))
            m.hip_lead_ms = round(frames[s_peak].t_ms - frames[h_peak].t_ms, 1)

    # Rounding proxy (sagittal wrist deviation from straight reach-back → release line)
    if (
        events.reach_back_idx is not None
        and events.release_idx is not None
        and events.reach_back_idx < events.release_idx
    ):
        rb = frames[events.reach_back_idx].landmarks.get(throw_wrist)
        rl = frames[events.release_idx].landmarks.get(throw_wrist)
        if rb and rl:
            x1, y1 = rb[0], rb[1]
            x2, y2 = rl[0], rl[1]
            max_dev = 0.0
            for i in range(events.reach_back_idx, events.release_idx + 1):
                p = frames[i].landmarks.get(throw_wrist)
                if not p:
                    continue
                dev = _point_line_distance(p[0], p[1], x1, y1, x2, y2)
                if dev > max_dev:
                    max_dev = dev
            m.wrist_path_sagittal_deviation_px = round(max_dev, 2)

    # Flags
    if m.plant_ankle_x_stddev_px is not None and m.plant_ankle_x_stddev_px > thresholds["plant_ankle_px_stddev_flag"]:
        m.flags.append("weak_plant")
    if m.reach_back_extent_ratio is not None and m.reach_back_extent_ratio < thresholds["reach_back_min_ratio"]:
        m.flags.append("shallow_reach_back")
    if m.hip_lead_ms is not None and m.hip_lead_ms < -thresholds["shoulder_leads_hips_ms_flag"]:
        m.flags.append("early_shoulders")
    if (
        m.wrist_path_sagittal_deviation_px is not None
        and m.wrist_path_sagittal_deviation_px > thresholds["rounding_wrist_deviation_px"]
    ):
        m.flags.append("rounding_suspected_side_view_proxy")

    return m


def _line_angle_series(
    frames: list[FramePose], a: str, b: str, lo: int, hi: int
) -> np.ndarray | None:
    out = []
    for i in range(lo, hi):
        la = frames[i].landmarks.get(a)
        lb = frames[i].landmarks.get(b)
        if not la or not lb:
            out.append(np.nan)
            continue
        out.append(math.degrees(math.atan2(lb[1] - la[1], lb[0] - la[0])))
    arr = _interp_nan(np.array(out))
    if not np.isfinite(arr).all():
        return None
    return arr


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / max(1e-6, den)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def annotate_video(
    src: Path, dst: Path, frames: list[FramePose], events: Events, fps: float
) -> None:
    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))

    event_labels = {
        events.plant_idx: "PLANT",
        events.reach_back_idx: "REACH-BACK",
        events.hit_idx: "HIT",
        events.release_idx: "RELEASE",
    }

    # Build a list of landmark lists in MediaPipe's expected form for drawing.
    # We saved smoothed pixel coords; re-normalize to draw with mediapipe's utility.
    for i in range(len(frames)):
        ok, img = cap.read()
        if not ok:
            break
        # Draw skeleton manually since our landmarks are already pixel-space.
        _draw_skeleton(img, frames[i].landmarks)

        # Top-left HUD
        hud = [
            f"t={frames[i].t_ms:6.0f}ms  frame={i}",
            f"wrist={events.throw_wrist}  facing={events.facing_direction}",
        ]
        y = 24
        for line in hud:
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y += 22

        # Event label if this frame is a keyframe
        if i in event_labels and event_labels[i] is not None:
            label = event_labels[i]
            cv2.rectangle(img, (0, h - 60), (w, h), (0, 0, 0), -1)
            cv2.putText(img, label, (14, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        writer.write(img)

    cap.release()
    writer.release()


_SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_foot_index"),
]


def _draw_skeleton(img, landmarks: dict[str, tuple[float, float, float]]) -> None:
    for a, b in _SKELETON_EDGES:
        if a in landmarks and b in landmarks:
            ax, ay, _ = landmarks[a]
            bx, by, _ = landmarks[b]
            cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 0), 2, cv2.LINE_AA)
    for name, (x, y, v) in landmarks.items():
        color = (0, 255, 255) if name.endswith("wrist") else (0, 180, 255)
        cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_event_snapshots(src: Path, events: Events, stem: str) -> None:
    """Save a PNG per detected key event into metrics/<stem>_frames/."""
    event_frames = {
        "plant": events.plant_idx,
        "reach_back": events.reach_back_idx,
        "hit": events.hit_idx,
        "release": events.release_idx,
    }
    out_dir = ROOT / "metrics" / f"{stem}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
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
        fps=round(fps, 3),
        frame_count=frame_count,
        width=width,
        height=height,
        notes=notes,
        events=events,
        metrics=metrics,
    )

    # Per-clip metrics JSON
    metrics_path = ROOT / "metrics" / f"{clip.stem}.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    # Append row to history.jsonl (metrics only + clip id + date)
    history_path = ROOT / "history.jsonl"
    history_row = {
        "clip": clip.name,
        "run_at": run_at,
        "fps": result.fps,
        "metrics": asdict(metrics),
        "events": {k: getattr(events, k) for k in ("plant_idx", "reach_back_idx", "hit_idx", "release_idx", "throw_wrist", "facing_direction")},
        "notes": notes,
    }
    with history_path.open("a") as f:
        f.write(json.dumps(history_row, default=str) + "\n")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze a disc golf tee shot video.")
    ap.add_argument("clip", type=Path, help="path to the video file in clips/")
    ap.add_argument("--notes", default="", help="optional session notes to log")
    ap.add_argument("--no-annotate", action="store_true", help="skip writing the annotated video (faster iteration)")
    args = ap.parse_args()

    clip: Path = args.clip
    if not clip.exists():
        print(f"clip not found: {clip}", file=sys.stderr)
        return 2

    cfg = load_config()
    print(f"[1/7] probing video: {clip.name}")
    fps, frame_count, w, h = probe_video(clip)
    print(f"      {w}x{h} @ {fps:.2f}fps, {frame_count} frames")

    print("[2/7] running MediaPipe Pose...")
    frames = run_pose(clip, cfg)
    detected = sum(1 for f in frames if f.landmarks)
    print(f"      landmarks detected on {detected}/{len(frames)} frames")

    print("[3/7] smoothing landmarks...")
    frames = smooth_landmarks(frames, cfg)

    print("[4/7] detecting handedness + facing...")
    handedness = detect_handedness(frames, cfg["player"].get("handedness", "auto"))
    print(f"      throwing hand: {handedness}")

    print("[5/7] detecting events + computing metrics...")
    events = detect_events(frames, fps, handedness)
    metrics = compute_metrics(frames, events, fps, h, cfg)

    if not args.no_annotate:
        dst = ROOT / "annotated" / f"{clip.stem}.mp4"
        dst.parent.mkdir(exist_ok=True)
        print(f"[6/7] writing annotated video: {dst}")
        annotate_video(clip, dst, frames, events, fps)
    else:
        print("[6/7] skipping annotation (--no-annotate)")

    print("[7/7] writing metrics + history + event snapshots...")
    result = write_outputs(clip, fps, frame_count, w, h, events, metrics, args.notes)
    save_event_snapshots(clip, events, clip.stem)

    print("\n=== summary ===")
    print(f"events: plant@{metrics.plant_t_ms}ms  reachback@{metrics.reach_back_t_ms}ms  "
          f"hit@{metrics.hit_t_ms}ms  release@{metrics.release_t_ms}ms")
    print(f"flags: {', '.join(metrics.flags) or 'none'}")
    print(f"\nmetrics/{clip.stem}.json and history.jsonl updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
