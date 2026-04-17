"""Flask app for the local Disc Golf Analyzer web UI.

Routes are grouped by surface:
  - Page routes (server-rendered): /, /setup, /analyze/<clip>, /results/<clip>,
    /override/<clip>
  - JSON APIs (consumed by page JS): /api/config, /api/upload, /api/analyze
  - Static file serving: /clips/<f>, /annotated/<f>, /metrics/<path>

The app holds no state itself. All persistent state lives on disk:
  config.yaml, clips/, metrics/, annotated/, history.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
    url_for,
)
from werkzeug.utils import secure_filename

from web.pipeline_runner import PROJECT_ROOT, run_pipeline

CLIPS_DIR = PROJECT_ROOT / "clips"
METRICS_DIR = PROJECT_ROOT / "metrics"
ANNOTATED_DIR = PROJECT_ROOT / "annotated"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config.example.yaml"

ALLOWED_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
VALID_PIPELINES = ("cv2", "pose")


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB per upload

    # ------------------------------------------------------------------
    # Page routes — stub implementations. Filled in by subsequent tasks.
    # ------------------------------------------------------------------
    @app.get("/")
    def landing() -> str:
        return render_template(
            "landing.html",
            config_exists=CONFIG_PATH.exists(),
            clips=list_clips(),
        )

    @app.get("/setup")
    def setup() -> str:
        existing = load_config() or {}
        return render_template("setup.html", existing=existing)

    @app.get("/analyze/<path:clip>")
    def analyze_page(clip: str) -> str:
        clip_path = safe_clip_path(clip)
        return render_template(
            "analyze.html",
            clip_name=clip_path.name,
            existing_pipelines=pipelines_with_output(clip_path.stem),
        )

    @app.get("/results/<path:clip>")
    def results_page(clip: str) -> str:
        clip_path = safe_clip_path(clip)
        available = pipelines_with_output(clip_path.stem)
        if not available:
            abort(404, description=f"no analysis output found for {clip_path.name}")
        player_level = (load_config() or {}).get("player", {}).get("level", "recreational")
        return render_template(
            "results.html",
            clip_name=clip_path.name,
            clip_stem=clip_path.stem,
            available=available,
            metrics_by_pipeline={
                p: load_metrics(clip_path.stem, p) for p in available
            },
            rows_by_pipeline={
                p: metrics_rows(p, load_metrics(clip_path.stem, p), player_level)
                for p in available
            },
            keyframes_by_pipeline={
                p: list_keyframes(clip_path.stem, p) for p in available
            },
        )

    @app.get("/override/<path:clip>")
    def override_page(clip: str) -> str:
        clip_path = safe_clip_path(clip)
        # Prefer cv2 metrics for fps / current events; fall back to pose.
        available = pipelines_with_output(clip_path.stem)
        preferred = available[0] if available else None
        metrics = load_metrics(clip_path.stem, preferred) if preferred else {}
        return render_template(
            "override.html",
            clip_name=clip_path.name,
            clip_stem=clip_path.stem,
            available=available,
            preferred=preferred,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # JSON APIs
    # ------------------------------------------------------------------
    @app.post("/api/config")
    def api_config() -> Response:
        payload = request.get_json(force=True)
        merged = merge_config(payload)
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)
        return jsonify({"ok": True})

    @app.post("/api/upload")
    def api_upload() -> Response:
        if "file" not in request.files:
            abort(400, description="no file field in form")
        f = request.files["file"]
        if not f.filename:
            abort(400, description="empty filename")
        filename = secure_filename(f.filename)
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            abort(400, description=f"unsupported extension: {ext}")
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        dest = CLIPS_DIR / filename
        f.save(str(dest))
        return jsonify(
            {
                "ok": True,
                "clip": filename,
                "analyze_url": url_for("analyze_page", clip=filename),
            }
        )

    @app.post("/api/analyze")
    def api_analyze() -> Response:
        payload = request.get_json(force=True)
        clip_name = payload.get("clip", "")
        pipelines = payload.get("pipelines", ["cv2"])
        overrides = payload.get("overrides") or {}
        notes = payload.get("notes") or ""

        clip_path = safe_clip_path(clip_name)
        for p in pipelines:
            if p not in VALID_PIPELINES:
                abort(400, description=f"unknown pipeline: {p!r}")

        @stream_with_context
        def generate():
            for p in pipelines:
                yield f"\n=== running {p} pipeline ===\n"
                for line in run_pipeline(
                    p, clip_path, overrides=overrides, notes=notes
                ):
                    yield line
            yield "\n=== all done ===\n"

        return Response(generate(), mimetype="text/plain")

    # ------------------------------------------------------------------
    # Static file serving for pipeline artifacts (outside web/static)
    # ------------------------------------------------------------------
    @app.get("/clips/<path:filename>")
    def serve_clip(filename: str) -> Response:
        return send_from_directory(CLIPS_DIR, filename)

    @app.get("/annotated/<path:filename>")
    def serve_annotated(filename: str) -> Response:
        return send_from_directory(ANNOTATED_DIR, filename)

    @app.get("/metrics-file/<path:filename>")
    def serve_metrics_file(filename: str) -> Response:
        # /metrics is used by Flask conventions; namespace under /metrics-file.
        return send_from_directory(METRICS_DIR, filename)

    return app


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def safe_clip_path(clip: str) -> Path:
    """Resolve a user-supplied clip name to an absolute path inside clips/.

    Rejects path traversal, absolute paths, and non-existent files.
    """
    if not clip or "/" in clip or "\\" in clip or clip.startswith("."):
        abort(400, description="invalid clip name")
    path = (CLIPS_DIR / clip).resolve()
    if CLIPS_DIR.resolve() not in path.parents and path.parent != CLIPS_DIR.resolve():
        abort(400, description="clip escapes clips/ dir")
    if not path.exists():
        abort(404, description=f"clip not found: {clip}")
    return path


def list_clips() -> list[dict[str, Any]]:
    if not CLIPS_DIR.exists():
        return []
    entries = []
    for p in sorted(CLIPS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            stem = p.stem
            entries.append(
                {
                    "name": p.name,
                    "stem": stem,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    "analyzed_pipelines": pipelines_with_output(stem),
                }
            )
    return entries


def pipelines_with_output(stem: str) -> list[str]:
    present = []
    if (METRICS_DIR / f"{stem}.cv.json").exists():
        present.append("cv2")
    if (METRICS_DIR / f"{stem}.pose.json").exists():
        present.append("pose")
    return present


def load_metrics(stem: str, pipeline: str) -> dict[str, Any]:
    if pipeline == "cv2":
        path = METRICS_DIR / f"{stem}.cv.json"
    elif pipeline == "pose":
        path = METRICS_DIR / f"{stem}.pose.json"
    else:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config() -> dict[str, Any] | None:
    for p in (CONFIG_PATH, CONFIG_EXAMPLE_PATH):
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return None


# --------------------------------------------------------------------------
# Metrics rendering — pipeline-aware schema + threshold-based coloring.
# --------------------------------------------------------------------------

# Per-pipeline metric schemas. Each entry:
#   key: field name inside `metrics.*`
#   label: human-readable row label
#   unit: suffix for display ("ms", "px", "", etc.)
#   classifier: fn(value, player_level) -> "ok" | "warn" | "err" | None
#
# Keep these calibrated to recreational by default; the coach tightens for
# intermediate / advanced.

LEVEL_TIGHTEN = {"brand_new": 1.4, "recreational": 1.0, "intermediate": 0.8, "advanced": 0.65}


def _band_classifier(
    ok_band: tuple[float, float],
    warn_band: tuple[float, float],
):
    def cls(value, level):
        if value is None:
            return None
        t = LEVEL_TIGHTEN.get(level, 1.0)
        ok_lo, ok_hi = ok_band[0] * t, ok_band[1] * (1 / t if t > 0 else 1)
        warn_lo, warn_hi = warn_band
        if ok_lo <= value <= ok_hi:
            return "ok"
        if warn_lo <= value <= warn_hi:
            return "warn"
        return "err"
    return cls


def _gt_classifier(ok_min: float, warn_min: float):
    """Classifier: higher is better (e.g. hit_to_baseline_ratio)."""
    def cls(value, _level):
        if value is None:
            return None
        if value >= ok_min:
            return "ok"
        if value >= warn_min:
            return "warn"
        return "err"
    return cls


def _lt_classifier(ok_max: float, warn_max: float):
    """Classifier: lower is better (e.g. plant stddev)."""
    def cls(value, _level):
        if value is None:
            return None
        if value <= ok_max:
            return "ok"
        if value <= warn_max:
            return "warn"
        return "err"
    return cls


METRIC_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "cv2": [
        {"key": "reach_back_to_hit_ms", "label": "Reach-back → hit", "unit": "ms",
         "cls": _band_classifier((250, 600), (150, 1200))},
        {"key": "power_pocket_to_hit_ms", "label": "Power pocket → hit (snap)", "unit": "ms",
         "cls": _band_classifier((150, 300), (100, 500))},
        {"key": "hit_to_baseline_ratio", "label": "Hit / baseline ratio", "unit": "×",
         "cls": _gt_classifier(10, 5)},
        {"key": "spike_width_ms", "label": "Hit spike width", "unit": "ms", "cls": None},
        {"key": "plant_foot_y_stddev", "label": "Plant foot y stddev", "unit": "px",
         "cls": _lt_classifier(2.5, 5.0)},
        {"key": "baseline_motion_mag", "label": "Baseline motion", "unit": "", "cls": None},
        {"key": "peak_motion_mag", "label": "Peak motion", "unit": "", "cls": None},
        {"key": "cam_max_drift_px", "label": "Camera drift (ORB)", "unit": "px", "cls": None},
        {"key": "stabilization_applied", "label": "Stabilization applied", "unit": "", "cls": None},
    ],
    "pose": [
        {"key": "plant_to_release_ms", "label": "Plant → release (total)", "unit": "ms", "cls": None},
        {"key": "reach_back_to_hit_ms", "label": "Reach-back → hit", "unit": "ms",
         "cls": _band_classifier((250, 700), (150, 1200))},
        {"key": "reach_back_to_power_pocket_ms", "label": "Reach-back → power pocket", "unit": "ms",
         "cls": None},
        {"key": "power_pocket_to_hit_ms", "label": "Power pocket → hit (snap)", "unit": "ms",
         "cls": _band_classifier((150, 300), (100, 500))},
        {"key": "reach_back_extent_ratio", "label": "Reach-back extent / torso", "unit": "",
         "cls": _band_classifier((0.55, 0.80), (0.40, 1.10))},
        {"key": "hip_lead_ms", "label": "Hip lead over shoulders", "unit": "ms",
         "cls": _gt_classifier(80, 30)},
        {"key": "spine_lean_deg_at_hit", "label": "Spine lean at hit", "unit": "°",
         "cls": _band_classifier((-5, 15), (-15, 30))},
        {"key": "release_height_norm", "label": "Release height / torso", "unit": "",
         "cls": _band_classifier((0.35, 0.75), (0.20, 0.90))},
        {"key": "wrist_path_sagittal_deviation_px", "label": "Wrist path deviation", "unit": "px",
         "cls": _lt_classifier(25, 60)},
        {"key": "plant_ankle_x_stddev_px", "label": "Plant ankle x stddev", "unit": "px",
         "cls": _lt_classifier(6, 14)},
        {"key": "plant_ankle_y_stddev_px", "label": "Plant ankle y stddev", "unit": "px",
         "cls": _lt_classifier(2.5, 5.0)},
    ],
}


def _format_value(value: Any, unit: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            fmt = f"{value:.1f}" if abs(value) < 1000 else f"{value:.0f}"
        else:
            fmt = str(value)
        return f"{fmt} {unit}".rstrip()
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "—"
    return str(value)


def metrics_rows(
    pipeline: str, metrics_json: dict[str, Any], player_level: str
) -> list[dict[str, str]]:
    """Flatten a pipeline's metrics JSON into rendered rows for the results table."""
    schema = METRIC_SCHEMA.get(pipeline, [])
    m = (metrics_json or {}).get("metrics", {})
    flags = m.get("flags") or []
    rows: list[dict[str, str]] = []
    for field in schema:
        raw = m.get(field["key"])
        status = field["cls"](raw, player_level) if field["cls"] else None
        rows.append(
            {
                "key": field["key"],
                "label": field["label"],
                "value": _format_value(raw, field["unit"]),
                "status": status or "",
            }
        )
    if flags:
        rows.append(
            {
                "key": "flags",
                "label": "Flags",
                "value": ", ".join(flags),
                "status": "warn",
            }
        )
    return rows


def list_keyframes(stem: str, pipeline: str) -> list[dict[str, str]]:
    """Return [{event, src}, ...] for the PNGs saved per event."""
    if pipeline == "cv2":
        frames_dir = METRICS_DIR / f"{stem}_frames"
    else:
        frames_dir = METRICS_DIR / f"{stem}.pose_frames"
    if not frames_dir.is_dir():
        return []
    entries: list[dict[str, str]] = []
    order = ["setup", "plant", "reach_back", "power_pocket", "hit", "release"]
    files = {p.stem: p for p in frames_dir.iterdir() if p.suffix.lower() == ".png"}
    for event in order:
        # Match files like "hit.png", "hit_264.png", "03_hit.png", etc.
        for stem_name, path in files.items():
            if event in stem_name:
                rel = path.relative_to(METRICS_DIR).as_posix()
                entries.append({"event": event, "src": f"/metrics-file/{rel}"})
                break
    return entries


def merge_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge a wizard submission into the example config structure.

    We start from config.example.yaml's shape so we don't drop pipeline tuning
    defaults when writing the user's profile back.
    """
    base = load_config() or {}
    # Start from example to preserve pipeline/coaching defaults if config.yaml
    # is missing those sections.
    if CONFIG_EXAMPLE_PATH.exists():
        with CONFIG_EXAMPLE_PATH.open("r", encoding="utf-8") as f:
            example = yaml.safe_load(f) or {}
        for section, defaults in example.items():
            base.setdefault(section, defaults)

    player_updates = updates.get("player") or {}
    base.setdefault("player", {})
    base["player"].update(player_updates)
    return base


# Module-level app instance for `from web import app` imports.
app = create_app()
