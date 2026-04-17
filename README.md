# Disc Golf Tee Shot Analyzer

A local, privacy-respecting pipeline that takes a side-view clip of a backhand
disc golf throw and produces quantitative metrics (plant stability, tempo
between throw phases, motion-spike character at hit) plus an annotated video
and event keyframes — then feeds them to a Claude "coach" that produces
grounded, specific feedback.

Designed for two audiences:

1. **Claude Code / Cowork users** — drop a clip into `clips/`, ask Claude to
   analyze it, get coaching. Claude runs the pipeline for you. See
   `CLAUDE.md` for the agent-side conventions.
2. **Python / CLI users** — run the pipeline yourself and read the JSON.

## Two pipelines

This repo ships two analyzers. They share the event/metrics JSON shape but
take very different paths to get there:

### `scripts/analyze_throw_cv.py` — active (OpenCV, silhouette + motion)

- Background-subtraction + silhouette extraction (OpenCV MOG2)
- Bbox-restricted motion magnitude per frame (frame-diff inside the silhouette
  + 20% margin — blocks camera/background jitter from dominating the signal)
- ORB-based camera-motion estimation; affine-warp stabilization if the clip
  has >50 px drift (tripod-panning or handheld)
- Event detection: `setup`, `plant`, `reach-back`, `power-pocket`, `hit`,
  `release` — with a rolling-minimum motion baseline and half-prominence
  spike-width (±0.75s bound around hit)
- Emits `metrics/<clip>.cv.json`, an annotated overlay video, and per-event
  keyframe stills

This is what coaching runs through today. It's **silhouette + motion only** —
no joint-angle data. That means it's good at tempo, plant stability, and spike
character; it is *not* reliable for rounding or shoulder-hip separation (those
need down-the-line / face-on angles and/or pose estimation).

### `scripts/analyze_throw.py` — experimental (MediaPipe Pose, WIP)

The aspirational version: extract 33 body landmarks per frame and compute
joint angles (shoulder-hip separation, wrist trajectory, elbow path, knee
bend) directly. Currently not reliably working end-to-end — tracked in the
repo as the starting point for the pose-based rewrite. Do not use for
coaching until it's fixed.

## Install

Requires Python 3.10+ and `ffmpeg` on PATH.

```bash
git clone <your-fork-url> disc-golf-analyzer
cd disc-golf-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then edit with your profile
```

macOS ffmpeg: `brew install ffmpeg`. Ubuntu: `apt install ffmpeg`.

> `mediapipe` and `scipy` are only needed for `analyze_throw.py` (the WIP pose
> pipeline). If you hit an install error and only want the cv2 pipeline, you
> can comment those lines out of `requirements.txt`.

## CLI usage

Drop a side-view clip into `clips/`, then:

```bash
python scripts/analyze_throw_cv.py clips/my-throw.mov
```

Outputs:

- `metrics/my-throw.cv.json` — structured metrics and event frame indices
- `annotated/my-throw.cv.mp4` — overlay video with event labels, motion meter,
  bbox, and tempo markers (suppress with `--no-annotate`)
- `metrics/my-throw_frames/` — PNG stills for setup, plant, reach-back, power
  pocket, hit, release

Additional stills (arbitrary frame indices, e.g. for a tempo-timeline montage):

```bash
python scripts/extract_keyframes.py clips/my-throw.mov 29 163 172 195 208
```

### Event overrides

When the auto-detector picks the wrong frame for an event (common on fast
throws with little post-release footage, or heavily panned clips), pin the
event manually:

```bash
python scripts/analyze_throw_cv.py clips/my-throw.mov \
  --setup 29 --plant 163 --reach-back 172 --power-pocket 195 --hit 208
```

Any flag explicitly set overrides the auto-detector for that one event;
unspecified events are still auto-detected.

### iPhone slo-mo clips

If an HEVC `.MOV` refuses to open, transcode first:

```bash
ffmpeg -i input.MOV -c:v libx264 -crf 18 -preset veryfast -c:a aac output.mp4
```

## Metrics reference

Key fields in `metrics/<clip>.cv.json`:

| Field | Meaning |
|---|---|
| `events.setup_idx` ... `events.release_idx` | Frame indices for each throw phase |
| `metrics.reach_back_to_hit_ms` | Pull duration. Elite reference: 250–400 ms |
| `metrics.power_pocket_to_hit_ms` | Snap duration. Elite reference: 150–250 ms |
| `metrics.baseline_motion_mag` | Rolling-min motion floor (ambient noise) |
| `metrics.peak_motion_mag` | Motion magnitude at hit |
| `metrics.hit_to_baseline_ratio` | Spike height / floor — a real hit is >5× |
| `metrics.spike_width_ms` | Width of the hit spike at half-prominence |
| `metrics.plant_foot_y_stddev` | Front-foot vertical pixel stddev across hit window — plant stability proxy |
| `metrics.cam_max_drift_px` | Max ORB-estimated camera translation (0 = tripod) |
| `metrics.stabilization_applied` | True if the clip was warped before analysis |

## Project layout

```
scripts/
  analyze_throw_cv.py    # active pipeline (run this)
  analyze_throw.py       # WIP pose-based pipeline
  extract_keyframes.py   # pull arbitrary frames from a clip
config.example.yaml      # copy to config.yaml and edit
COACH.md                 # coaching prompt (drives Claude's voice)
CLAUDE.md                # Cowork/Claude-Code agent instructions
kb/                      # cached coaching knowledge + drills
clips/                   # input clips (gitignored)
metrics/                 # output JSON + event PNGs (gitignored)
annotated/               # overlay videos (gitignored)
keyframes/               # extra stills (gitignored)
```

Pipeline-generated outputs and your personal `config.yaml` / `history.jsonl`
are gitignored. See `.gitignore`.

## Camera angle notes

Side view is a solid all-rounder but has blind spots:

- **Rounding** → best from down-the-line (behind thrower, facing target)
- **Plant spin** → best face-on
- **X-factor magnitude** → best from down-the-line

The cv2 pipeline still reports best-effort metrics from side view; the coach
will flag when a session needs a different angle.

## Known limitations (cv2 pipeline)

- **Hit detection on fast throws.** If the clip ends soon after release,
  follow-through motion can be larger than the release spike and get picked as
  "hit". Symptom: `reach_back_to_hit_ms` or `power_pocket_to_hit_ms` looks
  implausible (e.g. 80 ms or 1400 ms for a clean throw). Workaround: eyeball
  the keyframes and pass `--hit N` and any other wrong events on the command
  line.
- **Stabilization edge artifacts.** On heavily panned clips, ORB-warp +
  `BORDER_REPLICATE` creates a stretched strip along one edge that MOG2 picks
  up as foreground and inflates the silhouette bbox. If
  `plant_foot_y_stddev` is high but the plant looks clean visually, suspect
  artifacts rather than a real settling issue.
- **No pose data.** Rounding, early shoulders, grip flaws, shoulder-hip
  separation — all need human-eye confirmation from the annotated video or a
  down-the-line angle. The pose pipeline (`analyze_throw.py`) is meant to
  close this gap once it's working.
- **Backhand focus.** Metrics and the KB are tuned for backhand drives.
  Forehand throws will produce numbers but the event-detection heuristics and
  coaching cues assume backhand mechanics.

## License

MIT — see `LICENSE`.
