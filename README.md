# Disc Golf Tee Shot Analyzer

Drop in a side-view clip of your backhand drive. Get quantitative metrics
(plant stability, throw tempo, reach-back depth, snap timing, camera drift
compensation) plus an annotated overlay video and event keyframes. Optionally,
hand the whole thing to an AI and get grounded, specific coaching feedback
instead of vague "work on your form" platitudes.

Runs entirely locally. Nothing leaves your machine.

---

## Quickstart

Three supported workflows. Pick one.

- **A. AI coach** — drop a clip, get natural-language coaching. The richest output.
- **B. Local GUI** — upload, analyze, and view results in your browser. No AI, but no shell either.
- **C. CLI** — you run `python`, you read JSON.

### A. Use with an AI coach (recommended)

1. **Set up the repo once** — see [Setup](#setup).
2. **Record your throw.** Side-on, 30 fps or better, whole body in frame from
   setup through follow-through. Drop the file into `clips/`.
3. **Hand the repo to an AI that can read files and run Python.** A few
   concrete options:
   - **Claude Code** (terminal agent):
     ```bash
     cd disc-golf-analyzer
     claude
     ```
     It reads `CLAUDE.md` and `COACH.md` on its own.
   - **Cowork / Claude desktop app**: open this folder in a Project. Same
     auto-pickup of `CLAUDE.md` / `COACH.md`.
   - **Any other AI with file access** (ChatGPT Code Interpreter, Cursor,
     Codex, Continue, etc.): point it at `CLAUDE.md` and `COACH.md` manually
     and tell it to follow them.
4. **Ask for analysis.** Example:
   > I just dropped `my-throw.mov` in `clips/`. Run it and coach me.
5. **What the AI will do**, per `COACH.md`:
   - Run `scripts/analyze_throw_cv.py clips/my-throw.mov`.
   - Read `metrics/my-throw.cv.json` and any relevant `kb/` entries.
   - Produce **What I see** → **This session's fix** → **Drill**, grounded
     in specific metric values and tied to your configured level.

On first run, if `config.yaml` doesn't exist, the AI will interview you to
build your player profile (handedness, distance range, level, known issues).
One question at a time — takes two minutes.

### B. Local GUI (no AI, no shell)

1. **Set up the repo once** — see [Setup](#setup).
2. **Start the local server**:
   ```bash
   python scripts/serve.py
   ```
   This boots a Flask app on `http://127.0.0.1:8765` and opens your browser.
   Nothing is hosted externally; the app only listens on localhost.
3. **Walk the wizard.** First run asks for your profile (handedness, distance
   range, level, known issues) and writes it to `config.yaml`.
4. **Upload a clip.** Drag-and-drop into the dropzone, or click to pick a
   file. Accepts `.mov`, `.mp4`, `.m4v`, `.avi`, `.mkv`.
5. **Analyze.** Pick one or both pipelines (cv2 is fast; pose adds joint
   angles). Pipeline output streams live to the page.
6. **Scrub the results.** The results page shows the annotated video with
   clickable event chapter markers, a metrics table with traffic-light
   coloring tuned to your level, event keyframes, and a "Copy metrics JSON"
   button for pasting into any AI coach. If the auto-detected events look
   wrong, the **Events look wrong?** link opens a frame-accurate scrubber
   (keys `,` `.` `j` `k` `l`) where you can pin events by eye and re-run.

### C. Pure CLI (no AI, no UI)

1. **Set up the repo once** — see [Setup](#setup).
2. **Drop a clip** into `clips/`.
3. **Run the pipeline**:
   ```bash
   python scripts/analyze_throw_cv.py clips/my-throw.mov
   ```
4. **Read the outputs**:
   - `metrics/my-throw.cv.json` — structured metrics (see [Outputs](#outputs))
   - `annotated/my-throw.cv.mp4` — overlay video with event labels, motion
     meter, bbox, and tempo markers
   - `metrics/my-throw_frames/` — event keyframe PNGs

---

## Setup

Requires Python 3.10+ and `ffmpeg` on `PATH`. macOS: `brew install ffmpeg`.
Ubuntu: `apt install ffmpeg`.

```bash
git clone https://github.com/daitkenbacon/disc-golf-analyzer.git
cd disc-golf-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml    # edit with your profile (or let the AI interview you)
```

> `mediapipe` and `scipy` are only needed for the pose pipeline
> (`analyze_throw.py`). If you only want the cv2 pipeline, you can comment
> those lines out of `requirements.txt`.

---

## The two pipelines

This repo ships two analyzers. They share the event-and-metrics JSON shape
but take different paths to get there.

### `scripts/analyze_throw_cv.py` — cv2 pipeline (default)

Silhouette + motion, no joint angles. What it does:

- MOG2 background subtraction + bbox-restricted frame-diff motion magnitude
- ORB-based camera-motion estimation; affine-warp stabilization on panned clips
- Event detection (`setup`, `plant`, `reach-back`, `power-pocket`, `hit`,
  `release`) from rolling-minimum motion baseline + spike prominence
- Emits `metrics/<clip>.cv.json`, `annotated/<clip>.cv.mp4`,
  `metrics/<clip>_frames/`

**Strongest signals**: tempo, plant stability, hit-spike character.
**Won't tell you about**: rounding, shoulder-hip separation, grip — no joint
data. The coach (`COACH.md`) flags when a session needs a different camera
angle.

### `scripts/analyze_throw.py` — pose pipeline

MediaPipe Pose, 33 body landmarks per frame. Same events, but detected from
wrist kinematics; additional joint-angle metrics (reach-back extent vs torso
length, hip-vs-shoulder lead, spine lean at hit, release height normalized to
torso, wrist-path sagittal deviation). Emits `metrics/<clip>.pose.json`,
`annotated/<clip>.pose.mp4`, `metrics/<clip>.pose_frames/` — namespaced so it
coexists with cv2 outputs.

### Which to use

| Goal | Pipeline |
|---|---|
| Tempo, plant stability, general coaching | cv2 |
| Reach-back depth, hip lead, spine lean, release height | pose |
| You're running coaching through `COACH.md` | cv2 (default) |
| You want both — run both | both (outputs are separately namespaced) |

Command shape is the same for both scripts (including event overrides below).

---

## Outputs

### Metrics JSON (cv2)

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

### Metrics JSON (pose)

Additional fields in `metrics/<clip>.pose.json`:

| Field | Meaning |
|---|---|
| `metrics.reach_back_extent_ratio` | Wrist distance behind hip-midline / torso length. Target 0.55–0.80; >1.0 suggests over-reach |
| `metrics.hip_lead_ms` | How many ms hips rotate before shoulders. Positive = correct kinetic chain |
| `metrics.spine_lean_deg_at_hit` | Forward/back spine lean at hit. Strong negative = leaning back at release |
| `metrics.release_height_norm` | Wrist height at release, normalized to torso (0 = hip, 1 = shoulder) |
| `metrics.wrist_path_sagittal_deviation_px` | How far the wrist deviates from a straight pull-path — rounding proxy |
| `metrics.plant_ankle_x_stddev_px` / `y_stddev_px` | Plant stability, separately for horizontal and vertical wobble |

### Annotated video

Each pipeline writes `annotated/<clip>.<pipeline>.mp4` with event labels
burned in at the right frame, a motion / velocity meter, and bbox or skeleton
overlay. Scrub through it when you want to ground-truth event detection.

### Event keyframes

Stills for setup, plant, reach-back, power-pocket, hit, release are written
to `metrics/<clip>_frames/` (cv2) or `metrics/<clip>.pose_frames/` (pose).

Arbitrary extra stills:

```bash
python scripts/extract_keyframes.py clips/my-throw.mov 29 163 172 195 208
```

---

## Troubleshooting

### Auto-detected events look wrong

Common. Fast throws with little post-release footage, or clips with heavy
camera panning, can fool the hit detector into picking a follow-through motion
peak. Symptom: `reach_back_to_hit_ms` or `power_pocket_to_hit_ms` outside the
500–1500 ms range for a normal recreational throw.

Pin any event manually — unspecified events are still auto-detected:

```bash
python scripts/analyze_throw_cv.py clips/my-throw.mov \
  --setup 29 --plant 163 --reach-back 172 --power-pocket 195 --hit 208 --release 215
```

Same flags work on `analyze_throw.py`.

Easiest way to find the right frame numbers: scrub the annotated video (or
run `extract_keyframes.py` across the clip) and eyeball when each phase
actually happens.

### iPhone HEVC `.MOV` won't open

Transcode to H.264 first:

```bash
ffmpeg -i input.MOV -c:v libx264 -crf 18 -preset veryfast -c:a aac output.mp4
```

### `plant_foot_y_stddev` looks high but plant looked clean

ORB-warp + `BORDER_REPLICATE` stabilization can create a stretched strip
along the panned edge; MOG2 picks it up as foreground and inflates the bbox.
Visual gut-check from the annotated video wins over the metric here — if the
plant looks clean, it probably is.

---

## Camera angles

Side-on is a solid all-rounder but has blind spots:

- **Rounding** → down-the-line (behind thrower, facing target)
- **Plant spin** → face-on
- **X-factor magnitude** → down-the-line

The pipelines still report best-effort metrics from side view; the coach
will flag when a flagged issue genuinely needs a different angle.

---

## Project layout

```
scripts/
  analyze_throw_cv.py    # cv2 pipeline (default for coaching)
  analyze_throw.py       # pose pipeline (MediaPipe-based)
  extract_keyframes.py   # pull arbitrary frames from a clip
  serve.py               # launches the local web UI on 127.0.0.1:8765
web/                     # Flask app + templates + static assets for the GUI
config.example.yaml      # copy to config.yaml and edit
COACH.md                 # coaching prompt (drives the AI's voice)
CLAUDE.md                # agent-side instructions (Cowork / Claude Code)
kb/                      # cached coaching knowledge + drills
clips/                   # input clips (gitignored)
metrics/                 # output JSON + event PNGs (gitignored)
annotated/               # overlay videos (gitignored)
keyframes/               # extra stills (gitignored)
history.jsonl            # trend log accumulated per session (gitignored)
```

Your personal `config.yaml`, `history.jsonl`, and all pipeline outputs are
gitignored. See `.gitignore`.

---

## License

MIT — see `LICENSE`.
