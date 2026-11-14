# EdgeGuard AI: Edge AI-Based Intrusion Detection for Smart Buildings

Software-only project for **CSE276: Artificial Intelligence Foundations**. It combines
**local YOLO person detection on video** with a **simulated smart building** (rooms, doors, motion sensors)
and an **explainable rule-based intrusion engine**, shown in a Streamlit dashboard with SQLite history.

> Person detection ≠ intrusion. A person only raises an alert when a rule says they are somewhere
> restricted at a time they are not allowed to be. Alerts are prompts for human review, never verdicts.

## Objectives
1. Run a pretrained deep-learning detector locally (Edge AI, no cloud, no API keys).
2. Model a building in software with restricted zones and access schedules.
3. Separate *observation* → *rule-based alert* → *human-confirmed incident*.
4. Explain every alert (rule + evidence). Log history. Evaluate honestly.

## Features
- Upload a video → frame-by-frame YOLO person detection with boxes, confidence slider, measured ms/frame and FPS.
- Configurable rooms: restricted flag, permitted hours (incl. overnight), simulated door and motion state, camera zone.
- Rules R1-R4 (person in restricted zone / door open / motion when closed / multiple suspicious events), with cooldown and video persistence to avoid per-frame alert spam.
- Alert detail: reason, rule, evidence, severity (Low/Medium/High), acknowledge / false alarm / confirm-incident controls.
- SQLite event + alert history with filters (date, room, severity, type) and CSV export.
- Sensor simulator (start/pause/reset/single events) - everything labelled **SIMULATED**.
- Labelled synthetic dataset + evaluation (precision, recall, F1, confusion matrix, FPR, detector speed).
- Works in simulation-only mode if the model or a video is unavailable.

## Technology stack
Python · OpenCV (headless) · Ultralytics YOLOv8n (pretrained, COCO, **AGPL-3.0**) · NumPy · Streamlit · SQLite · Pandas · Matplotlib · Pytest.

## Architecture
```
video/sim events → Observation → IntrusionEngine (rules + cooldown) → Alert → SQLite → Streamlit dashboard
        ▲                                ▲
 PersonDetector (YOLO, local)     ZoneManager (rooms, hours, zones)
```
Details: `docs/architecture.md`.

## Installation (macOS, Apple-silicon MacBook)
Prerequisite: Python 3.10-3.13 (`brew install python@3.12` or python.org installer). Internet is needed once for packages and the model.
```bash
cd edgeguard-ai
chmod +x scripts/*.sh        # only if needed
./scripts/setup.sh           # creates .venv, installs deps (~1 GB incl. PyTorch), initialises DB, downloads model
./scripts/run.sh             # opens http://localhost:8501
```
Model only: `source .venv/bin/activate && python scripts/download_model.py` (source: the Ultralytics assets release; ~6.5 MB).
Setup flag `--skip-model` installs everything except the model.

## Usage (short)
1. **Simulator** tab → choose *Door opening after closing time* → *Generate event* → see the alert in **Overview**.
2. **Alerts & History** → inspect the explanation, acknowledge, filter.
3. **Video Monitor** → upload a clip with people, pick the camera's room, adjust the zone, set the *assumed clock time* (e.g. 03:00), *Process video*.
4. **Evaluation** → *Run synthetic-dataset evaluation*.
Full guide: `docs/user_manual.md`. Demo script: `docs/demo_script.md`.

## Getting a sample video
No video is bundled. Use a short clip of people walking filmed by you or friends (with consent), or a video whose licence allows reuse
(e.g. from a source offering CC0 / public-domain clips such as Pexels or Pixabay - **open the page and read the licence yourself**, I cannot guarantee any particular URL). Fixed-camera footage works best. Do not upload videos of people who have not agreed.

## Troubleshooting
| Problem | Fix |
|---|---|
| `Python 3.10-3.13 not found` | `brew install python@3.12`, rerun setup |
| Setup fails during pip | Check internet; rerun `./scripts/setup.sh` |
| Sidebar says model not downloaded | `python scripts/download_model.py` or use the button in the Video Monitor tab |
| "video could not be opened" | Re-encode to H.264 MP4 (`ffmpeg -i in.mov -c:v libx264 out.mp4`) |
| Port busy | `streamlit run app.py --server.port 8502` |
| Permission denied on scripts | `chmod +x scripts/*.sh` |
| Reset everything | Delete `data/edgeguard.db` (recreated on launch) |

## Limitations
- Pretrained COCO detector: misses/false detections under occlusion, low light, tiny people; no accuracy was measured on real footage in this project (no ground-truth annotations).
- Video has no trusted clock; you enter an assumed start time.
- Synthetic evaluation scores check rule logic against the written policy, not real-world performance.
- Door/motion data is simulated. Rules cover restricted rooms only; unrestricted rooms never alert.
- CPU is the default; `mps` is offered only if PyTorch reports it, and was **not benchmarked**.
- Single user, single camera at a time, not a certified security product.
- Ultralytics YOLO is AGPL-3.0; check obligations before redistribution/deployment as a service.
