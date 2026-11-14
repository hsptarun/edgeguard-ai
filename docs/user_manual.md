# User manual

**Start:** `./scripts/run.sh` → browser opens at http://localhost:8501. Sidebar shows database/model status. Stop with Ctrl+C in the terminal.

**Overview tab** - counts (events, normal, suspicious, active alerts) from the database, room cards (restricted/armed status for the *dashboard clock*, simulated door/motion), latest alerts. Empty until events exist. Use the sidebar "custom now" to see rooms as they would look at 03:00.

**Simulator tab** - pick a scenario (and optionally a room/time) → *Generate event*. *Generate 10 mixed* creates random events. *Start auto-run* produces one event every 2 s until paused. *Reset engine memory* clears cooldowns; *Wipe history* (tick confirm) deletes stored events/alerts.

**Video Monitor tab** - 1) choose the room the camera covers; 2) set confidence, frame stride and the monitored zone (fractions of the frame; red box = person's feet inside zone); 3) enter the assumed date/time at video start (alerts depend on it); 4) upload the video and press *Process video*. Annotated frames stream live; alert snapshots and measured timing appear afterwards. The upload is deleted after processing.

**Alerts & History tab** - filter by date, room, severity, type, status. Select an alert to read *why* it was raised, the rule and the evidence. Buttons: *Acknowledge*, *Mark as false alarm*, *Confirm as incident* (a human decision). CSV export available.

**Building Setup tab** - edit building hours, force-closed override, cooldown/escalation/persistence settings, per-room restriction, permitted hours (`09:00-17:00; 19:00-21:00`), simulated door state, motion sensor; add/remove rooms. *Apply* affects this session; *Save to file* persists to `config/building_config.json`.

**Evaluation tab** - *Run synthetic-dataset evaluation* computes metrics and a confusion matrix and writes files to `reports/`. Detector speed shows after a video run.

**Responsible AI tab** - privacy, limits, licence and terminology.
