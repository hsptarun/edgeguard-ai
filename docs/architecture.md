# Architecture

## Modules
| Module | Responsibility |
|---|---|
| `config/settings.py` | Paths, model URL/licence, limits. |
| `detection/zone_manager.py` | `Room`, `BuildingConfig`, `RuleSettings`, `ZoneManager`: schedules (minute resolution, inclusive, overnight-aware), building open/closed, camera zone test, JSON load/save with validation and safe fallback. |
| `detection/person_detector.py` | `PersonDetector` (lazy-loaded YOLOv8n, class 0 only), video open/iterate helpers, annotation, model download. |
| `detection/intrusion_engine.py` | `Observation`, `Alert`, `IntrusionEngine` (R1-R4, cooldown, escalation), `PersistenceGate` (video persistence). |
| `simulation/event_simulator.py` | Scenario generator; every event `simulated=True`. |
| `simulation/synthetic_dataset.py` | 23 hand-written + generated labelled scenarios. |
| `database/db_manager.py` | SQLite schema, parameterised queries, filters, acknowledgement, disposition. |
| `evaluation/` | `metrics.py` (confusion matrix, P/R/F1, FPR, latency), `evaluate.py` (classifier + detector benchmark, report files). |
| `dashboard/` | `components.py` (CSS/cards), `views.py` (tabs). `app.py` wires session state. |

Dependency direction (no cycles): `zone_manager` ← `intrusion_engine` ← `db_manager`/`simulation` ← `evaluation` ← `dashboard` ← `app`.

## Data flow
1. **Source**: either a video frame (YOLO → person boxes) or a simulated event.
2. **Video path**: box foot-point (bottom-centre) inside the room's camera zone? → `PersistenceGate` requires N consecutive processed frames, then re-fires at most every `min_gap_seconds` of stream time → one `Observation(person_detected, source=video_yolo)`.
3. **Engine**: `evaluate(obs)` checks the room's rules using the observation timestamp (not the wall clock). Matched → suspicious; if the same (room, rule) alerted within `cooldown_seconds`, the result is *suppressed*; else an `Alert` is created. R4 counts alerts in a sliding window and escalates once per window.
4. **Storage**: `log_result` writes the event and alerts in one transaction.
5. **UI**: counts and tables are read back from SQLite; nothing is pre-filled.

## Rules
| ID | Condition | Severity |
|---|---|---|
| R1 | person_detected in restricted room outside its permitted hours | High |
| R2 | door_open in restricted room outside permitted hours | Medium |
| R3 | motion in restricted room (sensor enabled) while building closed | Low |
| R4 | ≥ threshold alerts within window (building-wide) | High |

## Database
`events(event_id PK, timestamp, room, event_type, source, simulated, suspicious, suppressed, rule_id, details)`
`alerts(alert_id PK, event_id, timestamp, room, event_type, severity CHECK, source, reason, rule_id, rule_name, evidence, simulated, acknowledged, acknowledged_at, disposition)`
Uploaded video is never stored.

## Model inference workflow (Unit VI)
select pretrained model → download once → load → preprocess (Ultralytics letterbox, imgsz 640) → forward pass on CPU → NMS and confidence filter → post-process (zone test) → decision rules → human review.

## Agent view (Unit III)
PEAS: *Performance* - alert correctness, few duplicate alerts; *Environment* - simulated building + camera video; *Actuators* - alerts/log entries; *Sensors* - video frames, simulated door/motion events. The engine is a **model-based reflex agent**: condition-action rules plus internal state (cooldown timers, recent-alert window). Goal-/utility-based/learning agents are **not** implemented.
