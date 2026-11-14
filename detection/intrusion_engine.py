"""Rule-based intrusion detection engine (a small production system).

Terminology (also shown in the dashboard):
  * Observation   - something the system saw or was told: a YOLO person detection in a camera
                    zone, or a simulated door/motion event. Observations are facts, not verdicts.
  * Alert         - an observation that matched a rule. It means "a human should look at this".
  * Confirmed     - a human reviewer decided the alert was a real security incident. The software
    incident        never makes this decision and cannot infer identity or intent.
"""
from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from detection.zone_manager import ZoneManager, format_windows

log = logging.getLogger(__name__)

EVENT_TYPES = ("person_detected", "door_open", "door_close", "motion", "occupancy", "prolonged_presence")
SEVERITIES = ("Low", "Medium", "High")
DISPOSITIONS = ("unreviewed", "false_alarm", "confirmed_incident")
BUILDING_WIDE = "Building-wide"

TERMINOLOGY = {
    "Observed event": "Something the system saw or was told (a person detection or a simulated sensor event). It is not necessarily a problem.",
    "Rule-based alert": "An observed event that matched a configured rule. It is a prompt for human review, not a verdict.",
    "Confirmed intrusion": "A human reviewer decided the alert was a real security incident. The software never decides this and cannot infer identity or intent.",
}

RULES = {
    "R1": ("Person in restricted zone outside permitted hours", "High"),
    "R2": ("Door opened in restricted room outside permitted hours", "Medium"),
    "R3": ("Motion in restricted room while building is closed", "Low"),
    "R4": ("Multiple suspicious events within a configurable period", "High"),
    "R5": ("Prolonged presence (loitering) in a restricted zone", "High"),
}


class UnknownRoomError(ValueError):
    """Observation refers to a room that is not configured."""


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class Observation:
    room: str
    event_type: str
    timestamp: datetime
    source: str                      # e.g. "video_yolo", "simulated_sensor", "synthetic_dataset"
    simulated: bool = True
    details: str = ""
    event_id: str = field(default_factory=lambda: new_id("EVT"))

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Invalid event type '{self.event_type}'. Expected one of {EVENT_TYPES}.")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime.")
        if not self.room or not self.source:
            raise ValueError("room and source are required.")


@dataclass
class Alert:
    event_id: str
    timestamp: datetime
    room: str
    event_type: str
    severity: str
    source: str
    reason: str
    rule_id: str
    rule_name: str
    evidence: str
    simulated: bool
    alert_id: str = field(default_factory=lambda: new_id("ALT"))
    acknowledged: bool = False
    disposition: str = "unreviewed"


@dataclass
class EvaluationResult:
    observation: Observation
    suspicious: bool = False
    suppressed: bool = False          # matched a rule but cooldown suppressed the duplicate alert
    matched_rule: str | None = None
    alerts: list[Alert] = field(default_factory=list)


class IntrusionEngine:
    """Evaluates observations against the building configuration.

    Time is taken from observation timestamps (never the wall clock), so the same engine works
    for live simulation, uploaded videos with an assumed start time, and deterministic tests.
    """

    def __init__(self, zones: ZoneManager):
        self.zones = zones
        self._last_alert: dict[tuple[str, str], datetime] = {}
        self._alert_times: deque[datetime] = deque()

    def reset(self) -> None:
        self._last_alert.clear()
        self._alert_times.clear()

    # ---- rule matching -----------------------------------------------------------------
    def _match(self, obs: Observation) -> tuple[str, str, str] | None:
        room = self.zones.get_room(obs.room)
        if not room.restricted:
            return None
        t = obs.timestamp
        permitted = room.is_access_permitted(t)
        windows = format_windows(room.permitted_hours)
        sim = "simulated " if obs.simulated else ""
        base_ev = (f"event={obs.event_type}; room={room.name} (restricted); time={t:%Y-%m-%d %H:%M:%S}; "
                   f"permitted hours={windows}; source={obs.source}; simulated={obs.simulated}")

        if obs.event_type == "person_detected" and not permitted:
            return ("R1",
                    f"A person was detected inside the configured restricted zone '{room.name}' at "
                    f"{t:%H:%M:%S}, outside its permitted hours ({windows})."
                    + (" This is a simulated event." if obs.simulated else ""),
                    base_ev + (f"; details={obs.details}" if obs.details else ""))
        if obs.event_type == "prolonged_presence":
            # Loitering matters regardless of permitted hours: someone authorised to be in a
            # restricted room is not automatically authorised to linger there indefinitely.
            return ("R5",
                    f"A person remained continuously inside the restricted zone '{room.name}' past the "
                    f"configured dwell threshold, detected at {t:%H:%M:%S}."
                    + (" This is a simulated event." if obs.simulated else ""),
                    base_ev + (f"; details={obs.details}" if obs.details else ""))
        if obs.event_type == "door_open" and not permitted:
            return ("R2",
                    f"A {sim}door-open event occurred in the restricted room '{room.name}' at "
                    f"{t:%H:%M:%S}, outside its permitted hours ({windows}).",
                    base_ev + f"; door_status_before={room.door_status}")
        if obs.event_type == "motion" and room.motion_sensor_enabled and self.zones.building_closed(t):
            cfg = self.zones.config
            why = "manually marked closed" if cfg.force_closed else f"open {cfg.open_time}-{cfg.close_time}"
            return ("R3",
                    f"A {sim}motion event occurred in the restricted room '{room.name}' at "
                    f"{t:%H:%M:%S} while the building was marked closed ({why}).",
                    base_ev + f"; building_closed=True ({why})")
        return None

    # ---- main entry point --------------------------------------------------------------
    def evaluate(self, obs: Observation) -> EvaluationResult:
        try:
            self.zones.get_room(obs.room)
        except KeyError as exc:
            raise UnknownRoomError(f"Unknown room '{obs.room}'") from exc

        result = EvaluationResult(observation=obs)
        match = self._match(obs)
        if match is None:
            return result

        rule_id, reason, evidence = match
        result.suspicious, result.matched_rule = True, rule_id
        cfg = self.zones.config.rules
        key = (obs.room, rule_id)
        last = self._last_alert.get(key)

        if last is not None and 0 <= (obs.timestamp - last).total_seconds() < cfg.cooldown_seconds:
            result.suppressed = True      # duplicate inside the cooldown period -> no new alert
            return result

        self._last_alert[key] = obs.timestamp
        name, severity = RULES[rule_id]
        result.alerts.append(Alert(obs.event_id, obs.timestamp, obs.room, obs.event_type, severity,
                                   obs.source, reason, rule_id, name, evidence, obs.simulated))

        escalation = self._check_multi(obs, result.alerts[0])
        if escalation:
            result.alerts.append(escalation)
        return result

    def _check_multi(self, obs: Observation, first: Alert) -> Alert | None:
        """R4: count alert-producing events in a sliding window; escalate once per window."""
        cfg = self.zones.config.rules
        self._alert_times.append(obs.timestamp)
        horizon = obs.timestamp - timedelta(seconds=cfg.multi_event_window_seconds)
        while self._alert_times and self._alert_times[0] < horizon:
            self._alert_times.popleft()
        n = len(self._alert_times)
        if n < cfg.multi_event_threshold:
            return None
        key = (BUILDING_WIDE, "R4")
        last = self._last_alert.get(key)
        if last is not None and 0 <= (obs.timestamp - last).total_seconds() < cfg.multi_event_window_seconds:
            return None
        self._last_alert[key] = obs.timestamp
        name, severity = RULES["R4"]
        return Alert(
            event_id=obs.event_id, timestamp=obs.timestamp, room=BUILDING_WIDE,
            event_type="multiple_suspicious", severity=severity, source="rule_engine",
            reason=(f"{n} rule-based alerts were raised within {cfg.multi_event_window_seconds} seconds "
                    f"(threshold: {cfg.multi_event_threshold}). The latest was '{first.rule_name}' in '{first.room}'."),
            rule_id="R4", rule_name=name,
            evidence=f"alerts_in_window={n}; window_s={cfg.multi_event_window_seconds}; "
                     f"threshold={cfg.multi_event_threshold}; triggering_event={obs.event_id}",
            simulated=obs.simulated,
        )


class PersistenceGate:
    """Turns per-frame detections into discrete events.

    An event fires after `min_frames` consecutive processed frames with the target present,
    then at most once every `min_gap_seconds` of stream time while the target stays present.
    This prevents one alert per video frame and suppresses single-frame false detections.
    """

    def __init__(self, min_frames: int, min_gap_seconds: float):
        if min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        self.min_frames, self.min_gap = min_frames, min_gap_seconds
        self._state: dict[str, dict] = {}

    def update(self, key: str, present: bool, t: float) -> bool:
        st = self._state.setdefault(key, {"run": 0, "last": None})
        if not present:
            st["run"] = 0
            return False
        st["run"] += 1
        if st["run"] >= self.min_frames and (st["last"] is None or t - st["last"] >= self.min_gap):
            st["last"] = t
            return True
        return False


class DwellTracker:
    """Fires once when a target's CONTINUOUS presence in a zone crosses `dwell_seconds`.

    Unlike PersistenceGate (which turns flicker into a single "present" event quickly),
    this tracks how long someone has stayed, to catch loitering rather than a brief visit.
    It re-arms only after presence breaks, so one long stay produces exactly one alert.
    """

    def __init__(self, dwell_seconds: float):
        if dwell_seconds <= 0:
            raise ValueError("dwell_seconds must be > 0")
        self.dwell_seconds = dwell_seconds
        self._state: dict[str, dict] = {}

    def update(self, key: str, present: bool, t: float) -> bool:
        st = self._state.setdefault(key, {"start": None, "fired": False})
        if not present:
            st["start"], st["fired"] = None, False
            return False
        if st["start"] is None:
            st["start"] = t
        if not st["fired"] and (t - st["start"]) >= self.dwell_seconds:
            st["fired"] = True
            return True
        return False
