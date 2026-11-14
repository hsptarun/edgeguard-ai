"""Software sensor simulator. Every generated event is labelled simulated=True.

Scenario timestamps are chosen so that the scenario's intent (normal / suspicious) is true under
the *current building configuration*. The expected label is therefore set by construction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from detection.intrusion_engine import Observation
from detection.zone_manager import Room, ZoneManager

SCENARIOS = {
    "normal_door_open": "Normal door opening during permitted hours",
    "normal_occupancy": "Normal room occupancy during permitted hours",
    "normal_motion": "Normal motion in a room during permitted hours",
    "motion_empty_room": "Motion in a restricted room while the building is closed",
    "door_after_hours": "Door opening in a restricted room after permitted hours",
    "person_restricted_zone": "Simulated person detection in a restricted zone after hours",
    "multiple_suspicious": "Burst of suspicious events in several restricted rooms",
    "loitering_restricted_zone": "Person loiters in a restricted zone past the dwell threshold (any time)",
}
SOURCE = "simulated_sensor"


class ScenarioError(ValueError):
    """The building configuration cannot support the requested scenario."""


@dataclass
class GeneratedScenario:
    name: str
    observations: list[Observation]
    expected_suspicious: bool
    expected_alerts: int
    description: str = ""


@dataclass
class EventSimulator:
    zones: ZoneManager
    seed: int | None = None
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # ---- helpers -----------------------------------------------------------------------
    def _candidate_times(self, day: date, predicate) -> list[datetime]:
        out = []
        for minute in range(0, 24 * 60, 15):
            dt = datetime.combine(day, time(minute // 60, minute % 60))
            if predicate(dt):
                out.append(dt)
        return out

    def _pick(self, rooms: list[Room], day: date, pred_factory) -> tuple[Room, datetime]:
        """Pick a (room, time) pair satisfying pred_factory(room)(datetime)."""
        options = [(r, t) for r in rooms for t in self._candidate_times(day, pred_factory(r))]
        if not options:
            raise ScenarioError("The current building configuration cannot produce this scenario "
                                "(for example no restricted room has hours outside which it is closed).")
        room, t = self.rng.choice(options)
        return room, t + timedelta(seconds=self.rng.randint(0, 59))

    def _obs(self, room: Room, t: datetime, etype: str, details: str = "") -> Observation:
        return Observation(room.name, etype, t, SOURCE, simulated=True, details=details or "simulated event")

    # ---- public API --------------------------------------------------------------------
    def generate(self, scenario: str, room: str | None = None, at: datetime | None = None,
                 day: date | None = None) -> GeneratedScenario:
        """Generate one scenario. `room` pins the room; `at` pins the time (labels then follow the rules
        implicitly, so expected_* is only guaranteed when `at` is None)."""
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {sorted(SCENARIOS)}.")
        day = day or (at.date() if at else date.today())
        zones = self.zones
        pool = [zones.get_room(room)] if room else list(zones.rooms)
        restricted = [r for r in pool if r.restricted]
        desc = SCENARIOS[scenario]

        def normal_pred(r: Room):
            return lambda dt: r.is_access_permitted(dt) and not zones.building_closed(dt)

        def after_hours_pred(r: Room):
            return lambda dt: not r.is_access_permitted(dt)

        def closed_pred(r: Room):
            return lambda dt: zones.building_closed(dt)

        if scenario in ("normal_door_open", "normal_occupancy", "normal_motion"):
            etype = {"normal_door_open": "door_open", "normal_occupancy": "occupancy", "normal_motion": "motion"}[scenario]
            r, t = self._pick(pool, day, normal_pred)
            t = at or t
            return GeneratedScenario(scenario, [self._obs(r, t, etype)], False, 0, desc)

        if scenario == "multiple_suspicious":
            n = zones.config.rules.multi_event_threshold
            rooms = restricted if room else zones.restricted_rooms
            if len(rooms) < n:
                raise ScenarioError(f"Need at least {n} restricted rooms (found {len(rooms)}).")
            chosen = self.rng.sample(rooms, n)
            cand = [dt for dt in self._candidate_times(day, lambda d: zones.building_closed(d))
                    if zones.building_closed(dt + timedelta(minutes=3))
                    and all(not r.is_access_permitted(dt) and not r.is_access_permitted(dt + timedelta(minutes=3))
                            for r in chosen)]
            if not cand:
                raise ScenarioError("No time window found where all chosen rooms are off-hours and the building is closed.")
            start = at or (self.rng.choice(cand) + timedelta(seconds=self.rng.randint(0, 20)))
            kinds = ["person_detected", "door_open", "motion"]
            obs = [self._obs(r, start + timedelta(seconds=30 * i), self.rng.choice(kinds)) for i, r in enumerate(chosen)]
            # expected alerts: one per event (distinct rooms) + one R4 escalation
            return GeneratedScenario(scenario, obs, True, n + 1, desc)

        if not restricted:
            raise ScenarioError("This scenario needs at least one restricted room.")
        if scenario == "loitering_restricted_zone":
            # Loitering is suspicious at ANY time (it is about behaviour, not the clock),
            # so any time-of-day candidate works - unlike the after-hours scenarios above.
            r, t = self._pick(restricted, day, lambda room: (lambda dt: True))
            t = at or t
            return GeneratedScenario(scenario, [self._obs(r, t, "prolonged_presence")], True, 1, desc)
        etype = {"motion_empty_room": "motion", "door_after_hours": "door_open",
                 "person_restricted_zone": "person_detected"}[scenario]
        pred = closed_pred if scenario == "motion_empty_room" else after_hours_pred
        r, t = self._pick(restricted, day, pred)
        t = at or t
        return GeneratedScenario(scenario, [self._obs(r, t, etype)], True, 1, desc)

    def random_mixed(self, n: int, day: date | None = None) -> list[GeneratedScenario]:
        """n random single-event scenarios (excludes the burst scenario)."""
        names = [s for s in SCENARIOS if s != "multiple_suspicious"]
        out = []
        for _ in range(n):
            out.append(self.generate(self.rng.choice(names), day=day))
        return out
