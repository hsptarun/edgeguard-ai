"""Labelled synthetic test dataset for the rule-based classifier.

Two parts:
  1. Hand-written cases with literal clock times against DEFAULT_CONFIG (BuildingConfig.default()).
     Labels were assigned by hand from the written access policy, including boundary minutes.
  2. Generated cases from EventSimulator (labels are true by construction).

IMPORTANT: these labels encode the access policy, so metrics measure whether the engine implements
that policy correctly. They are NOT an estimate of real-world detection accuracy.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from detection.intrusion_engine import Observation
from detection.zone_manager import BuildingConfig, ZoneManager
from simulation.event_simulator import SCENARIOS, EventSimulator

DAY = date(2026, 3, 10)


@dataclass
class LabelledScenario:
    scenario_id: str
    name: str
    observations: list[Observation]
    expected_suspicious: bool
    expected_alerts: int
    expected_rules: tuple[str, ...]
    origin: str  # "hand_written" | "generated"


def _o(room: str, etype: str, hhmm: str, sec: int = 0) -> Observation:
    h, m = map(int, hhmm.split(":"))
    return Observation(room, etype, datetime(DAY.year, DAY.month, DAY.day, h, m, sec), "synthetic_dataset",
                       simulated=True, details="synthetic dataset")


def hand_written() -> list[LabelledScenario]:
    # (id, description, observations, suspicious?, expected alert count, expected rules)
    N, S = False, True
    cases = [
        ("H01", "Lobby person 10:00", [_o("Lobby", "person_detected", "10:00")], N, 0, ()),
        ("H02", "Office door open 14:30", [_o("Office", "door_open", "14:30")], N, 0, ()),
        ("H03", "Server Room person 10:15 (permitted)", [_o("Server Room", "person_detected", "10:15")], N, 0, ()),
        ("H04", "Server Room door 09:00 (window start, inclusive)", [_o("Server Room", "door_open", "09:00")], N, 0, ()),
        ("H05", "Server Room door 17:00 (window end, inclusive)", [_o("Server Room", "door_open", "17:00")], N, 0, ()),
        ("H06", "Laboratory motion 18:30 (building open)", [_o("Laboratory", "motion", "18:30")], N, 0, ()),
        ("H07", "Lobby person 23:00 (unrestricted room)", [_o("Lobby", "person_detected", "23:00")], N, 0, ()),
        ("H08", "Storage occupancy 12:00", [_o("Storage Room", "occupancy", "12:00")], N, 0, ()),
        ("H09", "Server Room motion 12:00 (building open)", [_o("Server Room", "motion", "12:00")], N, 0, ()),
        ("H10", "Office motion 02:00 (unrestricted; rules do not cover it)", [_o("Office", "motion", "02:00")], N, 0, ()),
        ("H11", "Server Room door_close 03:00 (closing is not an intrusion rule)", [_o("Server Room", "door_close", "03:00")], N, 0, ()),
        ("H12", "Server Room occupancy 03:00 (type not covered by rules)", [_o("Server Room", "occupancy", "03:00")], N, 0, ()),
        ("H13", "Lobby loitering 11:00 (unrestricted room, R5 does not apply)", [_o("Lobby", "prolonged_presence", "11:00")], N, 0, ()),
        ("S01", "Server Room person 03:00", [_o("Server Room", "person_detected", "03:00")], S, 1, ("R1",)),
        ("S02", "Server Room door open 22:30", [_o("Server Room", "door_open", "22:30")], S, 1, ("R2",)),
        ("S03", "Server Room motion 01:00 (building closed)", [_o("Server Room", "motion", "01:00")], S, 1, ("R3",)),
        ("S04", "Storage door 17:01 (one minute after window)", [_o("Storage Room", "door_open", "17:01")], S, 1, ("R2",)),
        ("S05", "Server Room door 08:59 (one minute before window)", [_o("Server Room", "door_open", "08:59")], S, 1, ("R2",)),
        ("S06", "Laboratory person 19:01 (after lab hours, building open)", [_o("Laboratory", "person_detected", "19:01")], S, 1, ("R1",)),
        ("S07", "Laboratory person 21:00", [_o("Laboratory", "person_detected", "21:00")], S, 1, ("R1",)),
        ("S08", "Storage motion 21:00 (building closed)", [_o("Storage Room", "motion", "21:00")], S, 1, ("R3",)),
        ("S09", "Burst: three restricted rooms within 60 s",
         [_o("Server Room", "door_open", "02:00", 0), _o("Laboratory", "person_detected", "02:00", 30),
          _o("Storage Room", "motion", "02:01", 0)], S, 4, ("R2", "R1", "R3", "R4")),
        ("S10", "Duplicate suppression: 5 person detections in 40 s, same room",
         [_o("Server Room", "person_detected", "03:00", 10 * i) for i in range(5)], S, 1, ("R1",)),
        ("S11", "Cooldown expiry: two detections 90 s apart",
         [_o("Server Room", "person_detected", "03:00", 0), _o("Server Room", "person_detected", "03:01", 30)], S, 2, ("R1", "R1")),
        ("S12", "Server Room loitering 11:00 (PERMITTED hours - R5 still fires)", [_o("Server Room", "prolonged_presence", "11:00")], S, 1, ("R5",)),
        ("S13", "Laboratory loitering 23:00 (also after hours)", [_o("Laboratory", "prolonged_presence", "23:00")], S, 1, ("R5",)),
    ]
    return [LabelledScenario(i, d, obs, s, n, rules, "hand_written") for i, d, obs, s, n, rules in cases]


def generated(seed: int = 42, per_scenario: int = 10) -> list[LabelledScenario]:
    sim = EventSimulator(ZoneManager(BuildingConfig.default()), seed=seed)
    out: list[LabelledScenario] = []
    for name in SCENARIOS:
        for k in range(per_scenario):
            g = sim.generate(name, day=DAY + timedelta(days=k))
            rules = ()
            out.append(LabelledScenario(f"G-{name}-{k:02d}", name, g.observations, g.expected_suspicious,
                                        g.expected_alerts, rules, "generated"))
    random.Random(seed).shuffle(out)
    return out


def build_dataset(seed: int = 42, per_scenario: int = 10) -> list[LabelledScenario]:
    return hand_written() + generated(seed, per_scenario)


def to_dataframe(scenarios: list[LabelledScenario]) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        for o in s.observations:
            rows.append({"scenario_id": s.scenario_id, "scenario": s.name, "origin": s.origin, "event_id": o.event_id,
                         "timestamp": o.timestamp.isoformat(), "room": o.room, "event_type": o.event_type,
                         "expected_suspicious": s.expected_suspicious, "expected_alerts": s.expected_alerts})
    return pd.DataFrame(rows)


def save_csv(scenarios: list[LabelledScenario], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_dataframe(scenarios).to_csv(path, index=False)
    return path
