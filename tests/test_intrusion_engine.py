import pytest

from detection.intrusion_engine import (EVENT_TYPES, DwellTracker, IntrusionEngine, Observation, PersistenceGate,
                                        UnknownRoomError)
from detection.zone_manager import ConfigError, parse_windows, in_window
from tests.conftest import obs


def test_normal_activity_in_permitted_hours(engine):
    for room, et, t in [("Server Room", "person_detected", "10:00"), ("Lobby", "door_open", "12:00"),
                        ("Laboratory", "motion", "14:00"), ("Storage Room", "door_open", "09:00")]:
        r = engine.evaluate(obs(room, et, t))
        assert not r.suspicious and r.alerts == []


def test_person_in_restricted_zone_after_hours_is_high(engine):
    r = engine.evaluate(obs("Server Room", "person_detected", "03:00", source="video_yolo", simulated=False))
    a = r.alerts[0]
    assert a.rule_id == "R1" and a.severity == "High" and not a.acknowledged
    assert "restricted zone" in a.reason and "Server Room" in a.reason and "simulated" not in a.reason.lower()
    assert a.evidence and a.event_id == r.observation.event_id and a.source == "video_yolo"


def test_door_open_after_hours_is_medium_and_labelled_simulated(engine):
    a = engine.evaluate(obs("Server Room", "door_open", "22:30")).alerts[0]
    assert a.rule_id == "R2" and a.severity == "Medium" and "simulated door-open" in a.reason and a.simulated


def test_motion_when_building_closed_is_low(engine):
    assert engine.evaluate(obs("Storage Room", "motion", "01:00")).alerts[0].rule_id == "R3"
    assert engine.evaluate(obs("Laboratory", "motion", "12:00")).alerts == []  # building open


def test_force_closed_makes_motion_suspicious(zones, engine):
    zones.config.force_closed = True
    assert engine.evaluate(obs("Server Room", "motion", "12:00")).matched_rule == "R3"


def test_disabled_motion_sensor_ignored(zones, engine):
    zones.get_room("Server Room").motion_sensor_enabled = False
    assert not engine.evaluate(obs("Server Room", "motion", "01:00")).suspicious


def test_unrestricted_room_never_alerts(engine):
    for et in ("person_detected", "door_open", "motion"):
        assert not engine.evaluate(obs("Lobby", et, "03:00")).suspicious


@pytest.mark.parametrize("t,expected", [("08:59", True), ("09:00", False), ("17:00", False), ("17:01", True)])
def test_window_boundaries(engine, t, expected):
    assert engine.evaluate(obs("Server Room", "door_open", t)).suspicious is expected


def test_cooldown_suppresses_duplicates_then_expires(engine):
    first = engine.evaluate(obs("Server Room", "person_detected", "03:00", 0))
    dup = engine.evaluate(obs("Server Room", "person_detected", "03:00", 30))
    later = engine.evaluate(obs("Server Room", "person_detected", "03:01", 30))
    assert len(first.alerts) == 1
    assert dup.suspicious and dup.suppressed and dup.alerts == []
    assert len(later.alerts) == 1


def test_cooldown_is_per_room_and_rule(engine):
    a = engine.evaluate(obs("Server Room", "person_detected", "03:00", 0))
    b = engine.evaluate(obs("Laboratory", "person_detected", "03:00", 5))
    c = engine.evaluate(obs("Server Room", "door_open", "03:00", 10))
    assert a.alerts and b.alerts and c.alerts


def test_multiple_suspicious_events_escalate_once(engine):
    results = [engine.evaluate(obs("Server Room", "door_open", "02:00", 0)),
               engine.evaluate(obs("Laboratory", "person_detected", "02:00", 30)),
               engine.evaluate(obs("Storage Room", "motion", "02:01", 0))]
    assert [len(r.alerts) for r in results] == [1, 1, 2]
    r4 = results[2].alerts[1]
    assert r4.rule_id == "R4" and r4.severity == "High" and r4.room == "Building-wide"
    # a further event inside the same window does not escalate again
    again = engine.evaluate(obs("Server Room", "person_detected", "02:02", 0))
    assert [a.rule_id for a in again.alerts] == ["R1"]


def test_events_outside_window_do_not_escalate(engine):
    engine.evaluate(obs("Server Room", "door_open", "02:00"))
    engine.evaluate(obs("Laboratory", "person_detected", "02:10"))
    assert [a.rule_id for a in engine.evaluate(obs("Storage Room", "motion", "02:20")).alerts] == ["R3"]


def test_reset_clears_memory(engine):
    engine.evaluate(obs("Server Room", "person_detected", "03:00"))
    engine.reset()
    assert engine.evaluate(obs("Server Room", "person_detected", "03:00", 5)).alerts


def test_invalid_inputs(engine):
    with pytest.raises(ValueError):
        Observation("Lobby", "teleport", obs("Lobby", "motion", "10:00").timestamp, "x")
    with pytest.raises(ValueError):
        Observation("Lobby", "motion", "not a datetime", "x")  # type: ignore[arg-type]
    with pytest.raises(UnknownRoomError):
        engine.evaluate(obs("Basement", "motion", "10:00"))
    with pytest.raises(ConfigError):
        parse_windows("9-17")
    with pytest.raises(ConfigError):
        in_window(10, "25:00", "26:00")


def test_overnight_window():
    assert in_window(23 * 60, "22:00", "06:00") and in_window(3 * 60, "22:00", "06:00")
    assert not in_window(12 * 60, "22:00", "06:00")


def test_persistence_gate():
    g = PersistenceGate(min_frames=3, min_gap_seconds=10)
    fires = [g.update("r", True, t) for t in range(0, 30)]
    assert fires.index(True) == 2 and sum(fires) == 3  # t=2, 12, 22
    g2 = PersistenceGate(3, 10)
    assert not any(g2.update("r", i % 2 == 0, i) for i in range(20))  # flicker never persists
    with pytest.raises(ValueError):
        PersistenceGate(0, 1)


def test_event_types_constant():
    assert "person_detected" in EVENT_TYPES


def test_loitering_fires_even_during_permitted_hours(engine):
    a = engine.evaluate(obs("Server Room", "prolonged_presence", "11:00")).alerts[0]
    assert a.rule_id == "R5" and a.severity == "High" and "loitering" in a.reason.lower() or "remained" in a.reason.lower()


def test_loitering_does_not_fire_in_unrestricted_room(engine):
    assert not engine.evaluate(obs("Lobby", "prolonged_presence", "11:00")).suspicious


def test_dwell_tracker_fires_once_per_continuous_stay():
    d = DwellTracker(dwell_seconds=5)
    fires = [d.update("r", True, t) for t in range(0, 12)]
    assert fires.index(True) == 5 and sum(fires) == 1
    assert not d.update("r", False, 12)          # presence breaks: resets
    assert not any(d.update("r", True, 12 + t) for t in range(4))
    assert d.update("r", True, 12 + 5)            # re-arms and fires again after a fresh continuous stay


def test_dwell_tracker_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        DwellTracker(0)
