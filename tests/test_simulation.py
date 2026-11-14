from datetime import datetime

import pytest

from detection.intrusion_engine import IntrusionEngine
from detection.zone_manager import BuildingConfig, ConfigError, Room, ZoneManager
from simulation.event_simulator import SCENARIOS, EventSimulator, ScenarioError
from simulation.synthetic_dataset import build_dataset, hand_written, to_dataframe


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_scenario_matches_engine(zones, name):
    sim = EventSimulator(zones, seed=1)
    for _ in range(15):
        g = sim.generate(name)
        eng = IntrusionEngine(ZoneManager(BuildingConfig.default()))
        alerts = [a for o in g.observations for a in eng.evaluate(o).alerts]
        assert bool(alerts) == g.expected_suspicious and len(alerts) == g.expected_alerts
        assert all(o.simulated for o in g.observations)


def test_seed_is_reproducible(zones):
    a = EventSimulator(zones, seed=7).generate("door_after_hours")
    b = EventSimulator(zones, seed=7).generate("door_after_hours")
    assert (a.observations[0].room, a.observations[0].timestamp) == (b.observations[0].room, b.observations[0].timestamp)


def test_pinned_room_and_time(zones):
    g = EventSimulator(zones, seed=1).generate("person_restricted_zone", room="Server Room", at=datetime(2026, 3, 10, 3, 0))
    assert g.observations[0].room == "Server Room" and g.observations[0].timestamp.hour == 3


def test_invalid_requests(zones):
    sim = EventSimulator(zones, seed=1)
    with pytest.raises(ValueError):
        sim.generate("nonsense")
    with pytest.raises(KeyError):
        sim.generate("door_after_hours", room="Basement")
    with pytest.raises(ScenarioError):
        sim.generate("door_after_hours", room="Lobby")  # unrestricted room


def test_scenario_error_when_config_cannot_support():
    cfg = BuildingConfig(rooms=[Room("A", True, [("00:00", "23:59")]), Room("B", False, [("08:00", "20:00")])])
    sim = EventSimulator(ZoneManager(cfg), seed=1)
    with pytest.raises(ScenarioError):
        sim.generate("door_after_hours")
    with pytest.raises(ScenarioError):
        sim.generate("multiple_suspicious")


def test_state_updates(zones):
    zones.apply_observation("Server Room", "door_open")
    assert zones.get_room("Server Room").door_status == "open"
    zones.apply_observation("Server Room", "door_close")
    assert zones.get_room("Server Room").door_status == "closed"
    zones.apply_observation("Nowhere", "motion")  # ignored silently


def test_random_mixed(zones):
    assert len(EventSimulator(zones, seed=3).random_mixed(20)) == 20


def test_config_validation_and_roundtrip(tmp_path):
    cfg = BuildingConfig.default()
    zm = ZoneManager(cfg)
    zm.save(tmp_path / "c.json")
    assert ZoneManager.load(tmp_path / "c.json").config.to_dict() == cfg.to_dict()
    (tmp_path / "bad.json").write_text("{not json")
    bad = ZoneManager.load(tmp_path / "bad.json")
    assert bad.load_warning and len(bad.rooms) == 5
    assert ZoneManager.load(tmp_path / "missing.json").load_warning
    with pytest.raises(ConfigError):
        BuildingConfig(rooms=[Room("A"), Room("a")]).validate()
    with pytest.raises(ConfigError):
        Room("A", zone=(0.5, 0.5, 0.4, 0.9)).validate()
    with pytest.raises(ConfigError):
        Room("A", door_status="ajar").validate()


def test_shipped_json_matches_default():
    from config.settings import BUILDING_CONFIG_PATH
    assert ZoneManager.load(BUILDING_CONFIG_PATH).config.to_dict() == BuildingConfig.default().to_dict()


def test_dataset_is_labelled_and_balanced():
    ds = build_dataset(seed=1, per_scenario=3)
    assert len(ds) == len(hand_written()) + 3 * len(SCENARIOS)
    assert any(s.expected_suspicious for s in ds) and any(not s.expected_suspicious for s in ds)
    assert len(to_dataframe(ds)) >= len(ds)
