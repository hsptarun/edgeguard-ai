from datetime import date

import pytest

from database.db_manager import DatabaseError, DatabaseManager
from detection.intrusion_engine import IntrusionEngine
from detection.zone_manager import BuildingConfig, ZoneManager
from tests.conftest import obs


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(tmp_path / "sub" / "test.db")


@pytest.fixture
def populated(db, engine):
    for o in [obs("Server Room", "person_detected", "03:00"), obs("Server Room", "door_open", "10:00"),
              obs("Laboratory", "door_open", "22:00"), obs("Lobby", "motion", "12:00")]:
        db.log_result(engine.evaluate(o))
    return db


def test_creation_and_empty_state(db):
    assert db.path.exists()
    assert db.counts() == {"total_events": 0, "suspicious_events": 0, "normal_events": 0, "active_alerts": 0, "total_alerts": 0}
    assert db.get_alerts().empty and db.get_events().empty


def test_persistence_across_instances(populated):
    again = DatabaseManager(populated.path)
    c = again.counts()
    assert c["total_events"] == 4 and c["suspicious_events"] == 2 and c["normal_events"] == 2 and c["active_alerts"] == 2


def test_filters(populated):
    assert len(populated.get_alerts(severity="High")) == 1
    assert len(populated.get_alerts(room="Laboratory")) == 1
    assert len(populated.get_alerts(event_type="door_open")) == 1
    assert len(populated.get_events(room="Lobby")) == 1
    assert len(populated.get_events(date_from=date(2026, 3, 10), date_to=date(2026, 3, 10))) == 4
    assert populated.get_events(date_from=date(2026, 3, 11)).empty
    assert populated.get_events(date_to=date(2026, 3, 9)).empty


def test_invalid_filters(populated):
    with pytest.raises(ValueError):
        populated.get_alerts(severity="Critical")
    with pytest.raises(ValueError):
        populated.get_events(date_from=date(2026, 3, 11), date_to=date(2026, 3, 10))


def test_acknowledge_and_disposition(populated):
    aid = populated.get_alerts().iloc[0].alert_id
    assert populated.acknowledge(aid)
    assert populated.counts()["active_alerts"] == 1
    assert len(populated.get_alerts(acknowledged=True)) == 1
    assert populated.set_disposition(aid, "confirmed_incident")
    assert populated.get_alerts(acknowledged=True).iloc[0].disposition == "confirmed_incident"
    assert not populated.acknowledge("ALT-NOPE")
    with pytest.raises(ValueError):
        populated.set_disposition(aid, "whatever")


def test_sql_injection_is_inert(populated):
    assert populated.get_alerts(room="x'; DROP TABLE alerts; --").empty
    assert populated.counts()["total_alerts"] == 2


def test_duplicate_event_id_raises_safe_error(db, engine):
    res = engine.evaluate(obs("Lobby", "motion", "10:00"))
    db.log_result(res)
    with pytest.raises(DatabaseError):
        db.log_result(res)
    assert db.counts()["total_events"] == 1  # failed transaction rolled back


def test_unwritable_path_raises_database_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(DatabaseError):
        DatabaseManager(blocker / "x.db")


def test_clear_and_room_activity(populated):
    assert len(populated.room_activity("Server Room")) == 2
    populated.clear_all()
    assert populated.counts()["total_events"] == 0
