"""Headless UI smoke tests using Streamlit's AppTest (no browser needed)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TEST_USER, TEST_PASS = "tester01", "testpassword123"


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=60)


def _signed_in_app(tmp_path, monkeypatch) -> AppTest:
    """Boot the app, create an account, log in, and return the AppTest positioned on the dashboard."""
    monkeypatch.setattr("config.settings.DB_PATH", tmp_path / "t.db")
    at = _app().run()
    at.text_input[0].set_value(TEST_USER)   # login tab: username
    at.text_input[1].set_value(TEST_PASS)   # login tab: password
    # account does not exist yet -> login tab shows the generic error; switch to signup instead
    at.tabs[1].text_input[0].set_value(TEST_USER)
    at.tabs[1].text_input[2].set_value(TEST_PASS)
    at.tabs[1].text_input[3].set_value(TEST_PASS)
    at.tabs[1].button[0].click().run()
    assert not at.exception
    assert "created" in at.success[0].value.lower()
    at.text_input[0].set_value(TEST_USER)
    at.text_input[1].set_value(TEST_PASS)
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state.user["username"] == TEST_USER
    return at


def test_app_starts_at_login_screen(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.DB_PATH", tmp_path / "t.db")
    at = _app().run()
    assert not at.exception
    assert any("EdgeGuard" in t.value for t in at.title)
    assert at.session_state.user is None
    # Dashboard tabs/buttons must not be reachable before login.
    assert not any(b.label == "Generate event" for b in at.button)


def test_signup_then_login_reaches_dashboard(tmp_path, monkeypatch):
    at = _signed_in_app(tmp_path, monkeypatch)
    assert any("EdgeGuard" in t.value for t in at.title)
    assert any(b.label == "Generate event" for b in at.button)


def test_wrong_password_shows_generic_error(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.DB_PATH", tmp_path / "t.db")
    at = _app().run()
    at.tabs[1].text_input[0].set_value("someone")
    at.tabs[1].text_input[2].set_value("correctpassword")
    at.tabs[1].text_input[3].set_value("correctpassword")
    at.tabs[1].button[0].click().run()
    at.text_input[0].set_value("someone")
    at.text_input[1].set_value("wrongpassword")
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state.user is None
    assert "Invalid username or password" in at.error[0].value


def test_logout_returns_to_login_screen(tmp_path, monkeypatch):
    at = _signed_in_app(tmp_path, monkeypatch)
    next(b for b in at.sidebar.button if b.label == "Log out").click().run()
    assert not at.exception
    assert at.session_state.user is None
    assert not any(b.label == "Generate event" for b in at.button)


def test_simulator_generates_event(tmp_path, monkeypatch):
    at = _signed_in_app(tmp_path, monkeypatch)
    next(b for b in at.button if b.label == "Generate event").click().run()
    assert not at.exception
    assert at.session_state.db.counts()["total_events"] == 1


def test_alert_flow_and_acknowledge(tmp_path, monkeypatch):
    at = _signed_in_app(tmp_path, monkeypatch)
    next(sb for sb in at.selectbox if sb.label == "Scenario").select("door_after_hours").run()
    next(b for b in at.button if b.label == "Generate event").click().run()
    assert not at.exception
    db = at.session_state.db
    assert db.counts()["active_alerts"] == 1
    next(b for b in at.button if b.label == "Acknowledge").click().run()
    assert not at.exception and db.counts()["active_alerts"] == 0


def test_loitering_scenario_fires_r5(tmp_path, monkeypatch):
    at = _signed_in_app(tmp_path, monkeypatch)
    next(sb for sb in at.selectbox if sb.label == "Scenario").select("loitering_restricted_zone").run()
    next(b for b in at.button if b.label == "Generate event").click().run()
    assert not at.exception
    alerts = at.session_state.db.get_alerts()
    assert alerts.iloc[0].rule_id == "R5"


def test_evaluation_tab_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.views.REPORTS_DIR", tmp_path)
    at = _signed_in_app(tmp_path, monkeypatch)
    next(b for b in at.button if b.label == "Run synthetic-dataset evaluation").click().run()
    assert not at.exception
