"""EdgeGuard AI - Streamlit entry point. Run with: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from config.settings import BUILDING_CONFIG_PATH, DB_PATH, ensure_directories, setup_logging

st.set_page_config(page_title="EdgeGuard AI", page_icon="🛡️", layout="wide")

from dashboard import views  # noqa: E402  (after set_page_config)
from dashboard.components import inject_css  # noqa: E402
from database.auth_manager import AuthManager  # noqa: E402
from database.db_manager import DatabaseError, DatabaseManager  # noqa: E402
from detection.intrusion_engine import IntrusionEngine  # noqa: E402
from detection.person_detector import PersonDetector  # noqa: E402
from detection.zone_manager import ZoneManager  # noqa: E402
from simulation.event_simulator import EventSimulator  # noqa: E402


def init_state() -> bool:
    """Create session-wide objects once. Returns False (and shows an error) if the database
    could not be initialised, in which case nothing else should be rendered."""
    ss = st.session_state
    if "zones" in ss:
        return True
    setup_logging()
    ensure_directories()
    try:
        ss.db = DatabaseManager(DB_PATH)
        ss.auth = AuthManager(DB_PATH)
    except DatabaseError as exc:
        st.error(f"Cannot start: {exc}")
        return False
    ss.zones = ZoneManager.load(BUILDING_CONFIG_PATH)
    ss.engine = IntrusionEngine(ss.zones)       # engine reads ss.zones.config live
    ss.simulator = EventSimulator(ss.zones)
    ss.detector = PersonDetector()
    ss.setdefault("user", None)
    return True


inject_css()
if init_state():
    if not st.session_state.user:
        views.render_login()
    else:
        views.render_sidebar()
        st.title("🛡️ EdgeGuard AI")
        st.caption("Edge AI-based intrusion detection for smart buildings - local inference, explainable rules, human review")
        tabs = st.tabs(["Overview", "Video Monitor", "Simulator", "Alerts & History", "Building Setup", "Evaluation", "Responsible AI"])
        for tab, fn in zip(tabs, [views.render_overview, views.render_video, views.render_simulator, views.render_alerts,
                                  views.render_setup, views.render_evaluation, views.render_responsible_ai]):
            with tab:
                fn()
