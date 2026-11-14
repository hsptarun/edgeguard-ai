"""Streamlit tab renderers. Each function reads/writes st.session_state via the `ctx` helpers in app.py."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import date, datetime, time as dtime

import pandas as pd
import streamlit as st

from config.settings import (ALLOWED_VIDEO_EXTENSIONS, BUILDING_CONFIG_PATH, DEFAULT_CONFIDENCE, MAX_UPLOAD_MB,
                             MODEL_LICENSE, MODEL_NAME, REPORTS_DIR)
from dashboard.components import banner, empty_state, metric_card, room_card, severity_badge
from database.auth_manager import AuthError
from database.db_manager import DatabaseError
from detection.intrusion_engine import (SEVERITIES, TERMINOLOGY, EVENT_TYPES, DwellTracker, Observation,
                                        PersistenceGate, UnknownRoomError)
from detection.person_detector import (ModelNotAvailableError, VideoError, annotate_frame, available_devices,
                                       download_model, iter_frames, open_video, video_info)
from detection.zone_manager import BuildingConfig, ConfigError, Room, ZoneManager, format_windows, parse_windows
from evaluation.evaluate import run as run_evaluation
from evaluation.metrics import latency_summary
from simulation.event_simulator import SCENARIOS, ScenarioError

log = logging.getLogger(__name__)


def _record(result) -> None:
    """Persist an evaluation result and update simulated room state."""
    st.session_state.db.log_result(result)
    st.session_state.zones.apply_observation(result.observation.room, result.observation.event_type)


def ref_time() -> datetime:
    return st.session_state.get("ref_time_override") or datetime.now()


# ---------------------------------------------------------------------------------------
def render_login() -> None:
    """Local account gate. Accounts live only in this project's own SQLite file on this machine."""
    ss = st.session_state
    st.title("🛡️ EdgeGuard AI")
    st.caption("Local security dashboard - sign in to continue")
    banner("Accounts are stored only in data/edgeguard.db on this computer. This is a simple access "
           "gate for a single-machine student project, not a hardened multi-user auth system.")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            try:
                display_name = ss.auth.login(u, p)
                ss.user = {"username": u.strip(), "display_name": display_name}
                st.rerun()
            except AuthError as exc:
                st.error(str(exc))

    with tab_signup:
        with st.form("signup_form"):
            u2 = st.text_input("Choose a username", help="3-32 characters: letters, digits, underscore.")
            d2 = st.text_input("Display name (optional)")
            p2 = st.text_input("Choose a password", type="password", help="At least 8 characters.")
            p3 = st.text_input("Confirm password", type="password")
            submitted2 = st.form_submit_button("Create account", type="primary")
        if submitted2:
            try:
                ss.auth.signup(u2, p2, p3, d2)
                st.success("Account created. Switch to the Log in tab to sign in.")
            except AuthError as exc:
                st.error(str(exc))


def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("## EdgeGuard AI")
        user = ss.get("user")
        if user:
            st.caption(f"Signed in as **{user['display_name']}**")
            if st.button("Log out", use_container_width=True):
                ss.user = None
                st.rerun()
        st.caption("Edge AI intrusion detection for smart buildings")
        det = ss.detector
        st.markdown("**System status**")
        st.write("Database: ✅ ready")
        st.write(f"Person detector: {'✅ model file present' if det.is_ready() else '⚠️ model not downloaded (simulation-only mode)'}")
        st.write("Processing: 🔒 local - no video leaves this computer")
        st.markdown("**Model**")
        st.caption(f"{MODEL_NAME} - {MODEL_LICENSE}")
        stats = ss.get("video_stats")
        if stats:
            st.markdown("**Last video run**")
            st.caption(f"{stats.get('frames_processed', 0)} frames, "
                       f"{stats.get('inference_ms_mean', 0):.0f} ms/frame mean, device {stats.get('device', '-')}")
        st.markdown("---")
        st.markdown("**Dashboard clock**")
        use_custom = st.checkbox("Use a custom 'now'", value=False, help="Changes only how room status is displayed.")
        if use_custom:
            d = st.date_input("Date", value=date.today(), key="ref_d")
            t = st.time_input("Time", value=dtime(3, 0), key="ref_t")
            ss.ref_time_override = datetime.combine(d, t)
        else:
            ss.ref_time_override = None
        st.markdown("---")
        banner("Alerts are prompts for human review. They are not proof of intrusion.")


# ---------------------------------------------------------------------------------------
def render_overview() -> None:
    ss = st.session_state
    zones, db = ss.zones, ss.db
    now = ref_time()
    try:
        c = db.counts()
        per_room = db.events_per_room()
        recent = db.get_alerts(limit=5)
    except DatabaseError as exc:
        st.error(f"Database problem: {exc}")
        return

    closed = zones.building_closed(now)
    st.markdown(f"### {zones.config.building_name}")
    st.caption(f"Reference time {now:%Y-%m-%d %H:%M} - building is **{'CLOSED' if closed else 'OPEN'}** "
               f"({zones.config.open_time}-{zones.config.close_time}{', manually closed' if zones.config.force_closed else ''})")
    cols = st.columns(4)
    with cols[0]: metric_card("Events processed", c["total_events"])
    with cols[1]: metric_card("Normal events", c["normal_events"], "#22c55e")
    with cols[2]: metric_card("Suspicious events", c["suspicious_events"], "#f59e0b")
    with cols[3]: metric_card("Active alerts", c["active_alerts"], "#ef4444" if c["active_alerts"] else "#e2e8f0")

    st.markdown("#### Rooms and restricted zones")
    rcols = st.columns(3)
    for i, r in enumerate(zones.rooms):
        with rcols[i % 3]:
            room_card(r.name, r.restricted, r.is_access_permitted(now), format_windows(r.permitted_hours), r.door_status,
                      r.motion_sensor_enabled, r.motion_detected, per_room.get(r.name, 0))
            st.write("")
    st.caption("Door and motion states are simulated values updated by the Simulator tab; no physical sensors exist.")

    st.markdown("#### Latest alerts")
    if recent.empty:
        empty_state("No alerts yet. Use the Simulator or Video Monitor tab to generate real events.")
    else:
        for _, a in recent.iterrows():
            st.markdown(f"{severity_badge(a.severity)} &nbsp; **{a.room}** - {a.rule_name} "
                        f"<span class='eg-muted'>({a.timestamp}{' - SIMULATED' if a.simulated else ''})</span>", unsafe_allow_html=True)
            st.caption(a.reason)

    with st.expander("Observed event vs. alert vs. confirmed intrusion"):
        for k, v in TERMINOLOGY.items():
            st.markdown(f"**{k}** - {v}")


# ---------------------------------------------------------------------------------------
def _download_model_button() -> None:
    if st.button("Download model now (needs internet once)"):
        with st.spinner("Downloading..."):
            try:
                download_model()
                st.success("Model downloaded. Inference will run locally from now on.")
                st.rerun()
            except ModelNotAvailableError as exc:
                st.error(str(exc))


def render_video() -> None:
    ss = st.session_state
    zones, det = ss.zones, ss.detector
    st.markdown("### Video monitor - local person detection")
    banner("Detecting a person is NOT an intrusion. A person only creates an alert if they are inside a restricted "
           "zone outside its permitted hours. Video is processed locally and is not stored.")
    if not det.is_ready():
        st.warning("The YOLO model file is not available, so video analysis is disabled. The Simulator tab still works.")
        _download_model_button()
        return

    left, right = st.columns([1, 1])
    with left:
        room_name = st.selectbox("Camera covers room", zones.room_names, key="vid_room")
        room = zones.get_room(room_name)
        conf = st.slider("Confidence threshold", 0.10, 0.90, DEFAULT_CONFIDENCE, 0.05)
        stride = st.slider("Process every Nth frame", 1, 15, 3)
        max_frames = st.number_input("Maximum frames to process", 10, 5000, 600, 50)
    with right:
        st.markdown("**Monitored zone (fraction of the frame)**")
        zx = st.slider("Left - right (x1, x2)", 0.0, 1.0, (float(room.zone[0]), float(room.zone[2])), 0.05, key=f"zx_{room_name}")
        zy = st.slider("Top - bottom (y1, y2)", 0.0, 1.0, (float(room.zone[1]), float(room.zone[3])), 0.05, key=f"zy_{room_name}")
        d = st.date_input("Assumed date at video start", value=date.today(), key="vid_d")
        t = st.time_input("Assumed clock time at video start", value=dtime(3, 0), key="vid_t",
                          help="Videos carry no trusted wall-clock time, so you state it. Alert rules use this assumed time.")
        devices = available_devices()
        device = st.selectbox("Device", devices, help="CPU is the tested default. 'mps' (Apple GPU) appears only if PyTorch reports it available; it has not been benchmarked by this project.")

    if zx[0] >= zx[1] or zy[0] >= zy[1]:
        st.error("Zone must have positive width and height.")
        return
    room.zone = (zx[0], zy[0], zx[1], zy[1])

    up = st.file_uploader("Upload a video", type=[e.lstrip(".") for e in ALLOWED_VIDEO_EXTENSIONS])
    if up is None:
        empty_state("No video uploaded. Upload a short clip of people walking to begin.")
        _show_video_summary()
        return
    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large (limit {MAX_UPLOAD_MB} MB).")
        return
    if not st.button("Process video", type="primary"):
        _show_video_summary()
        return

    det.device = device
    start = datetime.combine(d, t)
    gate = PersistenceGate(zones.config.rules.persistence_frames, zones.config.rules.min_gap_seconds)
    dwell = DwellTracker(zones.config.rules.dwell_alert_seconds)
    tmp_path = None
    try:
        suffix = os.path.splitext(up.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:   # deleted in `finally`
            tmp.write(up.getbuffer())
            tmp_path = tmp.name
        det.load()
        cap = open_video(tmp_path)
        info = video_info(cap)
        total = max(1, min(int(max_frames), (info["frames"] // stride) if info["frames"] else int(max_frames)))
        st.caption(f"{info['width']}x{info['height']} @ {info['fps']:.1f} fps")
        prog, view, status = st.progress(0.0), st.empty(), st.empty()
        times, peak, n_events, snaps = [], 0, 0, []
        t_wall = time.perf_counter()
        try:
            for k, (idx, sec, frame) in enumerate(iter_frames(cap, stride, int(max_frames))):
                dets, ms = det.detect(frame, conf)
                times.append(ms)
                h, w = frame.shape[:2]
                inside = [zones.point_in_zone(room, dt.foot_point[0] / w, dt.foot_point[1] / h) for dt in dets]
                peak = max(peak, len(dets))
                annotated = annotate_frame(frame, dets, room.zone, inside, room.name)
                any_inside = any(inside)
                if gate.update(room.name, any_inside, sec):
                    obs = Observation(room.name, "person_detected", start + pd.Timedelta(seconds=sec), "video_yolo",
                                      simulated=False, details=f"frame={idx}; persons_in_zone={sum(inside)}; conf>={conf}")
                    res = zones_engine().evaluate(obs)
                    _record(res)
                    n_events += 1
                    if res.alerts:
                        snaps.append((f"{res.alerts[0].severity} - {res.alerts[0].rule_name} @ t={sec:.1f}s",
                                      annotated[:, :, ::-1].copy()))
                if dwell.update(room.name, any_inside, sec):
                    # Behavioural signal: the person has not merely appeared, they have stayed
                    # continuously past the configured dwell threshold - fires regardless of permitted hours.
                    obs2 = Observation(room.name, "prolonged_presence", start + pd.Timedelta(seconds=sec), "video_yolo",
                                       simulated=False, details=f"frame={idx}; dwell_s>={zones.config.rules.dwell_alert_seconds}")
                    res2 = zones_engine().evaluate(obs2)
                    _record(res2)
                    n_events += 1
                    if res2.alerts:
                        snaps.append((f"{res2.alerts[0].severity} - {res2.alerts[0].rule_name} @ t={sec:.1f}s",
                                      annotated[:, :, ::-1].copy()))
                view.image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
                prog.progress(min(1.0, (k + 1) / total))
                status.caption(f"frame {idx} - {len(dets)} person(s) - {ms:.0f} ms inference")
        finally:
            cap.release()
        wall = time.perf_counter() - t_wall
        stats = latency_summary(times[1:] if len(times) > 1 else times, wall, len(times))
        stats.update({"frames_processed": len(times), "device": device, "peak_people": peak, "events_logged": n_events})
        ss.video_stats, ss.video_snaps = stats, snaps
        st.success(f"Done: {len(times)} frames processed in {wall:.1f}s. Events logged: {n_events}.")
    except (VideoError, ModelNotAvailableError, ValueError, UnknownRoomError, DatabaseError) as exc:
        st.error(f"Could not process video: {exc}")
    except Exception:
        log.exception("Unexpected video-processing error")
        st.error("Unexpected error while processing the video. Check the terminal log for details.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)  # uploaded video is never kept
    _show_video_summary()


def zones_engine():
    return st.session_state.engine


def _show_video_summary() -> None:
    stats = st.session_state.get("video_stats")
    if not stats:
        return
    st.markdown("#### Processing statistics (measured on this run)")
    c = st.columns(4)
    c[0].metric("Frames processed", stats.get("frames_processed", 0))
    c[1].metric("Mean inference", f"{stats.get('inference_ms_mean', 0):.0f} ms")
    c[2].metric("Inference FPS", f"{stats.get('inference_fps', 0):.1f}")
    c[3].metric("Pipeline FPS", f"{stats.get('pipeline_fps_wall_clock', 0):.1f}")
    st.caption("First frame excluded from inference averages (model warm-up). Pipeline FPS includes decoding and drawing.")
    for caption, img in st.session_state.get("video_snaps", [])[:6]:
        st.image(img, caption=caption, channels="RGB", use_container_width=True)


# ---------------------------------------------------------------------------------------
def render_simulator() -> None:
    ss = st.session_state
    zones, sim, engine = ss.zones, ss.simulator, ss.engine
    st.markdown("### Software sensor simulator")
    banner("Every event here is SIMULATED. It is not sensor data and not real-world evidence.")

    c1, c2, c3 = st.columns(3)
    scenario = c1.selectbox("Scenario", list(SCENARIOS), format_func=lambda s: SCENARIOS[s])
    room = c2.selectbox("Room", ["(automatic)"] + zones.room_names)
    custom = c3.checkbox("Set event time manually")
    at = None
    if custom:
        d = c3.date_input("Date", value=date.today(), key="sim_d")
        t = c3.time_input("Time", value=dtime(3, 0), key="sim_t")
        at = datetime.combine(d, t)

    def generate(name: str, pin_room: str | None, when: datetime | None):
        g = sim.generate(name, room=pin_room, at=when)
        out = []
        for o in g.observations:
            res = engine.evaluate(o)
            _record(res)
            out.append({"time": o.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "room": o.room, "event": o.event_type,
                        "label": "SIMULATED", "result": ("ALERT " + ", ".join(a.rule_id for a in res.alerts)) if res.alerts
                        else ("suppressed (cooldown)" if res.suppressed else "normal")})
        ss.sim_log = (out + ss.get("sim_log", []))[:50]

    b = st.columns(5)
    try:
        if b[0].button("Generate event", type="primary"):
            generate(scenario, None if room == "(automatic)" else room, at)
        if b[1].button("Generate 10 mixed"):
            for g in sim.random_mixed(10):
                for o in g.observations:
                    res = engine.evaluate(o); _record(res)
            ss.sim_log = [{"time": "-", "room": "-", "event": "10 random normal/suspicious events", "label": "SIMULATED", "result": "done"}] + ss.get("sim_log", [])
        running = ss.get("sim_running", False)
        if b[2].button("Pause" if running else "Start auto-run"):
            ss.sim_running = not running
            st.rerun()
        if b[3].button("Reset engine memory", help="Clears cooldown/escalation memory only."):
            engine.reset(); st.toast("Engine memory cleared")
        confirm = b[4].checkbox("Confirm wipe")
        if b[4].button("Wipe history", disabled=not confirm):
            ss.db.clear_all(); engine.reset(); ss.sim_log = []
            for r in zones.rooms:
                r.door_status, r.motion_detected = "closed", False
            st.toast("All stored events and alerts deleted")
    except (ScenarioError, ValueError, KeyError, UnknownRoomError, DatabaseError) as exc:
        st.error(str(exc))

    @st.fragment(run_every=2 if ss.get("sim_running") else None)
    def feed():
        if ss.get("sim_running"):
            try:
                generate(sim.rng.choice([s for s in SCENARIOS if s != "multiple_suspicious"]), None, None)
            except (ScenarioError, DatabaseError) as exc:
                ss.sim_running = False
                st.error(str(exc))
            st.caption("Auto-run is ON: one simulated event every 2 seconds.")
        log_ = ss.get("sim_log", [])
        if log_:
            st.dataframe(pd.DataFrame(log_), use_container_width=True, hide_index=True)
        else:
            empty_state("No simulated events generated yet.")
    feed()


# ---------------------------------------------------------------------------------------
def render_alerts() -> None:
    ss = st.session_state
    db, zones = ss.db, ss.zones
    st.markdown("### Alerts and incident history")
    f = st.columns(5)
    use_dates = f[0].checkbox("Filter dates")
    d1 = d2 = None
    if use_dates:
        d1 = f[0].date_input("From", value=date.today().replace(day=1), key="f_d1")
        d2 = f[0].date_input("To", value=date.today(), key="f_d2")
    room = f[1].selectbox("Room", ["All", "Building-wide"] + zones.room_names)
    sev = f[2].selectbox("Severity", ["All", *SEVERITIES])
    et = f[3].selectbox("Event type", ["All", *EVENT_TYPES, "multiple_suspicious"])
    state = f[4].selectbox("Status", ["All", "Active (unacknowledged)", "Acknowledged"])
    try:
        alerts = db.get_alerts(d1, d2, None if room == "All" else room, None if et == "All" else et,
                               None if sev == "All" else sev,
                               None if state == "All" else (state == "Acknowledged"))
    except (ValueError, DatabaseError) as exc:
        st.error(str(exc)); return

    if alerts.empty:
        empty_state("No alerts match the current filters.")
    else:
        view = alerts[["alert_id", "timestamp", "room", "event_type", "severity", "rule_id", "source", "acknowledged", "disposition"]]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download alerts (CSV)", alerts.to_csv(index=False), "edgeguard_alerts.csv", "text/csv")
        pick = st.selectbox("Inspect alert", alerts.alert_id.tolist(),
                            format_func=lambda i: f"{i} - {alerts.set_index('alert_id').loc[i, 'room']} - {alerts.set_index('alert_id').loc[i, 'severity']}")
        a = alerts.set_index("alert_id").loc[pick]
        st.markdown(f"{severity_badge(a.severity)} &nbsp; **{a.rule_id}: {a.rule_name}**" +
                    (" &nbsp; <span class='eg-tag'>SIMULATED</span>" if a.simulated else ""), unsafe_allow_html=True)
        st.markdown(f"**Why this alert was raised:** {a.reason}")
        st.markdown(f"**Evidence:** `{a.evidence}`")
        st.caption(f"Event ID {a.event_id} - status: {'acknowledged' if a.acknowledged else 'active'} - review: {a.disposition}")
        b = st.columns(3)
        try:
            if b[0].button("Acknowledge", disabled=bool(a.acknowledged)):
                db.acknowledge(pick); st.rerun()
            if b[1].button("Mark as false alarm"):
                db.set_disposition(pick, "false_alarm"); st.rerun()
            if b[2].button("Confirm as incident (human decision)"):
                db.set_disposition(pick, "confirmed_incident"); st.rerun()
        except (ValueError, DatabaseError) as exc:
            st.error(str(exc))
        st.caption("Only a human reviewer can confirm an incident. The software cannot determine identity or intent.")

    st.markdown("#### Event history (all observed events)")
    try:
        ev = db.get_events(d1, d2, None if room in ("All", "Building-wide") else room, None if et in ("All", "multiple_suspicious") else et)
    except (ValueError, DatabaseError) as exc:
        st.error(str(exc)); return
    if ev.empty:
        empty_state("No events recorded for these filters.")
    else:
        st.dataframe(ev.drop(columns=["details"]), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------------------
def render_setup() -> None:
    ss = st.session_state
    zones = ss.zones
    cfg = zones.config
    st.markdown("### Building setup")
    if zones.load_warning:
        st.warning(zones.load_warning)
    with st.form("building_form"):
        c = st.columns(4)
        name = c[0].text_input("Building name", cfg.building_name)
        open_t = c[1].text_input("Opens (HH:MM)", cfg.open_time)
        close_t = c[2].text_input("Closes (HH:MM)", cfg.close_time)
        force = c[3].checkbox("Mark building closed (override)", cfg.force_closed)
        r = st.columns(6)
        cooldown = r[0].number_input("Alert cooldown (s)", 0, 3600, cfg.rules.cooldown_seconds)
        thr = r[1].number_input("Escalation threshold (alerts)", 2, 20, cfg.rules.multi_event_threshold)
        win = r[2].number_input("Escalation window (s)", 10, 7200, cfg.rules.multi_event_window_seconds)
        pers = r[3].number_input("Video persistence (frames)", 1, 60, cfg.rules.persistence_frames)
        gap = r[4].number_input("Video re-trigger gap (s)", 0.0, 600.0, float(cfg.rules.min_gap_seconds))
        dwell = r[5].number_input("Loitering dwell threshold (s)", 1.0, 3600.0, float(cfg.rules.dwell_alert_seconds),
                                   help="R5: continuous stream-time inside a restricted zone before a loitering alert fires - regardless of permitted hours.")
        st.markdown("**Rooms**")
        edits = []
        for rm in cfg.rooms:
            with st.expander(rm.name + (" (restricted)" if rm.restricted else "")):
                e = st.columns(5)
                edits.append({
                    "name": rm.name,
                    "restricted": e[0].checkbox("Restricted", rm.restricted, key=f"rs_{rm.name}"),
                    "hours": e[1].text_input("Permitted hours (HH:MM-HH:MM; ...)", ";".join(f"{a}-{b}" for a, b in rm.permitted_hours), key=f"hr_{rm.name}"),
                    "door": e[2].selectbox("Simulated door", ["closed", "open", "locked"], ["closed", "open", "locked"].index(rm.door_status), key=f"dr_{rm.name}"),
                    "motion": e[3].checkbox("Motion sensor enabled", rm.motion_sensor_enabled, key=f"mo_{rm.name}"),
                    "remove": e[4].checkbox("Remove room", False, key=f"rm_{rm.name}"),
                })
        new_room = st.text_input("Add a room (name)")
        applied = st.form_submit_button("Apply changes", type="primary")
    if applied:
        try:
            rooms = []
            for rm, e in zip(cfg.rooms, edits):
                if e["remove"]:
                    continue
                rooms.append(Room(rm.name, e["restricted"], parse_windows(e["hours"]), e["door"], e["motion"], rm.motion_detected, rm.zone))
            if new_room.strip():
                rooms.append(Room(new_room.strip(), False, [("08:00", "20:00")]))
            new_cfg = BuildingConfig(name, open_t, close_t, force,
                                     type(cfg.rules)(int(cooldown), int(thr), int(win), int(pers), float(gap), float(dwell)),
                                     rooms)
            new_cfg.validate()
            cfg_obj = zones.config
            zones.config = new_cfg
            ss.engine.reset()
            st.success("Applied for this session. Click 'Save to file' to keep it after restart.")
            st.rerun()
        except ConfigError as exc:
            st.error(f"Invalid configuration: {exc}")
    b = st.columns(2)
    if b[0].button("Save to file"):
        try:
            zones.save(BUILDING_CONFIG_PATH); st.success("Saved config/building_config.json")
        except (OSError, ConfigError) as exc:
            st.error(str(exc))
    if b[1].button("Reset to defaults"):
        zones.config = BuildingConfig.default(); ss.engine.reset(); st.rerun()
    st.caption("Restricted rooms with no permitted hours are off-limits at all times.")


# ---------------------------------------------------------------------------------------
def render_evaluation() -> None:
    st.markdown("### Evaluation")
    banner("Two separate evaluations. (A) Rule classifier on labelled synthetic scenarios - labels come from the access "
           "policy, so this verifies the rules, not real-world accuracy. (B) Detector speed on your video - accuracy is not measured (no annotations).")
    if st.button("Run synthetic-dataset evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            out = run_evaluation(out_dir=REPORTS_DIR)
        st.session_state.eval_table = out["table"]
    p = REPORTS_DIR / "evaluation_summary.json"
    if not p.exists():
        empty_state("No evaluation results yet. Click the button above (results are computed live, never pre-filled).")
    else:
        m = json.loads(p.read_text())["classifier"]
        c = st.columns(5)
        for col, (lab, k) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("False-positive rate", "false_positive_rate")]):
            col.metric(lab, f"{m[k]:.3f}")
        st.caption(f"{m['n']} scenarios - TP {m['tp']}, FP {m['fp']}, TN {m['tn']}, FN {m['fn']} - alert-count match {m['alert_count_match_rate']:.3f}")
        img = REPORTS_DIR / "confusion_matrix.png"
        if img.exists():
            st.image(str(img), width=360)
        tbl = st.session_state.get("eval_table")
        if tbl is not None:
            st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.markdown("#### Person detector speed")
    s = st.session_state.get("video_stats")
    if s:
        st.json({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
    else:
        empty_state("No video processed in this session. Process a video in the Video Monitor tab, or run `python -m evaluation.evaluate --video <file>`.")


def render_responsible_ai() -> None:
    st.markdown("### Responsible AI, privacy and limitations")
    st.markdown(f"""
- **Local processing:** video is decoded and analysed on this computer. Nothing is uploaded; the only network use is the one-time model download.
- **Data minimisation:** uploaded videos are written to a temporary file and deleted immediately after processing. Only event metadata (time, room, type) is stored in the local SQLite database.
- **No identity, no intent:** the system detects *people-shaped objects*. It does not recognise faces, track identities, or judge behaviour or intent.
- **False positives / negatives:** the detector can miss people (occlusion, darkness) or detect non-people. Rules depend on correctly configured zones and an assumed video clock.
- **Human review:** an alert is a prompt for a person to look. Only a human can acknowledge, dismiss, or confirm an incident.
- **Model licence:** `{MODEL_NAME}` - {MODEL_LICENSE}. Check AGPL-3.0 obligations before redistributing a modified service built on it.
- **Simulation:** door and motion events are synthetic and labelled as simulated; they are not evidence of real activity.
""")
    for k, v in TERMINOLOGY.items():
        st.markdown(f"**{k}** - {v}")
