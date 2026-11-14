"""Reusable UI pieces: dark-theme CSS, metric cards, badges, room cards. User text is HTML-escaped."""
from __future__ import annotations

import html

import streamlit as st

SEV_COLOR = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#38bdf8"}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.eg-card {background:#111a2e; border:1px solid #1f2b47; border-radius:12px; padding:14px 16px;}
.eg-metric .v {font-size:2rem; font-weight:700; line-height:1.1;}
.eg-metric .l {font-size:.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.06em;}
.eg-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; color:#0b1220;}
.eg-tag {display:inline-block; padding:1px 8px; border-radius:6px; font-size:.7rem; background:#1e293b; color:#cbd5e1; margin-right:4px;}
.eg-room {border-left:4px solid #334155;}
.eg-room.public {border-left-color:#22c55e;}
.eg-room.armed {border-left-color:#ef4444;}
.eg-room.permitted {border-left-color:#f59e0b;}
.eg-muted {color:#94a3b8; font-size:.85rem;}
.eg-empty {border:1px dashed #334155; border-radius:12px; padding:22px; text-align:center; color:#94a3b8;}
.eg-banner {background:#1e293b; border-left:4px solid #38bdf8; border-radius:8px; padding:10px 14px; font-size:.85rem; color:#cbd5e1;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def metric_card(label: str, value, color: str = "#e2e8f0") -> None:
    st.markdown(f'<div class="eg-card eg-metric"><div class="l">{html.escape(label)}</div>'
                f'<div class="v" style="color:{color}">{html.escape(str(value))}</div></div>', unsafe_allow_html=True)


def severity_badge(sev: str) -> str:
    return f'<span class="eg-badge" style="background:{SEV_COLOR.get(sev, "#94a3b8")}">{html.escape(sev)}</span>'


def empty_state(message: str) -> None:
    st.markdown(f'<div class="eg-empty">{html.escape(message)}</div>', unsafe_allow_html=True)


def banner(message: str) -> None:
    st.markdown(f'<div class="eg-banner">{html.escape(message)}</div>', unsafe_allow_html=True)


def room_card(name: str, restricted: bool, permitted_now: bool, hours: str, door: str, motion_on: bool,
              motion_seen: bool, event_count: int) -> None:
    cls = "public" if not restricted else ("permitted" if permitted_now else "armed")
    status = "Public" if not restricted else ("Restricted - access permitted now" if permitted_now else "Restricted - ARMED (no access now)")
    st.markdown(
        f'<div class="eg-card eg-room {cls}"><b>{html.escape(name)}</b><br>'
        f'<span class="eg-muted">{html.escape(status)}</span><br>'
        f'<span class="eg-tag">hours {html.escape(hours)}</span>'
        f'<span class="eg-tag">door {html.escape(door)} (sim)</span>'
        f'<span class="eg-tag">motion {"sensor on" if motion_on else "sensor off"}{" - seen" if motion_seen else ""} (sim)</span>'
        f'<span class="eg-tag">{event_count} events</span></div>', unsafe_allow_html=True)
