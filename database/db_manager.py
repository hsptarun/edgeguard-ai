"""SQLite persistence for observed events and alerts. All SQL is parameterised."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd

from detection.intrusion_engine import DISPOSITIONS, SEVERITIES, Alert, EvaluationResult, Observation

log = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Wraps sqlite3 errors so the UI can show a safe message."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    room        TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    source      TEXT NOT NULL,
    simulated   INTEGER NOT NULL,
    suspicious  INTEGER NOT NULL,
    suppressed  INTEGER NOT NULL DEFAULT 0,
    rule_id     TEXT,
    details     TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    room            TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    severity        TEXT NOT NULL CHECK (severity IN ('Low','Medium','High')),
    source          TEXT NOT NULL,
    reason          TEXT NOT NULL,
    rule_id         TEXT NOT NULL,
    rule_name       TEXT NOT NULL,
    evidence        TEXT NOT NULL,
    simulated       INTEGER NOT NULL,
    acknowledged    INTEGER NOT NULL DEFAULT 0,
    acknowledged_at TEXT,
    disposition     TEXT NOT NULL DEFAULT 'unreviewed'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
"""


def _iso(dt) -> str:
    return dt.isoformat(timespec="seconds")


class DatabaseManager:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as conn:
                conn.executescript(SCHEMA)
        except (sqlite3.Error, OSError) as exc:
            log.exception("Database initialisation failed")
            raise DatabaseError(f"Could not initialise database: {exc}") from exc

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            log.error("Database error: %s", exc)
            raise DatabaseError(f"Database operation failed: {exc}") from exc
        finally:
            conn.close()

    # ---- writes ------------------------------------------------------------------------
    @staticmethod
    def _insert_event(conn: sqlite3.Connection, obs: Observation, res: EvaluationResult | None) -> None:
        conn.execute(
            "INSERT INTO events (event_id,timestamp,room,event_type,source,simulated,suspicious,suppressed,rule_id,details)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (obs.event_id, _iso(obs.timestamp), obs.room, obs.event_type, obs.source, int(obs.simulated),
             int(bool(res and res.suspicious)), int(bool(res and res.suppressed)),
             res.matched_rule if res else None, obs.details))

    @staticmethod
    def _insert_alert(conn: sqlite3.Connection, a: Alert) -> None:
        conn.execute(
            "INSERT INTO alerts (alert_id,event_id,timestamp,room,event_type,severity,source,reason,rule_id,rule_name,"
            "evidence,simulated,acknowledged,disposition) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (a.alert_id, a.event_id, _iso(a.timestamp), a.room, a.event_type, a.severity, a.source, a.reason,
             a.rule_id, a.rule_name, a.evidence, int(a.simulated), int(a.acknowledged), a.disposition))

    def log_result(self, result: EvaluationResult) -> None:
        """Store the observation and any alerts it produced in a single transaction."""
        with self._conn() as conn:
            self._insert_event(conn, result.observation, result)
            for a in result.alerts:
                self._insert_alert(conn, a)

    def acknowledge(self, alert_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE alerts SET acknowledged=1, acknowledged_at=datetime('now') WHERE alert_id=?", (alert_id,))
            return cur.rowcount > 0

    def set_disposition(self, alert_id: str, disposition: str) -> bool:
        """Human review outcome. Only a person should call this (dashboard buttons)."""
        if disposition not in DISPOSITIONS:
            raise ValueError(f"Invalid disposition '{disposition}'.")
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE alerts SET disposition=?, acknowledged=1, acknowledged_at=COALESCE(acknowledged_at, datetime('now'))"
                " WHERE alert_id=?", (disposition, alert_id))
            return cur.rowcount > 0

    def clear_all(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM alerts")
            conn.execute("DELETE FROM events")

    # ---- reads -------------------------------------------------------------------------
    @staticmethod
    def _where(date_from: date | None, date_to: date | None, room: str | None, event_type: str | None,
               severity: str | None = None, acknowledged: bool | None = None) -> tuple[str, list]:
        clauses, params = [], []
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must not be after date_to.")
        if date_from:
            clauses.append("substr(timestamp,1,10) >= ?"); params.append(date_from.isoformat())
        if date_to:
            clauses.append("substr(timestamp,1,10) <= ?"); params.append(date_to.isoformat())
        if room:
            clauses.append("room = ?"); params.append(room)
        if event_type:
            clauses.append("event_type = ?"); params.append(event_type)
        if severity:
            if severity not in SEVERITIES:
                raise ValueError(f"Invalid severity '{severity}'.")
            clauses.append("severity = ?"); params.append(severity)
        if acknowledged is not None:
            clauses.append("acknowledged = ?"); params.append(int(acknowledged))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _query(self, sql: str, params: list) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_events(self, date_from: date | None = None, date_to: date | None = None, room: str | None = None,
                   event_type: str | None = None, limit: int = 500) -> pd.DataFrame:
        where, params = self._where(date_from, date_to, room, event_type)
        return self._query(f"SELECT * FROM events{where} ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                           params + [int(limit)])

    def get_alerts(self, date_from: date | None = None, date_to: date | None = None, room: str | None = None,
                   event_type: str | None = None, severity: str | None = None, acknowledged: bool | None = None,
                   limit: int = 500) -> pd.DataFrame:
        where, params = self._where(date_from, date_to, room, event_type, severity, acknowledged)
        return self._query(f"SELECT * FROM alerts{where} ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                           params + [int(limit)])

    def counts(self) -> dict[str, int]:
        with self._conn() as conn:
            total, susp = conn.execute("SELECT COUNT(*), COALESCE(SUM(suspicious),0) FROM events").fetchone()
            active = conn.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged=0").fetchone()[0]
            alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        return {"total_events": total, "suspicious_events": int(susp), "normal_events": total - int(susp),
                "active_alerts": active, "total_alerts": alerts}

    def events_per_room(self) -> dict[str, int]:
        with self._conn() as conn:
            return dict(conn.execute("SELECT room, COUNT(*) FROM events GROUP BY room").fetchall())

    def room_activity(self, room: str, limit: int = 20) -> pd.DataFrame:
        return self._query("SELECT timestamp,event_type,source,simulated,suspicious FROM events "
                           "WHERE room=? ORDER BY timestamp DESC, rowid DESC LIMIT ?", [room, int(limit)])
