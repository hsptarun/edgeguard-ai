"""Building model: rooms, access schedules, restricted zones and rule parameters.

Time handling works at minute resolution. A window ("09:00", "17:00") is inclusive of both
end minutes; if start > end the window wraps over midnight (e.g. "22:00"-"06:00").
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VALID_DOOR_STATUS = ("closed", "open", "locked")


class ConfigError(ValueError):
    """Raised when the building configuration is invalid."""


def parse_hhmm(value: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    try:
        h_s, m_s = str(value).strip().split(":")
        h, m = int(h_s), int(m_s)
    except (ValueError, AttributeError) as exc:
        raise ConfigError(f"Invalid time '{value}'. Use HH:MM (24-hour).") from exc
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ConfigError(f"Time out of range: '{value}'.")
    return h * 60 + m


def in_window(minute: int, start: str, end: str) -> bool:
    s, e = parse_hhmm(start), parse_hhmm(end)
    return s <= minute <= e if s <= e else (minute >= s or minute <= e)


def parse_windows(text: str) -> list[tuple[str, str]]:
    """Parse '09:00-17:00; 19:00-21:00' into validated windows. Empty text -> []."""
    windows: list[tuple[str, str]] = []
    for part in [p.strip() for p in text.replace(",", ";").split(";") if p.strip()]:
        if "-" not in part:
            raise ConfigError(f"Invalid window '{part}'. Use HH:MM-HH:MM.")
        start, end = [x.strip() for x in part.split("-", 1)]
        parse_hhmm(start), parse_hhmm(end)
        windows.append((start, end))
    return windows


def format_windows(windows: list[tuple[str, str]]) -> str:
    return "; ".join(f"{s}-{e}" for s, e in windows) if windows else "none"


@dataclass
class Room:
    name: str
    restricted: bool = False
    permitted_hours: list[tuple[str, str]] = field(default_factory=list)
    door_status: str = "closed"
    motion_sensor_enabled: bool = True
    motion_detected: bool = False
    # Camera zone as normalised (x1, y1, x2, y2) of the video frame. Used by the video monitor.
    zone: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ConfigError("Room name must not be empty.")
        if self.door_status not in VALID_DOOR_STATUS:
            raise ConfigError(f"Room '{self.name}': door status must be one of {VALID_DOOR_STATUS}.")
        for s, e in self.permitted_hours:
            parse_hhmm(s), parse_hhmm(e)
        x1, y1, x2, y2 = self.zone
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ConfigError(f"Room '{self.name}': zone must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1.")

    def is_access_permitted(self, dt: datetime) -> bool:
        """Unrestricted rooms are always permitted; restricted rooms only inside their windows."""
        if not self.restricted:
            return True
        minute = dt.hour * 60 + dt.minute
        return any(in_window(minute, s, e) for s, e in self.permitted_hours)


@dataclass
class RuleSettings:
    cooldown_seconds: int = 60              # same (room, rule) alerts suppressed for this long
    multi_event_threshold: int = 3          # R4: number of alerts ...
    multi_event_window_seconds: int = 300   # ... inside this window
    persistence_frames: int = 5             # video: consecutive processed frames with a person in zone
    min_gap_seconds: float = 10.0           # video: minimum stream time between re-triggers
    dwell_alert_seconds: float = 20.0       # R5: continuous stream-time in a restricted zone before loitering fires

    def validate(self) -> None:
        if self.cooldown_seconds < 0 or self.multi_event_window_seconds <= 0:
            raise ConfigError("Cooldown must be >= 0 and the multi-event window must be > 0.")
        if self.multi_event_threshold < 2:
            raise ConfigError("Multi-event threshold must be at least 2.")
        if self.persistence_frames < 1 or self.min_gap_seconds < 0:
            raise ConfigError("Persistence frames must be >= 1 and minimum gap >= 0.")
        if self.dwell_alert_seconds <= 0:
            raise ConfigError("Dwell alert seconds must be > 0.")


@dataclass
class BuildingConfig:
    building_name: str = "Demo Smart Building"
    open_time: str = "08:00"
    close_time: str = "20:00"
    force_closed: bool = False
    rules: RuleSettings = field(default_factory=RuleSettings)
    rooms: list[Room] = field(default_factory=list)

    @staticmethod
    def default() -> "BuildingConfig":
        return BuildingConfig(
            rooms=[
                Room("Lobby", False, [("06:00", "22:00")], zone=(0.0, 0.0, 1.0, 1.0)),
                Room("Office", False, [("08:00", "20:00")], zone=(0.0, 0.0, 1.0, 1.0)),
                Room("Server Room", True, [("09:00", "17:00")], zone=(0.3, 0.2, 0.9, 1.0)),
                Room("Laboratory", True, [("09:00", "19:00")], zone=(0.3, 0.2, 0.9, 1.0)),
                Room("Storage Room", True, [("09:00", "17:00")], zone=(0.3, 0.2, 0.9, 1.0)),
            ]
        )

    def validate(self) -> None:
        parse_hhmm(self.open_time), parse_hhmm(self.close_time)
        self.rules.validate()
        names = [r.name.strip().lower() for r in self.rooms]
        if len(names) != len(set(names)):
            raise ConfigError("Room names must be unique.")
        if not self.rooms:
            raise ConfigError("At least one room is required.")
        for r in self.rooms:
            r.validate()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for r in d["rooms"]:
            r["permitted_hours"] = [list(w) for w in r["permitted_hours"]]
            r["zone"] = list(r["zone"])
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "BuildingConfig":
        try:
            rooms = [
                Room(
                    name=r["name"],
                    restricted=bool(r.get("restricted", False)),
                    permitted_hours=[tuple(w) for w in r.get("permitted_hours", [])],
                    door_status=r.get("door_status", "closed"),
                    motion_sensor_enabled=bool(r.get("motion_sensor_enabled", True)),
                    motion_detected=bool(r.get("motion_detected", False)),
                    zone=tuple(r.get("zone", (0.0, 0.0, 1.0, 1.0))),
                )
                for r in d.get("rooms", [])
            ]
            cfg = BuildingConfig(
                building_name=d.get("building_name", "Demo Smart Building"),
                open_time=d.get("open_time", "08:00"),
                close_time=d.get("close_time", "20:00"),
                force_closed=bool(d.get("force_closed", False)),
                rules=RuleSettings(**d.get("rules", {})),
                rooms=rooms,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Malformed building configuration: {exc}") from exc
        cfg.validate()
        return cfg


class ZoneManager:
    """Holds the live building configuration and per-room simulated state."""

    def __init__(self, config: BuildingConfig | None = None):
        self.config = config or BuildingConfig.default()
        self.config.validate()
        self.load_warning: str | None = None

    # ---- persistence -------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "ZoneManager":
        """Load config from JSON; fall back to defaults (with a warning) if missing/invalid."""
        zm = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                zm = cls(BuildingConfig.from_dict(json.load(fh)))
        except FileNotFoundError:
            zm.load_warning = f"{Path(path).name} not found; using built-in defaults."
        except (json.JSONDecodeError, ConfigError, OSError) as exc:
            zm.load_warning = f"Could not read {Path(path).name} ({exc}); using built-in defaults."
        if zm.load_warning:
            log.warning(zm.load_warning)
        return zm

    def save(self, path: Path) -> None:
        """Atomically write the configuration to JSON."""
        self.config.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.config.to_dict(), fh, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ---- queries -----------------------------------------------------------------------
    @property
    def rooms(self) -> list[Room]:
        return self.config.rooms

    @property
    def room_names(self) -> list[str]:
        return [r.name for r in self.config.rooms]

    @property
    def restricted_rooms(self) -> list[Room]:
        return [r for r in self.config.rooms if r.restricted]

    def get_room(self, name: str) -> Room:
        for r in self.config.rooms:
            if r.name == name:
                return r
        raise KeyError(f"Unknown room '{name}'")

    def building_closed(self, dt: datetime) -> bool:
        if self.config.force_closed:
            return True
        return not in_window(dt.hour * 60 + dt.minute, self.config.open_time, self.config.close_time)

    @staticmethod
    def point_in_zone(room: Room, x: float, y: float) -> bool:
        """x, y are normalised (0-1) frame coordinates."""
        x1, y1, x2, y2 = room.zone
        return x1 <= x <= x2 and y1 <= y <= y2

    # ---- simulated state ---------------------------------------------------------------
    def apply_observation(self, room_name: str, event_type: str) -> None:
        """Update the simulated door/motion state shown on the dashboard."""
        try:
            room = self.get_room(room_name)
        except KeyError:
            return
        if event_type == "door_open":
            room.door_status = "open"
        elif event_type == "door_close":
            room.door_status = "closed"
        elif event_type == "motion":
            room.motion_detected = True
