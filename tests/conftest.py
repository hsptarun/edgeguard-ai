from datetime import datetime

import pytest

from detection.intrusion_engine import IntrusionEngine, Observation
from detection.zone_manager import BuildingConfig, ZoneManager


@pytest.fixture
def zones() -> ZoneManager:
    return ZoneManager(BuildingConfig.default())


@pytest.fixture
def engine(zones) -> IntrusionEngine:
    return IntrusionEngine(zones)


def obs(room: str, etype: str, hhmm: str, sec: int = 0, source: str = "test", simulated: bool = True) -> Observation:
    h, m = map(int, hhmm.split(":"))
    return Observation(room, etype, datetime(2026, 3, 10, h, m, sec), source, simulated)
