from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum


class State(str, Enum):
    LYING_IN_BED = "LYING_IN_BED"
    SITTING_ON_BED = "SITTING_ON_BED"
    SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
    STANDING = "STANDING"
    WALKING = "WALKING"
    LYING_OUTSIDE_BED = "LYING_OUTSIDE_BED"
    OUT_OF_BED = "OUT_OF_BED"
    UNKNOWN = "UNKNOWN"


STATES = list(State)
IN_BED = {State.LYING_IN_BED, State.SITTING_ON_BED}
AWAY = {State.SITTING_OUTSIDE_BED, State.STANDING, State.WALKING, State.LYING_OUTSIDE_BED, State.OUT_OF_BED}


@dataclass
class Frame:
    time: float
    visible: bool
    track_id: int | None = None
    torso_angle: float | None = None
    box_aspect: float | None = None
    knee_angle: float | None = None
    speed: float | None = None
    in_bed: float = 0.0
    distance_to_bed: float | None = None
    keypoint_quality: float = 0.0
    low_light: bool = False

    @classmethod
    def load(cls, row: dict) -> "Frame":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in row.items() if k in names})

    def __post_init__(self) -> None:
        # OpenCV and NumPy return numpy scalars; plain types keep the cache JSON-friendly.
        for name, value in vars(self).items():
            if value is None:
                continue
            if name in {"visible", "low_light"}:
                setattr(self, name, bool(value))
            elif name == "track_id":
                setattr(self, name, int(value))
            else:
                setattr(self, name, round(float(value), 4))


@dataclass
class Segment:
    start: float
    end: float
    state: State
    confidence: float
    source: str = "temporal"
    note: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def row(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2), "state": self.state.value,
                "confidence": round(self.confidence, 3), "source": self.source, "note": self.note}


@dataclass
class Event:
    event: str
    start: float
    confirmed: float
    previous_state: State
    current_state: State
    confidence: float
    decision: str = "NORMAL"
    reason: str = ""

    def row(self) -> dict:
        return {
            "event": self.event,
            "start_time": clock(self.start, hours=True),
            "confirmed_time": clock(self.confirmed, hours=True),
            "previous_state": self.previous_state.value.lower(),
            "current_state": self.current_state.value.lower(),
            "confidence": round(self.confidence, 2),
            "decision": self.decision,
            "start_sec": round(self.start, 1),
            "confirmed_sec": round(self.confirmed, 1),
            "reason": self.reason,
        }


@dataclass
class Trace:
    item: str
    observation: str
    steps: list[dict] = field(default_factory=list)
    conclusion: str = ""

    def add(self, action: str, finding: dict) -> None:
        self.steps.append({"action": action, "finding": finding})


def clock(seconds: float, hours: bool = False) -> str:
    total = int(round(seconds))
    h, rest = divmod(total, 3600)
    if hours or h:
        return f"{h:02d}:{rest // 60:02d}:{rest % 60:02d}"
    return f"{rest // 60:02d}:{rest % 60:02d}"


def human(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"
