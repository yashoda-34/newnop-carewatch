from __future__ import annotations

from dataclasses import asdict

from .models import AWAY, IN_BED, Event, Frame, Segment, State, Trace

S = State


class Tools:
    """Temporal tools the agent calls when one moment is not enough. Every result goes into the trace."""

    def __init__(self, segments: list[Segment], frames: list[Frame], duration: float):
        self.segments, self.frames, self.duration = segments, frames, duration

    def look_back(self, start: float, seconds: float) -> dict:
        return self.timeline(max(0.0, start - seconds), start)

    def look_ahead(self, end: float, seconds: float) -> dict:
        return self.timeline(end, min(self.duration, end + seconds))

    def timeline(self, start: float, end: float) -> dict:
        rows = [{"state": s.state.value, "seconds": round(min(s.end, end) - max(s.start, start), 1)}
                for s in self.segments if s.end > start and s.start < end]
        return {"from": round(start, 1), "to": round(end, 1), "timeline": rows,
                "features": self.get_features(start, end)}

    def get_features(self, start: float, end: float) -> dict:
        rows = [f for f in self.frames if start <= f.time < end]
        seen = [f for f in rows if f.visible]

        def mean(name: str):
            values = [getattr(f, name) for f in seen if getattr(f, name) is not None]
            return round(sum(values) / len(values), 2) if values else None

        return {
            "samples": len(rows),
            "visible_fraction": round(len(seen) / len(rows), 2) if rows else 0,
            "in_bed": mean("in_bed"),
            "distance_to_bed": mean("distance_to_bed"),
            "speed": mean("speed"),
            "keypoint_quality": mean("keypoint_quality"),
            "low_light": round(sum(f.low_light for f in rows) / len(rows), 2) if rows else 0,
        }


def states(window: dict) -> list[State]:
    return [State(row["state"]) for row in window["timeline"]]


def review(segments: list[Segment], frames: list[Frame], duration: float,
           cfg: dict) -> tuple[list[Segment], list[Trace]]:
    """Observe an unclear segment, call the tools for context, then conclude."""
    tools = Tools(segments, frames, duration)
    window = cfg["agent_context_sec"]
    traces: list[Trace] = []
    for seg in segments:
        unclear = seg.state in {S.UNKNOWN, S.LYING_OUTSIDE_BED} or seg.confidence < .45
        if not unclear:
            continue
        trace = Trace(f"{seg.start:.1f}-{seg.end:.1f}",
                      f"Timeline says {seg.state.value} with confidence {seg.confidence:.2f}. "
                      "This moment alone is not enough, so look at the context.")
        before, after = tools.look_back(seg.start, window), tools.look_ahead(seg.end, window)
        inside = tools.get_features(seg.start, seg.end)
        trace.add("look_back", before)
        trace.add("look_ahead", after)
        trace.add("get_features", inside)
        previous = (states(before) or [None])[-1]
        following = (states(after) or [None])[0]
        state, confidence, reason = seg.state, seg.confidence, "Not enough evidence; keep the timeline result."

        if previous is not None and previous == following and seg.duration <= 12:
            state, confidence = previous, .65
            reason = f"Short gap between two {previous.value} segments (blanket, turn, or brief occlusion)."
        elif seg.state == S.LYING_OUTSIDE_BED and ((inside["in_bed"] or 0) >= .3
                                                     or (inside["distance_to_bed"] or 99) <= .15):
            state, confidence = S.LYING_IN_BED, .6
            reason = "Lying horizontally, but the body overlaps the bed region: lying normally in bed, not on the floor."
        elif inside["visible_fraction"] < .25 and previous in {S.STANDING, S.WALKING}:
            state, confidence = S.OUT_OF_BED, .65
            reason = "Was upright, then disappeared from view: left the camera view."

        if state != seg.state:
            seg.state, seg.confidence, seg.source, seg.note = state, confidence, "agent", reason
        trace.conclusion = f"{state.value}: {reason}"
        traces.append(trace)

    merged: list[Segment] = []
    for seg in segments:
        if merged and merged[-1].state == seg.state:
            merged[-1].end = seg.end
            merged[-1].confidence = (merged[-1].confidence + seg.confidence) / 2
            merged[-1].note = merged[-1].note or seg.note
        else:
            merged.append(seg)
    return merged, traces


def check_events(segments: list[Segment], events: list[Event], frames: list[Frame],
                 duration: float, cfg: dict) -> tuple[list[Event], list[Trace]]:
    """Confirm each proposed bed event from the seconds before and after it."""
    tools = Tools(segments, frames, duration)
    window = cfg["agent_context_sec"]
    kept: list[Event] = []
    traces: list[Trace] = []
    for event in events:
        trace = Trace(f"{event.event} @ {event.start:.1f}s",
                      f"Timeline changes from {event.previous_state.value} to {event.current_state.value}. "
                      "One moment does not prove a bed event.")
        before = tools.look_back(event.start, window)
        after = tools.timeline(event.start, min(duration, event.confirmed + window))
        trace.add("look_back", before)
        trace.add("look_ahead", after)
        trace.add("get_features", tools.get_features(event.start, min(duration, event.confirmed + 1)))
        earlier, later = states(before), states(after)

        if event.event == "bed_exit" and any(s in IN_BED for s in earlier) and any(s in AWAY for s in later):
            path = " then ".join(dict.fromkeys(s.value for s in later))
            event.reason = (f"BED_EXIT confirmed. Was {earlier[-1].value} just before, then {path}. "
                            "Sitting up without moving away would have been rejected.")
            event.decision = "MONITOR"
            kept.append(event)
        elif event.event == "bed_return" and any(s in AWAY for s in earlier) and (
                any(s in IN_BED for s in later) or event.current_state in IN_BED):
            event.reason = "RETURN_TO_BED confirmed. Was out of bed, then came back and settled on the bed."
            event.decision = "NORMAL"
            kept.append(event)
        else:
            event.reason = (f"{event.event.upper()} rejected. The seconds before and after do not show "
                            "a completed exit or return.")
        trace.conclusion = event.reason
        traces.append(trace)
    return kept, traces


def as_dicts(traces: list[Trace]) -> list[dict]:
    return [asdict(t) for t in traces]
