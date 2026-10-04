from __future__ import annotations

import csv
import json
from pathlib import Path

from .agent import as_dicts
from .models import IN_BED, Event, Segment, State, Trace, clock, human
from .temporal import add_time

S = State


def out_periods(events: list[Event], duration: float) -> list[float]:
    """Length of every exit-to-return gap; an exit with no return runs to the end of the video."""
    start, periods = None, []
    for event in events:
        if event.event == "bed_exit" and start is None:
            start = event.start
        elif event.event == "bed_return" and start is not None:
            periods.append(event.start - start)
            start = None
    if start is not None:
        periods.append(duration - start)
    return periods


def still_out(events: list[Event]) -> bool:
    last = [e.event for e in events]
    return bool(last) and last[-1] == "bed_exit"


def decide(segments: list[Segment], events: list[Event], cfg: dict) -> tuple[str, list[str]]:
    alerts, monitors = [], []
    for seg in segments:
        if seg.state == S.LYING_OUTSIDE_BED and seg.duration >= cfg["lying_outside_alert_sec"]:
            alerts.append(f"Possible fall: lying outside the bed for {human(seg.duration)}.")
        if seg.state == S.OUT_OF_BED and seg.duration >= cfg["out_of_view_alert_sec"]:
            alerts.append(f"Out of camera view for {human(seg.duration)}.")
        if seg.state == S.SITTING_ON_BED and seg.duration >= cfg["sitting_monitor_sec"]:
            monitors.append(f"Sitting on the edge of the bed for {human(seg.duration)}.")
        if seg.state == S.UNKNOWN and seg.duration >= cfg["unknown_monitor_sec"]:
            monitors.append(f"Activity could not be determined for {human(seg.duration)}.")
    duration = segments[-1].end if segments else 0
    for period in out_periods(events, duration):
        if period >= cfg["out_of_bed_alert_sec"]:
            alerts.append(f"Out of bed for {human(period)} without returning.")
    if still_out(events):
        monitors.append("Bed exit with no return yet: higher fall risk until the person is back in bed.")
    if alerts:
        return "ALERT", alerts
    if monitors:
        return "MONITOR", monitors
    return "NORMAL", ["Lying, sitting, standing, or walking normally; no safety rule triggered."]


def build_summary(segments: list[Segment], events: list[Event], duration: float,
                  overall: str, reasons: list[str]) -> dict:
    totals = add_time(segments)
    seconds = {state.value.lower(): round(totals[state], 1) for state in State}
    in_bed = sum(totals[state] for state in IN_BED)
    exits = sum(e.event == "bed_exit" for e in events)
    returns = sum(e.event == "bed_return" for e in events)
    return {
        "summary": {
            "observation_duration_sec": round(duration, 1),
            "activity_duration_sec": seconds,
            "bed_exit_count": exits,
            "bed_return_count": returns,
            "total_in_bed_sec": round(in_bed, 1),
            "total_out_of_bed_sec": round(duration - in_bed, 1),
            "longest_out_of_bed_period_sec": round(max(out_periods(events, duration), default=0), 1),
            "final_state": segments[-1].state.value.lower() if segments else "unknown",
            "overall_decision": overall,
            "decision_reasons": reasons,
        },
        "readable": {
            "total_observation_time": human(duration),
            "activity_summary": {name: human(value) for name, value in seconds.items()},
            "bed_summary": {
                "time_in_bed": human(in_bed),
                "time_out_of_bed": human(duration - in_bed),
                "bed_exit_count": exits,
                "bed_return_count": returns,
            },
        },
    }


def write_outputs(folder: str | Path, segments: list[Segment], events: list[Event],
                  traces: list[Trace], summary: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    timeline = "".join(f"{clock(s.start)} – {clock(s.end)}  {s.state.value}\n" for s in segments)
    (folder / "timeline.txt").write_text(timeline, encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (folder / "events.json").write_text(json.dumps({"events": [e.row() for e in events]}, indent=2),
                                        encoding="utf-8")
    (folder / "agent_trace.json").write_text(json.dumps(as_dicts(traces), indent=2), encoding="utf-8")

    with (folder / "segments.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["start", "end", "state", "confidence", "source", "note"])
        writer.writeheader()
        writer.writerows(s.row() for s in segments)

    lines = ["# Agent trace", ""]
    for trace in traces:
        lines += [f"## {trace.item}", f"- Observation: {trace.observation}"]
        for step in trace.steps:
            lines.append(f"- Action: `{step['action']}`  Finding: `{json.dumps(step['finding'])}`")
        lines += [f"- Conclusion: {trace.conclusion}", ""]
    (folder / "agent_trace.md").write_text("\n".join(lines), encoding="utf-8")
