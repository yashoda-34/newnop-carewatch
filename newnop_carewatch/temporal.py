from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .models import AWAY, IN_BED, STATES, Event, Frame, Segment, State

S = State

ALLOWED = {
    S.LYING_IN_BED: {S.SITTING_ON_BED, S.UNKNOWN},
    S.SITTING_ON_BED: {S.LYING_IN_BED, S.STANDING, S.SITTING_OUTSIDE_BED, S.UNKNOWN},
    S.SITTING_OUTSIDE_BED: {S.STANDING, S.SITTING_ON_BED, S.UNKNOWN},
    S.STANDING: {S.WALKING, S.SITTING_ON_BED, S.SITTING_OUTSIDE_BED, S.OUT_OF_BED, S.UNKNOWN},
    S.WALKING: {S.STANDING, S.OUT_OF_BED, S.SITTING_ON_BED, S.SITTING_OUTSIDE_BED, S.UNKNOWN},
    S.LYING_OUTSIDE_BED: {S.STANDING, S.SITTING_OUTSIDE_BED, S.UNKNOWN},
    S.OUT_OF_BED: {S.WALKING, S.STANDING, S.UNKNOWN},
    S.UNKNOWN: set(STATES),
}


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-50, min(50, value))))


def score(frame: Frame, cfg: dict, was_in_bed: bool) -> np.ndarray:
    """Probability of every state for one frame, from a few physical measurements."""
    p = defaultdict(lambda: .001)
    if not frame.visible:
        # Hidden after being in bed is most likely blanket or occlusion; hidden after being up is most likely gone.
        if was_in_bed:
            p[S.LYING_IN_BED], p[S.SITTING_ON_BED], p[S.UNKNOWN] = .25, .2, .55
        else:
            p[S.OUT_OF_BED], p[S.UNKNOWN] = .45, .55
    else:
        box = sigmoid(((frame.box_aspect or 0) - cfg["lying_box_aspect"]) / .15)
        lying = box if frame.torso_angle is None else .75 * sigmoid((frame.torso_angle - cfg["lying_angle"]) / 8) + .25 * box
        upright = 1 - lying
        on_bed = sigmoid((frame.in_bed - .5) / .1)
        knees = .5 if frame.knee_angle is None else sigmoid((cfg["sitting_knee_angle"] - frame.knee_angle) / 12)
        moving = (frame.speed or 0) >= cfg["walking_speed"] * .7
        sitting = knees if moving else max(knees, .75 if on_bed > .6 else 0)
        walking = .25 if frame.speed is None else sigmoid((frame.speed - cfg["walking_speed"]) / .06)
        standing = upright * (1 - sitting)
        p.update({
            S.LYING_IN_BED: lying * on_bed,
            S.LYING_OUTSIDE_BED: lying * (1 - on_bed),
            S.SITTING_ON_BED: upright * sitting * on_bed,
            S.SITTING_OUTSIDE_BED: upright * sitting * (1 - on_bed),
            S.STANDING: standing * (1 - walking),
            S.WALKING: standing * walking,
        })
        # Weak keypoints (blanket, darkness, occlusion) move weight to UNKNOWN instead of forcing a guess.
        quality = min(1, max(0, (frame.keypoint_quality - .1) / .35))
        total = sum(p.values())
        for state in list(p):
            p[state] = quality * p[state] / total
        p[S.UNKNOWN] += 1 - quality
    result = np.array([p[state] for state in STATES]) + 1e-4
    return result / result.sum()


def smooth(frames: list[Frame], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi: prefer staying in the same state and forbid impossible jumps such as lying -> walking."""
    rows, was_in_bed = [], False
    for frame in frames:
        rows.append(score(frame, cfg, was_in_bed))
        if frame.visible and frame.keypoint_quality >= cfg["weak_evidence_quality"]:
            was_in_bed = frame.in_bed >= .5
    emission = np.asarray(rows)
    if not len(emission):
        return np.zeros(0, dtype=int), emission

    n, stay = len(STATES), .94
    move = np.full((n, n), 1e-8)
    for i, state in enumerate(STATES):
        move[i, i] = stay
        targets = ALLOWED[state] - {state}
        for target in targets:
            move[i, STATES.index(target)] = (1 - stay) / len(targets)
        move[i] /= move[i].sum()

    best = np.log(emission[0]) - math.log(n)
    back = np.zeros(emission.shape, dtype=int)
    for t in range(1, len(emission)):
        options = best[:, None] + np.log(move)
        back[t] = options.argmax(axis=0)
        best = options.max(axis=0) + np.log(emission[t])
    path = np.empty(len(emission), dtype=int)
    path[-1] = best.argmax()
    for t in range(len(path) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, emission


def join(segments: list[Segment]) -> list[Segment]:
    result: list[Segment] = []
    for seg in segments:
        if result and result[-1].state == seg.state:
            last = result[-1]
            total = last.duration + seg.duration
            last.confidence = (last.confidence * last.duration + seg.confidence * seg.duration) / total if total else seg.confidence
            last.end = seg.end
            last.note = last.note or seg.note
            if seg.source == "agent":
                last.source = "agent"
        else:
            result.append(seg)
    return result


def make_segments(frames: list[Frame], path: np.ndarray, emission: np.ndarray,
                  duration: float, minimum: float) -> list[Segment]:
    if not len(path):
        return [Segment(0, duration, S.UNKNOWN, 0)]
    times = [f.time for f in frames] + [duration]
    segments, start = [], 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            k = int(path[start])
            segments.append(Segment(times[start], times[i], STATES[k], float(emission[start:i, k].mean())))
            start = i
    segments = join(segments)
    # Blips shorter than `minimum` take the state of their longer neighbour.
    while len(segments) > 1:
        short = next((i for i, s in enumerate(segments) if s.duration < minimum), None)
        if short is None:
            break
        if short == 0:
            other = 1
        elif short == len(segments) - 1:
            other = short - 1
        else:
            other = short - 1 if segments[short - 1].duration >= segments[short + 1].duration else short + 1
        segments[short].state, segments[short].confidence = segments[other].state, segments[other].confidence
        segments = join(segments)
    return segments


def mean_confidence(segments: list[Segment]) -> float:
    total = sum(s.duration for s in segments)
    return sum(s.confidence * s.duration for s in segments) / total if total else 0.0


def bed_events(segments: list[Segment], cfg: dict) -> list[Event]:
    """Four-state machine: in -> exit_pending -> out -> return_pending -> in."""
    events: list[Event] = []
    first = next((s.state for s in segments if s.state != S.UNKNOWN), S.LYING_IN_BED)
    status = "in" if first in IN_BED else "out"
    start, previous, pending = 0.0, first, []

    for seg in segments:
        state = seg.state
        if state == S.UNKNOWN:
            continue
        if status == "in" and state not in IN_BED:
            status, start, pending = "exit_pending", seg.start, []
        elif status == "out" and state in IN_BED:
            status, start, pending = "return_pending", seg.start, []
        elif status == "exit_pending" and state in IN_BED:
            status = "in"                           # stood briefly and sat back down
        elif status == "return_pending" and state not in IN_BED:
            status = "out"                          # touched the bed and got up again

        if status == "exit_pending":
            pending.append(seg)
            if state in AWAY - {S.STANDING} and seg.end - start >= cfg["exit_confirm_sec"]:
                events.append(Event("bed_exit", start, seg.start, previous, state, mean_confidence(pending),
                                    "MONITOR", "In bed before; then stood and moved away."))
                status = "out"
        elif status == "return_pending":
            pending.append(seg)
            if state == S.LYING_IN_BED or seg.end - start >= cfg["return_confirm_sec"]:
                events.append(Event("bed_return", start, seg.start, previous, state, mean_confidence(pending),
                                    "NORMAL", "Out of bed before; then came back and settled on the bed."))
                status = "in"
        if status in {"in", "out"}:
            previous = state
    return events


def add_time(segments: list[Segment]) -> dict[State, float]:
    totals = defaultdict(float)
    for seg in segments:
        totals[seg.state] += seg.duration
    return {state: totals[state] for state in STATES}
