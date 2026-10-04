from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .models import Event, Segment, State
from .temporal import add_time


def load_segments(path: str | Path, predicted: bool = False) -> list[Segment]:
    start, end = ("start", "end") if predicted else ("start_sec", "end_sec")
    with Path(path).open(newline="", encoding="utf-8") as file:
        return [Segment(float(r[start]), float(r[end]), State(r["state"].upper()), float(r.get("confidence") or 1))
                for r in csv.DictReader(file)]


def load_events(path: str | Path) -> list[tuple[str, float]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [(r["event"], float(r["time_sec"])) for r in csv.DictReader(file)]


def state_at(segments: list[Segment], time: float) -> State:
    return next((s.state for s in segments if s.start <= time < s.end),
                segments[-1].state if segments else State.UNKNOWN)


def score_activity(truth: list[Segment], predicted: list[Segment], duration: float) -> dict:
    """One check per second of video."""
    seconds = [t + .5 for t in range(int(round(duration)))]
    y_true = [state_at(truth, t).value for t in seconds]
    y_pred = [state_at(predicted, t).value for t in seconds]
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusions = [{"actual": a, "predicted": p, "seconds": int(matrix[i, j])}
                  for i, a in enumerate(labels) for j, p in enumerate(labels) if i != j and matrix[i, j]]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": sorted(confusions, key=lambda r: -r["seconds"]),
    }


def score_events(truth: list[tuple[str, float]], predicted: list[Event], tolerance: float = 5) -> dict:
    """Greedy match within `tolerance` seconds, then precision / recall / false detections."""
    result = {}
    for kind in ("bed_exit", "bed_return"):
        actual = [t for name, t in truth if name == kind]
        guesses = [e for e in predicted if e.event == kind]
        pairs = sorted((min(abs(t - e.start), abs(t - e.confirmed)), i, j)
                       for i, t in enumerate(actual) for j, e in enumerate(guesses))
        used_a, used_g, errors = set(), set(), []
        for error, i, j in pairs:
            if error <= tolerance and i not in used_a and j not in used_g:
                used_a.add(i)
                used_g.add(j)
                errors.append(error)
        tp = len(errors)
        fp, fn = len(guesses) - tp, len(actual) - tp
        result[kind] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "mean_time_error_sec": round(sum(errors) / tp, 1) if tp else None,
        }
    return result


def time_errors(truth: list[Segment], predicted: list[Segment]) -> dict:
    real, guess = add_time(truth), add_time(predicted)
    rows = [{"state": s.value, "ground_truth_sec": round(real[s], 1), "predicted_sec": round(guess[s], 1),
             "error_sec": round(abs(real[s] - guess[s]), 1)}
            for s in State if real[s] or guess[s]]
    return {"per_state": rows, "mean_error_sec": round(sum(r["error_sec"] for r in rows) / len(rows), 1) if rows else 0}


def save_report(folder: str | Path, metrics: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    activity = metrics["activity"]
    lines = ["# Evaluation", "",
             f"- Accuracy: **{activity['accuracy']:.1%}**",
             f"- Macro-F1: **{activity['macro_f1']:.2f}**",
             f"- Mean duration error: **{metrics['durations']['mean_error_sec']} s**",
             "", "## Confusions", ""]
    lines += [f"- {r['actual']} predicted as {r['predicted']}: {r['seconds']} s"
              for r in activity["top_confusions"]] or ["- None"]
    lines += ["", "## Bed events", ""]
    for kind, r in metrics["events"].items():
        lines.append(f"- {kind}: TP {r['tp']}, FP {r['fp']}, FN {r['fn']}, "
                     f"precision {r['precision']}, recall {r['recall']}")
    lines += ["", "## Duration", "", "| State | Ground truth | Predicted | Error |", "|---|---|---|---|"]
    lines += [f"| {r['state']} | {r['ground_truth_sec']} s | {r['predicted_sec']} s | {r['error_sec']} s |"
              for r in metrics["durations"]["per_state"]]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
