"""python -m newnop_carewatch <command> --video <path>

  setup-bed   draw the mattress once per camera view
  analyze     YOLO pose -> timeline -> agent -> bed events -> decision, written to outputs/<video>/
  evaluate    compare outputs/<video>/ with data/ground_truth/<video>_*.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .agent import check_events, review
from .config import bed_path, load_config, output_path, truth_path
from .evaluate import load_events, load_segments, save_report, score_activity, score_events, time_errors
from .models import Event, Frame, State, clock
from .report import build_summary, decide, write_outputs
from .temporal import bed_events, make_segments, smooth


def get_frames(video: Path, bed: list, cfg: dict, fresh: bool) -> tuple[list[Frame], float]:
    """YOLO runs once per video; later runs reuse outputs/<video>/features.json."""
    cache = output_path(video) / "features.json"
    if cache.exists() and not fresh:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [Frame.load(row) for row in data["features"]], data["duration"]
    if not video.exists():
        raise FileNotFoundError(f"{video} not found and no cached features at {cache}")
    from .vision import read_video

    frames, duration = read_video(video, bed, cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"duration": duration, "features": [asdict(f) for f in frames]}), encoding="utf-8")
    return frames, duration


def analyze(video: Path, cfg: dict, fresh: bool = False) -> dict:
    from .vision import load_bed

    bed = load_bed(bed_path(video))
    frames, duration = get_frames(video, bed, cfg, fresh)
    path, emission = smooth(frames, cfg)
    segments = make_segments(frames, path, emission, duration, cfg["minimum_segment_sec"])
    segments, traces = review(segments, frames, duration, cfg)
    events, event_traces = check_events(segments, bed_events(segments, cfg), frames, duration, cfg)
    overall, reasons = decide(segments, events, cfg)
    summary = build_summary(segments, events, duration, overall, reasons)
    write_outputs(output_path(video), segments, events, traces + event_traces, summary)

    print("Timeline")
    for seg in segments:
        print(f"  {clock(seg.start)} - {clock(seg.end)}  {seg.state.value}")
    print("\nBed events")
    for event in events:
        print("  " + json.dumps(event.row()))
    if not events:
        print("  none")
    print(f"\nDecision: {overall}")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"\nOutputs: {output_path(video)}")
    return summary


def evaluate(video: Path, tolerance: float) -> dict:
    truth = load_segments(truth_path(video, "segments"))
    predicted = load_segments(output_path(video) / "segments.csv", predicted=True)
    rows = json.loads((output_path(video) / "events.json").read_text(encoding="utf-8"))["events"]
    events = [Event(r["event"], r["start_sec"], r["confirmed_sec"], State(r["previous_state"].upper()),
                    State(r["current_state"].upper()), r["confidence"], r["decision"]) for r in rows]
    duration = max(s.end for s in truth)
    metrics = {
        "video": video.stem,
        "activity": score_activity(truth, predicted, duration),
        "events": score_events(load_events(truth_path(video, "events")), events, tolerance),
        "durations": time_errors(truth, predicted),
    }
    save_report(output_path(video) / "evaluation", metrics)
    exits = metrics["events"]["bed_exit"]
    print(f"{video.stem}: accuracy {metrics['activity']['accuracy']:.1%}, "
          f"macro-F1 {metrics['activity']['macro_f1']:.2f}, "
          f"bed exits TP {exits['tp']} FP {exits['fp']} FN {exits['fn']}, "
          f"mean duration error {metrics['durations']['mean_error_sec']} s")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m newnop_carewatch")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup-bed").add_argument("--video", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--video", required=True)
    p.add_argument("--fresh", action="store_true", help="re-run YOLO even if features are cached")
    p = sub.add_parser("evaluate")
    p.add_argument("--video", required=True)
    p.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    video = Path(args.video)
    if args.command == "setup-bed":
        from .vision import setup_bed

        setup_bed(video, bed_path(video))
        print(f"Saved {bed_path(video)}")
    elif args.command == "analyze":
        analyze(video, cfg, args.fresh)
    else:
        evaluate(video, args.tolerance)


if __name__ == "__main__":
    main()
