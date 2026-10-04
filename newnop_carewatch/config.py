from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    cfg = json.loads(Path(path or ROOT / "config.json").read_text(encoding="utf-8"))
    model = Path(cfg["pose_model"])
    cfg["pose_model"] = str(model if model.is_absolute() else ROOT / model)
    return cfg


def bed_path(video: str | Path) -> Path:
    return ROOT / "data" / "bed_regions" / f"{Path(video).stem}.json"


def output_path(video: str | Path) -> Path:
    return ROOT / "outputs" / Path(video).stem


def truth_path(video: str | Path, kind: str) -> Path:
    return ROOT / "data" / "ground_truth" / f"{Path(video).stem}_{kind}.csv"
