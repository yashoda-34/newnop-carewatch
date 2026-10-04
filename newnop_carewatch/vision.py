from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from .models import Frame

SHOULDERS, HIPS, KNEES, ANKLES = (5, 6), (11, 12), (13, 14), (15, 16)


def load_bed(path: str | Path) -> list[list[float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["bed"]


def save_bed(path: str | Path, polygon: list[list[float]], size: tuple[int, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bed": polygon, "frame_size": list(size)}, indent=2), encoding="utf-8")


def setup_bed(video: str | Path, path: str | Path) -> None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read {video}")
    x, y, w, h = cv2.selectROI("Draw a rectangle around the mattress, then press Enter", frame, False, False)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise ValueError("No bed rectangle selected")
    save_bed(path, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], (frame.shape[1], frame.shape[0]))


def brighten(frame: np.ndarray) -> np.ndarray:
    light, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    light = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def joint(points: np.ndarray, conf: np.ndarray, pair: tuple[int, int], minimum: float):
    seen = [points[i] for i in pair if conf[i] >= minimum]
    return np.mean(seen, axis=0) if seen else None


def bend(a, b, c) -> float | None:
    if a is None or b is None or c is None:
        return None
    first, second = a - b, c - b
    length = np.linalg.norm(first) * np.linalg.norm(second)
    if length < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / length, -1, 1))))


def measure(box, points, conf, bed: np.ndarray, minimum: float) -> dict:
    shoulder = joint(points, conf, SHOULDERS, minimum)
    hip = joint(points, conf, HIPS, minimum)
    knee = joint(points, conf, KNEES, minimum)
    ankle = joint(points, conf, ANKLES, minimum)
    x1, y1, x2, y2 = box
    body = max(x2 - x1, y2 - y1, 1)
    anchor = hip if hip is not None else np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    torso = None
    if shoulder is not None and hip is not None:
        dx, dy = shoulder - hip
        torso = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

    grid = [(x, y) for y in (y1, (y1 + y2) / 2, y2) for x in (x1, (x1 + x2) / 2, x2)]
    overlap = sum(cv2.pointPolygonTest(bed, (float(x), float(y)), False) >= 0 for x, y in grid) / len(grid)
    anchor_inside = float(cv2.pointPolygonTest(bed, (float(anchor[0]), float(anchor[1])), False) >= 0)
    return {
        "anchor": anchor,
        "body": body,
        "torso": torso,
        "box_aspect": (x2 - x1) / max(y2 - y1, 1),
        "knee": bend(hip, knee, ankle),
        "in_bed": 0.7 * anchor_inside + 0.3 * overlap,
        "distance": max(0.0, -cv2.pointPolygonTest(bed, (float(anchor[0]), float(anchor[1])), True)) / body,
        "quality": float(np.mean(conf)),
    }


def read_video(video: str | Path, bed_polygon: list[list[float]], cfg: dict) -> tuple[list[Frame], float]:
    """Run YOLO pose + ByteTrack once and return one measurement row per sampled frame."""
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or cfg["sample_fps"])
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    step = max(1, round(fps / cfg["sample_fps"]))
    model = YOLO(cfg["pose_model"])
    bed = np.array(bed_polygon, dtype=np.float32)

    sampled: list[dict] = []
    bed_time: defaultdict[int, float] = defaultdict(float)
    index = 0
    while cap.grab():
        if index % step:
            index += 1
            continue
        ok, image = cap.retrieve()
        if not ok:
            break
        dark = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()) < cfg["low_light_brightness"]
        result = model.track(brighten(image) if dark else image, persist=True, tracker="bytetrack.yaml",
                             conf=cfg["detection_confidence"], verbose=False)[0]
        people = {}
        if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
            for track, box, points, conf in zip(result.boxes.id.cpu().numpy().astype(int),
                                                result.boxes.xyxy.cpu().numpy(),
                                                result.keypoints.xy.cpu().numpy(),
                                                result.keypoints.conf.cpu().numpy()):
                people[int(track)] = measure(box, points, conf, bed, cfg["keypoint_confidence"])
                bed_time[int(track)] += people[int(track)]["in_bed"]
        sampled.append({"time": index / fps, "people": people, "dark": dark})
        index += 1
    cap.release()

    # The patient is the track that spends the most time on the bed; caregivers and visitors are ignored.
    patient = max(bed_time, key=bed_time.get, default=None)
    history: deque = deque()
    frames = []
    for sample in sampled:
        person = sample["people"].get(patient)
        if person is None:
            frames.append(Frame(sample["time"], False, low_light=sample["dark"]))
            continue
        history.append((sample["time"], person["anchor"]))
        while len(history) > 1 and sample["time"] - history[0][0] > 1:
            history.popleft()
        elapsed = sample["time"] - history[0][0]
        speed = np.linalg.norm(person["anchor"] - history[0][1]) / elapsed / person["body"] if elapsed > 0 else None
        frames.append(Frame(
            time=sample["time"], visible=True, track_id=patient, torso_angle=person["torso"],
            box_aspect=person["box_aspect"], knee_angle=person["knee"], speed=speed,
            in_bed=person["in_bed"], distance_to_bed=person["distance"],
            keypoint_quality=person["quality"], low_light=sample["dark"],
        ))
    return frames, duration
