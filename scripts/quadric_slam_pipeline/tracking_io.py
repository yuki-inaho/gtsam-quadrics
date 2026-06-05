"""Read depth-gated COCO tracking observations for the quadric SLAM pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

UNESTABLISHED = {-1, None}


@dataclass(frozen=True)
class TrackingObservation:
    annotation_id: int | None
    image_id: int
    frame_id: int
    file_name: str
    track_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    score: float | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrackingCocoData:
    input_path: str
    images_by_id: dict[int, dict]
    observations: list[TrackingObservation]
    tracks: dict[int, list[TrackingObservation]]
    summary: dict


def _track_id(ann: dict):
    return ann.get("attributes", {}).get("track_id")


def _score(ann: dict) -> float | None:
    value = ann.get("attributes", {}).get("score")
    return None if value is None else float(value)


def _frame_id_from_image(image: dict) -> int:
    stem = Path(image["file_name"]).stem
    if stem.isdigit():
        return int(stem)
    raise ValueError(f"image file stem must be numeric for frame join: {image['file_name']!r}")


def _bbox_xyxy(ann: dict) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    bbox = ann.get("bbox")
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"annotation has invalid bbox: {ann!r}")
    x, y, w, h = (float(value) for value in bbox)
    if w <= 0 or h <= 0:
        raise ValueError(f"annotation bbox must have positive width and height: {ann!r}")
    return (x, y, w, h), (x, y, x + w, y + h)


def _depth_gate_metadata(coco: dict) -> dict:
    info = coco.get("info") or {}
    depth_gate = info.get("depth_gate") or coco.get("depth_gate")
    return depth_gate if isinstance(depth_gate, dict) else {}


def load_tracking_coco_observations(
    path: str | Path,
    require_depth_gate: bool = True,
) -> TrackingCocoData:
    """Load ``coco_good_depth15.json`` and expose track-indexed observations.

    If ``require_depth_gate`` is True, the COCO must carry
    ``info.depth_gate.enabled == True``. This avoids silently feeding an
    ungated ``coco_good.json`` into the quadric pipeline.
    """
    path = Path(path)
    coco = json.loads(path.read_text())
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    depth_gate = _depth_gate_metadata(coco)
    if require_depth_gate and not depth_gate.get("enabled"):
        raise ValueError("tracking COCO is missing required info.depth_gate.enabled metadata")

    observations: list[TrackingObservation] = []
    skipped: dict[str, int] = {}
    for ann in coco.get("annotations", []):
        tid = _track_id(ann)
        if tid in UNESTABLISHED:
            skipped["unestablished_track"] = skipped.get("unestablished_track", 0) + 1
            continue

        image = images_by_id.get(int(ann.get("image_id", -1)))
        if image is None:
            skipped["missing_image_record"] = skipped.get("missing_image_record", 0) + 1
            continue

        bbox_xywh, bbox_xyxy = _bbox_xyxy(ann)
        observations.append(
            TrackingObservation(
                annotation_id=ann.get("id"),
                image_id=int(ann["image_id"]),
                frame_id=_frame_id_from_image(image),
                file_name=image["file_name"],
                track_id=int(tid),
                category_id=int(ann["category_id"]),
                bbox_xywh=bbox_xywh,
                bbox_xyxy=bbox_xyxy,
                score=_score(ann),
            )
        )

    tracks: dict[int, list[TrackingObservation]] = {}
    for observation in observations:
        tracks.setdefault(observation.track_id, []).append(observation)

    summary = {
        "input_path": str(path),
        "image_count": len(images_by_id),
        "annotation_count": len(coco.get("annotations", [])),
        "observation_count": len(observations),
        "track_count": len(tracks),
        "skipped_annotations": skipped,
        "depth_gate": depth_gate,
        "depth_gated": bool(depth_gate.get("enabled")),
    }
    return TrackingCocoData(
        input_path=str(path),
        images_by_id=images_by_id,
        observations=observations,
        tracks=tracks,
        summary=summary,
    )
