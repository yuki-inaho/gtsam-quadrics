"""Build the JSON-ready observation adapter output for quadric reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .colmap_model_io import ColmapTextModel, load_colmap_text_model
from .tracking_io import TrackingCocoData, TrackingObservation, load_tracking_coco_observations


def _measurement_dict(observation: TrackingObservation) -> dict[str, Any]:
    return {
        "annotation_id": observation.annotation_id,
        "image_id": observation.image_id,
        "frame_id": observation.frame_id,
        "file_name": observation.file_name,
        "track_id": observation.track_id,
        "category_id": observation.category_id,
        "bbox_xywh": list(observation.bbox_xywh),
        "bbox_xyxy": list(observation.bbox_xyxy),
        "score": observation.score,
    }


def _camera_dict(colmap_model: ColmapTextModel | None) -> dict[str, Any] | None:
    if colmap_model is None:
        return None
    if not colmap_model.cameras:
        raise ValueError("COLMAP model has no cameras")
    camera = colmap_model.cameras[sorted(colmap_model.cameras)[0]]
    return {
        "camera_id": camera.camera_id,
        "model": camera.model,
        "width": camera.width,
        "height": camera.height,
        "params": list(camera.params),
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
    }


def _colmap_frames(colmap_model: ColmapTextModel | None) -> dict[int, dict[str, Any]]:
    if colmap_model is None:
        return {}
    return {
        image.frame_id: {
            "frame_id": image.frame_id,
            "image_id": image.image_id,
            "file_name": image.file_name,
            "camera_id": image.camera_id,
            "registered": True,
            "pose_source": image.pose_source,
            "cam_from_world": {
                "qvec": list(image.qvec_wxyz),
                "tvec": list(image.tvec),
            },
        }
        for image in colmap_model.images
    }


def _depth_by_track(tracking: TrackingCocoData) -> dict[int, float]:
    tracks = tracking.summary.get("depth_gate", {}).get("tracks", {})
    depths: dict[int, float] = {}
    for key, value in tracks.items():
        if not isinstance(value, dict):
            continue
        depth = value.get("representative_depth_m")
        if depth is not None:
            depths[int(key)] = float(depth)
    return depths


def _pose_source_counts(frames: dict[int, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames.values():
        source = str(frame.get("pose_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def build_observations_dataset(
    tracking: TrackingCocoData,
    registered_frame_ids: set[int] | None = None,
    colmap_model: ColmapTextModel | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready observations dataset.

    ``registered_frame_ids`` represents COLMAP-registered frames. Measurements
    outside this set are dropped and counted; no interpolation or pose fallback
    is attempted. When ``colmap_model`` is provided, registered frame ids and
    camera poses are read from COLMAP ``images.txt``.
    """
    colmap_frames = _colmap_frames(colmap_model)
    if registered_frame_ids is None and colmap_frames:
        registered_frame_ids = set(colmap_frames)

    if registered_frame_ids is None:
        kept = list(tracking.observations)
        dropped_unregistered = 0
    else:
        kept = [obs for obs in tracking.observations if obs.frame_id in registered_frame_ids]
        dropped_unregistered = len(tracking.observations) - len(kept)

    frames: dict[int, dict[str, Any]] = {}
    track_counts: dict[int, int] = {}
    for obs in kept:
        frames[obs.frame_id] = colmap_frames.get(
            obs.frame_id,
            {
                "frame_id": obs.frame_id,
                "image_id": obs.image_id,
                "file_name": obs.file_name,
                "registered": obs.frame_id in (registered_frame_ids or {obs.frame_id}),
            },
        )
        track_counts[obs.track_id] = track_counts.get(obs.track_id, 0) + 1

    depths = _depth_by_track(tracking)
    return {
        "schema": "quadric_slam_pipeline/observations/v1",
        "camera": _camera_dict(colmap_model),
        "summary": {
            "input_tracking_path": tracking.input_path,
            "depth_gated": tracking.summary["depth_gated"],
            "depth_gate": tracking.summary["depth_gate"],
            "input_measurements": len(tracking.observations),
            "measurements": len(kept),
            "tracks": len(track_counts),
            "frames": len(frames),
            "dropped_unregistered_measurements": dropped_unregistered,
            "pose_source_counts": _pose_source_counts(frames),
        },
        "frames": [frames[key] for key in sorted(frames)],
        "tracks": [
            {
                "track_id": track_id,
                "measurement_count": count,
                "representative_depth_m": depths.get(track_id),
                "depth_gate_max_m": tracking.summary["depth_gate"].get("max_depth_m"),
            }
            for track_id, count in sorted(track_counts.items())
        ],
        "measurements": [_measurement_dict(obs) for obs in kept],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build quadric observations from COLMAP and COCO.")
    parser.add_argument("--colmap-text-model", required=True, help="directory containing cameras.txt/images.txt")
    parser.add_argument("--tracking-coco", required=True, help="coco_good_depth15.json")
    parser.add_argument("--out", required=True, help="observations.json")
    args = parser.parse_args(argv)

    colmap_model = load_colmap_text_model(args.colmap_text_model)
    tracking = load_tracking_coco_observations(args.tracking_coco)
    dataset = build_observations_dataset(tracking, colmap_model=colmap_model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=2, sort_keys=True))
    print(json.dumps(dataset["summary"], indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
