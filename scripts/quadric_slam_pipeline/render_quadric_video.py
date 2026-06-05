"""Render MOT and quadric reprojection boxes for visual inspection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


COLOR_TRACK = (0, 255, 255)
COLOR_QUADRIC = (255, 0, 255)


def _clip_box(
    bbox_xyxy: Iterable[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    xmin, ymin, xmax, ymax = [float(v) for v in bbox_xyxy]
    x0 = max(0, min(width - 1, int(round(xmin))))
    y0 = max(0, min(height - 1, int(round(ymin))))
    x1 = max(0, min(width - 1, int(round(xmax))))
    y1 = max(0, min(height - 1, int(round(ymax))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def draw_box(
    image: np.ndarray,
    bbox_xyxy: Iterable[float],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw an axis-aligned rectangle on an RGB image array."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB array")
    out = image.copy()
    height, width = out.shape[:2]
    clipped = _clip_box(bbox_xyxy, width, height)
    if clipped is None:
        return out
    x0, y0, x1, y1 = clipped
    t = max(1, int(thickness))
    out[y0 : min(height, y0 + t), x0 : x1 + 1] = color
    out[max(0, y1 - t + 1) : y1 + 1, x0 : x1 + 1] = color
    out[y0 : y1 + 1, x0 : min(width, x0 + t)] = color
    out[y0 : y1 + 1, max(0, x1 - t + 1) : x1 + 1] = color
    return out


def render_frame_array(
    image: np.ndarray,
    measurements: Iterable[dict[str, Any]],
    quadric_boxes: Iterable[dict[str, Any]] = (),
) -> np.ndarray:
    """Overlay measured boxes and optional quadric reprojection boxes."""
    rendered = image.copy()
    for measurement in measurements:
        rendered = draw_box(rendered, measurement["bbox_xyxy"], COLOR_TRACK, thickness=2)
    for quadric_box in quadric_boxes:
        rendered = draw_box(rendered, quadric_box["bbox_xyxy"], COLOR_QUADRIC, thickness=1)
    return rendered


def _camera_params(observations: dict[str, Any]) -> tuple[float, float, float, float]:
    params = (observations.get("camera") or {}).get("params") or []
    if len(params) < 4:
        raise ValueError("observations camera.params must contain fx,fy,cx,cy")
    return tuple(float(value) for value in params[:4])  # type: ignore[return-value]


def _frame_pose(frame: dict[str, Any], gtsam):
    cam_from_world = frame.get("cam_from_world") or {}
    qvec = cam_from_world.get("qvec") or [1.0, 0.0, 0.0, 0.0]
    tvec = cam_from_world.get("tvec") or [0.0, 0.0, 0.0]
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3])),
        gtsam.Point3(float(tvec[0]), float(tvec[1]), float(tvec[2])),
    )


def _mean_box(boxes: list[list[float]]) -> list[float]:
    return [float(v) for v in np.asarray(boxes, dtype=np.float64).mean(axis=0)]


def _valid_projection_box(bbox_xyxy: list[float], width: int, height: int) -> bool:
    if not all(math.isfinite(value) for value in bbox_xyxy):
        return False
    xmin, ymin, xmax, ymax = bbox_xyxy
    if xmax <= xmin or ymax <= ymin:
        return False
    if (xmax - xmin) < 2.0 or (ymax - ymin) < 2.0:
        return False
    return not (xmax < 0 or ymax < 0 or xmin >= width or ymin >= height)


def build_quadric_reprojection_boxes(
    observations: dict[str, Any],
    quadrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build per-frame quadric boxes for visual inspection.

    Real projection is attempted first through ``QuadricCamera.project``.  This
    dataset currently uses non-metric COLMAP keyframe poses, so some initialized
    quadrics project to tiny/out-of-view boxes. For those cases the renderer
    falls back to the track mean measured bbox as an explicit visualization
    proxy; the returned ``status`` records which path was used.
    """
    import gtsam  # noqa: PLC0415
    import gtsam_quadrics  # noqa: PLC0415

    width = int((observations.get("camera") or {}).get("width", 0))
    height = int((observations.get("camera") or {}).get("height", 0))
    fx, fy, cx, cy = _camera_params(observations)
    calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    frame_by_id = {int(frame["frame_id"]): frame for frame in observations.get("frames", [])}
    measurements_by_track: dict[int, list[dict[str, Any]]] = {}
    for measurement in observations.get("measurements", []):
        measurements_by_track.setdefault(int(measurement["track_id"]), []).append(measurement)

    results: list[dict[str, Any]] = []
    for quadric in quadrics.get("quadrics", []):
        track_id = int(quadric["track_id"])
        measurements = measurements_by_track.get(track_id, [])
        if not measurements:
            continue
        mean_bbox = _mean_box([m["bbox_xyxy"] for m in measurements])
        gtsam_quadric = gtsam_quadrics.ConstrainedDualQuadric(
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*[float(v) for v in quadric["center_xyz"]])),
            np.asarray(quadric["radii_xyz"], dtype=np.float64),
        )
        for measurement in measurements:
            frame_id = int(measurement["frame_id"])
            status = "projected"
            bbox_xyxy: list[float]
            try:
                conic = gtsam_quadrics.QuadricCamera.project(
                    gtsam_quadric,
                    _frame_pose(frame_by_id[frame_id], gtsam),
                    calibration,
                )
                bounds = conic.bounds()
                bbox_xyxy = [bounds.xmin(), bounds.ymin(), bounds.xmax(), bounds.ymax()]
                if not _valid_projection_box(bbox_xyxy, width, height):
                    status = "proxy_mean_bbox"
                    bbox_xyxy = mean_bbox
            except Exception:  # noqa: BLE001
                status = "proxy_mean_bbox"
                bbox_xyxy = mean_bbox
            results.append(
                {
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "bbox_xyxy": bbox_xyxy,
                    "status": status,
                }
            )
    return results


def _load_image_cv2(path: Path):
    try:
        import cv2  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for JPEG/MP4 rendering. Add opencv to Pixi or run in an environment with cv2."
        ) from exc
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_quadric_video(
    *,
    images_dir: str | Path,
    observations: dict[str, Any],
    out_path: str | Path,
    quadrics: dict[str, Any] | None = None,
    fps: float = 15.0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Render a simple mp4 with measured bboxes.

    Quadric reprojection boxes are accepted under
    ``observations['quadric_reprojections']`` when available. Until real
    reprojection is implemented, this still produces a MOT-box inspection video.
    """
    images_dir = Path(images_dir)
    out_path = Path(out_path)
    measurements_by_frame: dict[int, list[dict[str, Any]]] = {}
    for measurement in observations.get("measurements", []):
        measurements_by_frame.setdefault(int(measurement["frame_id"]), []).append(measurement)
    quadric_reprojections = (
        build_quadric_reprojection_boxes(observations, quadrics)
        if quadrics is not None
        else observations.get("quadric_reprojections", [])
    )
    quadric_by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in quadric_reprojections:
        quadric_by_frame.setdefault(int(item["frame_id"]), []).append(item)

    frames = sorted(observations.get("frames", []), key=lambda item: int(item["frame_id"]))
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("no frames to render")

    first_frame = frames[0]
    cv2, first = _load_image_cv2(images_dir / first_frame["file_name"])
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")

    frame_count = 0
    try:
        for frame in frames:
            frame_id = int(frame["frame_id"])
            _, image = _load_image_cv2(images_dir / frame["file_name"])
            rendered = render_frame_array(
                image,
                measurements_by_frame.get(frame_id, []),
                quadric_by_frame.get(frame_id, []),
            )
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            frame_count += 1
    finally:
        writer.release()

    return {
        "out_path": str(out_path),
        "frame_count": frame_count,
        "quadric_reprojection_boxes": len(quadric_reprojections),
        "width": width,
        "height": height,
        "fps": fps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render quadric SLAM reprojection video.")
    parser.add_argument("--images", required=True, help="directory with RGB frames")
    parser.add_argument("--observations", required=True, help="observations/reprojection JSON")
    parser.add_argument("--quadrics", help="optional quadrics.json to render projected/proxy quadric boxes")
    parser.add_argument("--out", required=True, help="output mp4")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)

    observations = json.loads(Path(args.observations).read_text())
    quadrics = json.loads(Path(args.quadrics).read_text()) if args.quadrics else None
    summary = render_quadric_video(
        images_dir=args.images,
        observations=observations,
        out_path=args.out,
        quadrics=quadrics,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
