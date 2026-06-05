"""Build a minimal GTSAM Quadrics factor graph from 2-D track observations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuadricResult:
    track_id: int
    status: str
    center_xyz: tuple[float, float, float]
    radii_xyz: tuple[float, float, float]
    mean_bbox_xyxy: tuple[float, float, float, float]
    bbox_center_jitter_px: float
    measurement_count: int
    factor_count: int
    representative_depth_m: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_gtsam_modules():
    import gtsam  # noqa: PLC0415
    import gtsam_quadrics  # noqa: PLC0415

    return gtsam, gtsam_quadrics


def _camera_params(dataset: dict[str, Any]) -> tuple[float, float, float, float]:
    camera = dataset.get("camera") or {}
    params = camera.get("params") or []
    if len(params) < 4:
        raise ValueError("observations dataset requires camera.params=[fx,fy,cx,cy]")
    return tuple(float(value) for value in params[:4])  # type: ignore[return-value]


def _track_depths(dataset: dict[str, Any]) -> dict[int, float]:
    depths: dict[int, float] = {}
    for track in dataset.get("tracks", []):
        track_id = int(track["track_id"])
        value = track.get("representative_depth_m")
        if value is not None:
            depths[track_id] = float(value)
    return depths


def _frame_pose(frame: dict[str, Any], gtsam):
    cam_from_world = frame.get("cam_from_world") or {}
    qvec = cam_from_world.get("qvec") or [1.0, 0.0, 0.0, 0.0]
    tvec = cam_from_world.get("tvec") or [0.0, 0.0, 0.0]
    if len(qvec) != 4 or len(tvec) != 3:
        raise ValueError(f"invalid cam_from_world pose for frame {frame!r}")
    rotation = gtsam.Rot3.Quaternion(float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3]))
    translation = gtsam.Point3(float(tvec[0]), float(tvec[1]), float(tvec[2]))
    return gtsam.Pose3(rotation, translation)


def _mean_bbox(measurements: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    boxes = np.asarray([m["bbox_xyxy"] for m in measurements], dtype=np.float64)
    return tuple(float(value) for value in np.mean(boxes, axis=0))  # type: ignore[return-value]


def _bbox_center_jitter_px(measurements: list[dict[str, Any]], mean_bbox: tuple[float, float, float, float]) -> float:
    mean_center = np.asarray([(mean_bbox[0] + mean_bbox[2]) / 2.0, (mean_bbox[1] + mean_bbox[3]) / 2.0])
    centers = []
    for measurement in measurements:
        xmin, ymin, xmax, ymax = [float(value) for value in measurement["bbox_xyxy"]]
        centers.append([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    if not centers:
        return 0.0
    distances = np.linalg.norm(np.asarray(centers, dtype=np.float64) - mean_center[None, :], axis=1)
    return float(np.mean(distances))


def _initial_quadric_from_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    depth_m: float,
    camera_params: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    fx, fy, cx, cy = camera_params
    xmin, ymin, xmax, ymax = bbox_xyxy
    center_u = (xmin + xmax) / 2.0
    center_v = (ymin + ymax) / 2.0
    width_px = max(1.0, xmax - xmin)
    height_px = max(1.0, ymax - ymin)

    x = (center_u - cx) * depth_m / fx
    y = (center_v - cy) * depth_m / fy
    z = depth_m
    rx = max(0.02, width_px * depth_m / (2.0 * fx))
    ry = max(0.02, height_px * depth_m / (2.0 * fy))
    rz = max(0.02, math.sqrt(rx * ry))
    return (x, y, z), (rx, ry, rz)


def run_quadric_reconstruction(
    observations: dict[str, Any],
    *,
    min_measurements: int = 3,
    box_sigma: float = 3.0,
    optimize: bool = False,
) -> dict[str, Any]:
    """Create quadric initial values and bbox factors for depth-gated tracks.

    The current implementation intentionally treats global scale as non-goal:
    track representative depth is used only to produce a plausible initial
    quadric for reprojection/debug visualization. If ``optimize`` is false, the
    returned result is the factor-graph-ready initialization.
    """
    gtsam, gtsam_quadrics = _load_gtsam_modules()
    camera_params = _camera_params(observations)
    fx, fy, cx, cy = camera_params
    calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    bbox_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([box_sigma] * 4, dtype=np.float64))
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1] * 6, dtype=np.float64))

    frame_by_id = {int(frame["frame_id"]): frame for frame in observations.get("frames", [])}
    measurements_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for measurement in observations.get("measurements", []):
        measurements_by_track[int(measurement["track_id"])].append(measurement)

    representative_depths = _track_depths(observations)
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    inserted_pose_keys: set[int] = set()
    prior_pose_key: int | None = None
    quadric_results: list[QuadricResult] = []
    skipped_tracks: list[dict[str, Any]] = []

    x_key = lambda frame_id: int(gtsam.symbol("x", frame_id))
    q_key = lambda index: int(gtsam.symbol("q", index))

    track_index = 0
    for track_id, track_measurements in sorted(measurements_by_track.items()):
        usable = [m for m in track_measurements if int(m["frame_id"]) in frame_by_id]
        if len(usable) < min_measurements:
            skipped_tracks.append(
                {"track_id": track_id, "reason": "too_few_measurements", "measurements": len(usable)}
            )
            continue
        depth_m = representative_depths.get(track_id)
        if depth_m is None:
            skipped_tracks.append({"track_id": track_id, "reason": "missing_representative_depth"})
            continue

        mean_bbox = _mean_bbox(usable)
        bbox_center_jitter_px = _bbox_center_jitter_px(usable, mean_bbox)
        center, radii = _initial_quadric_from_bbox(mean_bbox, depth_m, camera_params)
        quadric = gtsam_quadrics.ConstrainedDualQuadric(
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*center)),
            np.asarray(radii, dtype=np.float64),
        )
        q = q_key(track_index)
        quadric.addToValues(initial, q)

        factor_count = 0
        for measurement in usable:
            frame_id = int(measurement["frame_id"])
            x = x_key(frame_id)
            if x not in inserted_pose_keys:
                pose = _frame_pose(frame_by_id[frame_id], gtsam)
                initial.insert(x, pose)
                inserted_pose_keys.add(x)
                if prior_pose_key is None:
                    graph.add(gtsam.PriorFactorPose3(x, pose, pose_prior_noise))
                    prior_pose_key = x
            box = gtsam_quadrics.AlignedBox2(*[float(v) for v in measurement["bbox_xyxy"]])
            graph.add(gtsam_quadrics.BoundingBoxFactor(box, calibration, x, q, bbox_noise, "STANDARD"))
            factor_count += 1

        quadric_results.append(
            QuadricResult(
                track_id=track_id,
                status="initialized" if not optimize else "optimized_pending",
                center_xyz=center,
                radii_xyz=radii,
                mean_bbox_xyxy=mean_bbox,
                bbox_center_jitter_px=bbox_center_jitter_px,
                measurement_count=len(usable),
                factor_count=factor_count,
                representative_depth_m=depth_m,
            )
        )
        track_index += 1

    optimizer_status = "not_requested"
    if optimize and quadric_results:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(50)
        try:
            result_values = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
            optimizer_status = "success"
            optimized: list[QuadricResult] = []
            for index, item in enumerate(quadric_results):
                quadric = gtsam_quadrics.ConstrainedDualQuadric.getFromValues(result_values, q_key(index))
                center = tuple(float(v) for v in quadric.centroid())
                radii = tuple(float(v) for v in quadric.radii())
                optimized.append(
                    QuadricResult(
                        track_id=item.track_id,
                        status="optimized",
                        center_xyz=center,  # type: ignore[arg-type]
                        radii_xyz=radii,  # type: ignore[arg-type]
                        mean_bbox_xyxy=item.mean_bbox_xyxy,
                        bbox_center_jitter_px=item.bbox_center_jitter_px,
                        measurement_count=item.measurement_count,
                        factor_count=item.factor_count,
                        representative_depth_m=item.representative_depth_m,
                    )
                )
            quadric_results = optimized
        except Exception as exc:  # noqa: BLE001
            optimizer_status = f"failed:{type(exc).__name__}:{exc}"

    return {
        "schema": "quadric_slam_pipeline/quadrics/v1",
        "global_scale": "not_metric_non_goal",
        "summary": {
            "input_tracks": len(measurements_by_track),
            "output_quadrics": len(quadric_results),
            "skipped_tracks": len(skipped_tracks),
            "factor_count": int(graph.size()),
            "pose_count": len(inserted_pose_keys),
            "optimizer_status": optimizer_status,
            "reprojection_error_proxy": "bbox_center_jitter_px_nonmetric_keyframe_pose",
            "mean_bbox_center_jitter_px": float(np.mean([q.bbox_center_jitter_px for q in quadric_results]))
            if quadric_results
            else None,
        },
        "quadrics": [item.as_dict() for item in quadric_results],
        "skipped_tracks": skipped_tracks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run minimal quadric reconstruction from observations.")
    parser.add_argument("--observations", required=True, help="observations.json")
    parser.add_argument("--out", required=True, help="quadrics.json")
    parser.add_argument("--min-measurements", type=int, default=3)
    parser.add_argument("--box-sigma", type=float, default=3.0)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args(argv)

    observations = json.loads(Path(args.observations).read_text())
    result = run_quadric_reconstruction(
        observations,
        min_measurements=args.min_measurements,
        box_sigma=args.box_sigma,
        optimize=args.optimize,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
