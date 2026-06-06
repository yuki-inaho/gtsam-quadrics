"""Minimal COLMAP text-model parser for quadric SLAM adapters.

This parser intentionally preserves COLMAP's world-to-camera pose fields as
read from ``images.txt``. Conversion to a GTSAM ``Pose3`` is a later pipeline
step and should be tested against reprojection behavior before being treated as
final.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CameraModel:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]
    fx: float | None
    fy: float | None
    cx: float | None
    cy: float | None


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    frame_id: int
    file_name: str
    camera_id: int
    qvec_wxyz: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    pose_source: str = "colmap"


@dataclass(frozen=True)
class ColmapTextModel:
    cameras: dict[int, CameraModel]
    images: list[ColmapImage]

    def as_dict(self) -> dict:
        return {
            "cameras": {str(k): asdict(v) for k, v in self.cameras.items()},
            "images": [asdict(image) for image in self.images],
        }


def _data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _non_comment_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]


def _intrinsics(
    model: str,
    params: tuple[float, ...],
) -> tuple[float | None, float | None, float | None, float | None]:
    if model == "SIMPLE_PINHOLE" and len(params) >= 3:
        f, cx, cy = params[:3]
        return f, f, cx, cy
    if model in {"PINHOLE", "OPENCV", "FULL_OPENCV"} and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    if model in {"SIMPLE_RADIAL", "RADIAL"} and len(params) >= 3:
        f, cx, cy = params[:3]
        return f, f, cx, cy
    return None, None, None, None


def _frame_id_from_name(file_name: str) -> int:
    stem = Path(file_name).stem
    if not stem.isdigit():
        raise ValueError(f"image file stem must be numeric for frame join: {file_name!r}")
    return int(stem)


def load_cameras_txt(path: str | Path) -> dict[int, CameraModel]:
    cameras: dict[int, CameraModel] = {}
    for line in _data_lines(Path(path)):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"invalid cameras.txt line: {line!r}")
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = tuple(float(value) for value in parts[4:])
        fx, fy, cx, cy = _intrinsics(model, params)
        cameras[camera_id] = CameraModel(
            camera_id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
    return cameras


def _normalize_quaternion(qvec: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in qvec))
    if norm <= 0:
        raise ValueError(f"invalid zero quaternion: {qvec}")
    return tuple(value / norm for value in qvec)


def _slerp_quaternion(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Spherical interpolation for COLMAP qvec=(w,x,y,z)."""
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = tuple(-value for value in q1)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(tuple((1.0 - alpha) * a + alpha * b for a, b in zip(q0, q1)))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return _normalize_quaternion(tuple(s0 * a + s1 * b for a, b in zip(q0, q1)))


def _lerp_vec3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    return tuple((1.0 - alpha) * av + alpha * bv for av, bv in zip(a, b))


def load_pose_sources_json(path: str | Path) -> dict[int, str]:
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"pose_sources.json must be an object: {path}")
    return {int(key): str(value) for key, value in payload.items()}


def load_images_txt(path: str | Path, *, pose_sources: dict[int, str] | None = None) -> list[ColmapImage]:
    lines = _non_comment_lines(Path(path))
    images: list[ColmapImage] = []
    pose_sources = pose_sources or {}
    idx = 0
    while idx < len(lines):
        if not lines[idx]:
            idx += 1
            continue
        parts = lines[idx].split()
        if len(parts) < 10:
            raise ValueError(f"invalid images.txt image line: {lines[idx]!r}")
        image_id = int(parts[0])
        file_name = parts[9]
        frame_id = _frame_id_from_name(file_name)
        images.append(
            ColmapImage(
                image_id=image_id,
                frame_id=frame_id,
                file_name=file_name,
                camera_id=int(parts[8]),
                qvec_wxyz=tuple(float(value) for value in parts[1:5]),
                tvec=tuple(float(value) for value in parts[5:8]),
                pose_source=pose_sources.get(frame_id, "colmap"),
            )
        )
        idx += 2
    return images


def load_colmap_text_model(model_dir: str | Path) -> ColmapTextModel:
    model_dir = Path(model_dir)
    cameras = load_cameras_txt(model_dir / "cameras.txt")
    pose_sources = load_pose_sources_json(model_dir / "pose_sources.json")
    images = load_images_txt(model_dir / "images.txt", pose_sources=pose_sources)
    missing_cameras = sorted({image.camera_id for image in images} - set(cameras))
    if missing_cameras:
        raise ValueError(f"images.txt references missing camera ids: {missing_cameras}")
    return ColmapTextModel(cameras=cameras, images=images)


def densify_colmap_poses(
    model: ColmapTextModel,
    frame_ids: list[int],
    *,
    image_name_format: str = "{frame_id:06d}.jpg",
    registered_pose_source: str = "colmap_keyframe",
    interpolated_pose_source: str = "interpolated_se3_slerp_lerp",
) -> ColmapTextModel:
    """Return a model with one pose per requested frame id.

    The interpolation operates on COLMAP's world-to-camera fields: quaternion
    SLERP for rotation and linear interpolation for translation. It is intended
    as a documented fallback when dense COLMAP registration is unavailable.
    """
    if not model.images:
        raise ValueError("cannot densify a COLMAP model with no registered images")
    by_frame = {image.frame_id: image for image in model.images}
    keyframes = sorted(model.images, key=lambda image: image.frame_id)
    dense_images: list[ColmapImage] = []
    key_idx = 0
    default_camera_id = keyframes[0].camera_id
    for frame_id in sorted(frame_ids):
        if frame_id in by_frame:
            image = by_frame[frame_id]
            dense_images.append(
                ColmapImage(
                    image_id=image.image_id,
                    frame_id=image.frame_id,
                    file_name=image.file_name,
                    camera_id=image.camera_id,
                    qvec_wxyz=_normalize_quaternion(image.qvec_wxyz),
                    tvec=image.tvec,
                    pose_source=registered_pose_source,
                )
            )
            continue
        while key_idx + 1 < len(keyframes) and keyframes[key_idx + 1].frame_id < frame_id:
            key_idx += 1
        if frame_id < keyframes[0].frame_id or frame_id > keyframes[-1].frame_id:
            nearest = min(keyframes, key=lambda image: abs(image.frame_id - frame_id))
            dense_images.append(
                ColmapImage(
                    image_id=frame_id,
                    frame_id=frame_id,
                    file_name=image_name_format.format(frame_id=frame_id),
                    camera_id=nearest.camera_id,
                    qvec_wxyz=_normalize_quaternion(nearest.qvec_wxyz),
                    tvec=nearest.tvec,
                    pose_source="nearest_colmap_keyframe",
                )
            )
            continue
        left = keyframes[key_idx]
        right = keyframes[key_idx + 1]
        span = right.frame_id - left.frame_id
        if span <= 0:
            raise ValueError(f"non-increasing keyframes around frame {frame_id}")
        alpha = (frame_id - left.frame_id) / span
        dense_images.append(
            ColmapImage(
                image_id=frame_id,
                frame_id=frame_id,
                file_name=image_name_format.format(frame_id=frame_id),
                camera_id=default_camera_id,
                qvec_wxyz=_slerp_quaternion(left.qvec_wxyz, right.qvec_wxyz, alpha),
                tvec=_lerp_vec3(left.tvec, right.tvec, alpha),
                pose_source=interpolated_pose_source,
            )
        )
    return ColmapTextModel(cameras=model.cameras, images=dense_images)


def write_colmap_text_model(model: ColmapTextModel, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    camera_lines = ["# Camera list", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for camera in sorted(model.cameras.values(), key=lambda item: item.camera_id):
        params = " ".join(f"{value:.17g}" for value in camera.params)
        camera_lines.append(f"{camera.camera_id} {camera.model} {camera.width} {camera.height} {params}")
    (model_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n")

    image_lines = ["# Image list", "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    pose_sources: dict[str, str] = {}
    for image in sorted(model.images, key=lambda item: item.frame_id):
        q = " ".join(f"{value:.17g}" for value in image.qvec_wxyz)
        t = " ".join(f"{value:.17g}" for value in image.tvec)
        image_lines.append(f"{image.image_id} {q} {t} {image.camera_id} {image.file_name}")
        image_lines.append("")
        pose_sources[str(image.frame_id)] = image.pose_source
    (model_dir / "images.txt").write_text("\n".join(image_lines) + "\n")
    (model_dir / "pose_sources.json").write_text(json.dumps(pose_sources, indent=2, sort_keys=True) + "\n")
