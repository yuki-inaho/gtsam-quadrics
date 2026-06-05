"""Minimal COLMAP text-model parser for quadric SLAM adapters.

This parser intentionally preserves COLMAP's world-to-camera pose fields as
read from ``images.txt``. Conversion to a GTSAM ``Pose3`` is a later pipeline
step and should be tested against reprojection behavior before being treated as
final.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


def load_images_txt(path: str | Path) -> list[ColmapImage]:
    lines = _non_comment_lines(Path(path))
    images: list[ColmapImage] = []
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
        images.append(
            ColmapImage(
                image_id=image_id,
                frame_id=_frame_id_from_name(file_name),
                file_name=file_name,
                camera_id=int(parts[8]),
                qvec_wxyz=tuple(float(value) for value in parts[1:5]),
                tvec=tuple(float(value) for value in parts[5:8]),
            )
        )
        idx += 2
    return images


def load_colmap_text_model(model_dir: str | Path) -> ColmapTextModel:
    model_dir = Path(model_dir)
    cameras = load_cameras_txt(model_dir / "cameras.txt")
    images = load_images_txt(model_dir / "images.txt")
    missing_cameras = sorted({image.camera_id for image in images} - set(cameras))
    if missing_cameras:
        raise ValueError(f"images.txt references missing camera ids: {missing_cameras}")
    return ColmapTextModel(cameras=cameras, images=images)
