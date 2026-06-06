"""Create dense per-frame COLMAP text poses from a keyframe text model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .colmap_model_io import densify_colmap_poses, load_colmap_text_model, write_colmap_text_model


def _frame_ids_from_image_dir(image_dir: Path) -> list[int]:
    frame_ids: list[int] = []
    for path in sorted(image_dir.glob("*.jpg")):
        if not path.stem.isdigit():
            continue
        frame_ids.append(int(path.stem))
    if not frame_ids:
        raise ValueError(f"no numeric .jpg frames found under {image_dir}")
    return frame_ids


def _frame_ids_from_count(frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("--frame-count must be positive")
    return list(range(1, frame_count + 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-text-model", required=True, help="keyframe COLMAP text model directory")
    parser.add_argument("--out", required=True, help="output dense text model directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-dir", help="directory containing numeric RGB frame jpg files")
    group.add_argument("--frame-count", type=int, help="dense frame ids are 1..N")
    args = parser.parse_args(argv)

    frame_ids = _frame_ids_from_image_dir(Path(args.image_dir)) if args.image_dir else _frame_ids_from_count(args.frame_count)
    model = load_colmap_text_model(args.colmap_text_model)
    dense = densify_colmap_poses(model, frame_ids)
    write_colmap_text_model(dense, args.out)

    sources: dict[str, int] = {}
    for image in dense.images:
        sources[image.pose_source] = sources.get(image.pose_source, 0) + 1
    summary = {
        "input_keyframes": len(model.images),
        "output_frames": len(dense.images),
        "frame_min": min(frame_ids),
        "frame_max": max(frame_ids),
        "pose_sources": sources,
    }
    summary_path = Path(args.out) / "densify_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
