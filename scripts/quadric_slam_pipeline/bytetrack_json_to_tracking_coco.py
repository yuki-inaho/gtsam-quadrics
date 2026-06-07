"""Convert ByteTrack JSON detections into depth-gate-ready COCO tracking JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _image_id(file_name: str, fallback: int) -> int:
    stem = Path(file_name).stem
    return int(stem) if stem.isdigit() else fallback


def _bbox_xywh_from_xyxy(
    bbox_xyxy: list[float],
    width: int,
    height: int,
) -> list[float] | None:
    if len(bbox_xyxy) != 4:
        return None
    xmin, ymin, xmax, ymax = [float(value) for value in bbox_xyxy]
    xmin = max(0.0, min(float(width), xmin))
    xmax = max(0.0, min(float(width), xmax))
    ymin = max(0.0, min(float(height), ymin))
    ymax = max(0.0, min(float(height), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def _score_index(prediction: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    by_frame: dict[tuple[int, int], dict[str, Any]] = {}
    for frame in prediction.get("frames", []):
        frame_index = int(frame.get("frame_index", 0))
        for det in frame.get("detections", []):
            det_id = det.get("id")
            if det_id is None:
                continue
            by_frame[(frame_index, int(det_id))] = det
    return by_frame


def convert_bytetrack_json_to_coco(
    *,
    tracking: dict[str, Any],
    prediction: dict[str, Any],
    tracking_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    category_id: int = 1,
    category_name: str = "tomato",
) -> dict[str, Any]:
    """Return a COCO-style tracking dataset.

    The input ByteTrack JSON stores ``bbox_xyxy`` and ``source_detection_id``.
    The companion detection JSON is used only to recover the detection score
    and label metadata before depth gating and quadric reconstruction.
    """
    detections_by_key = _score_index(prediction)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    ann_id = 1

    for fallback_index, frame in enumerate(tracking.get("frames", []), start=1):
        file_name = str(frame["file_name"])
        width, height = [int(value) for value in frame.get("image_size_wh", [0, 0])]
        image_id = _image_id(file_name, fallback_index)
        frame_index = int(frame.get("frame_index", fallback_index - 1))
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for track in frame.get("tracks", []):
            bbox_xywh = _bbox_xywh_from_xyxy(track.get("bbox_xyxy", []), width, height)
            if bbox_xywh is None:
                skipped["invalid_bbox"] = skipped.get("invalid_bbox", 0) + 1
                continue
            source_detection_id = track.get("source_detection_id")
            source_det = (
                detections_by_key.get((frame_index, int(source_detection_id)))
                if source_detection_id is not None
                else None
            )
            score = source_det.get("score") if source_det else None
            label = source_det.get("label") if source_det else None
            label_name = source_det.get("label_name") if source_det else None
            attributes: dict[str, Any] = {
                "track_id": int(track["track_id"]),
                "source_detection_id": source_detection_id,
            }
            if score is not None:
                attributes["score"] = float(score)
            if label is not None:
                attributes["source_label"] = int(label)
            if label_name is not None:
                attributes["source_label_name"] = str(label_name)

            area = bbox_xywh[2] * bbox_xywh[3]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox_xywh,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                    "attributes": attributes,
                }
            )
            ann_id += 1

    return {
        "licenses": [],
        "info": {
            "schema": "quadric_slam_pipeline/tracking_coco/v1",
            "source_tracking_json": str(tracking_path) if tracking_path else None,
            "source_prediction_json": str(prediction_path) if prediction_path else None,
            "tracker": tracking.get("tracker"),
            "prediction_model": prediction.get("model"),
            "prediction_conf": prediction.get("conf"),
            "prediction_iou": prediction.get("iou"),
            "nms": prediction.get("nms"),
            "conversion_summary": {
                "images": len(images),
                "annotations": len(annotations),
                "skipped": skipped,
            },
        },
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "fruit",
            }
        ],
        "images": images,
        "annotations": annotations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert ByteTrack JSON to COCO tracking JSON."
    )
    parser.add_argument("--tracking-json", required=True, type=Path, help="ByteTrack JSON")
    parser.add_argument(
        "--prediction-json",
        required=True,
        type=Path,
        help="Detection JSON with scores",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output COCO JSON")
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--category-name", default="tomato")
    args = parser.parse_args(argv)

    tracking = json.loads(args.tracking_json.read_text())
    prediction = json.loads(args.prediction_json.read_text())
    coco = convert_bytetrack_json_to_coco(
        tracking=tracking,
        prediction=prediction,
        tracking_path=args.tracking_json,
        prediction_path=args.prediction_json,
        category_id=args.category_id,
        category_name=args.category_name,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(coco, indent=2, sort_keys=True))
    print(json.dumps(coco["info"]["conversion_summary"], indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
