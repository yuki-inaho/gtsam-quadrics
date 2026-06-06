import json
import tempfile
import unittest
from pathlib import Path

from scripts.quadric_slam_pipeline import (
    build_observations_dataset,
    load_tracking_coco_observations,
)
from scripts.quadric_slam_pipeline.colmap_model_io import (
    CameraModel,
    ColmapImage,
    ColmapTextModel,
)


def _ann(ann_id, image_id, track_id, bbox, score=0.9):
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        "segmentation": [],
        "attributes": {"track_id": track_id, "score": score},
    }


def _depth_gated_coco():
    return {
        "licenses": [],
        "info": {
            "depth_gate": {
                "enabled": True,
                "max_depth_m": 1.5,
                "kept_tracks": 1,
                "dropped_tracks": 1,
                "tracks": {"10": {"representative_depth_m": 0.85}},
            }
        },
        "categories": [{"id": 1, "name": "tomato", "supercategory": "fruit"}],
        "images": [
            {"id": 1, "file_name": "000001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "000002.jpg", "width": 640, "height": 480},
            {"id": 3, "file_name": "000003.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            _ann(1, 1, 10, [10, 20, 30, 40], score=0.91),
            _ann(2, 2, 10, [11, 21, 31, 41], score=0.92),
            _ann(3, 3, 10, [12, 22, 32, 42], score=0.93),
            _ann(4, 1, -1, [1, 2, 3, 4], score=0.99),
        ],
    }


class TrackingObservationsTest(unittest.TestCase):
    def test_load_tracking_coco_observations_requires_depth_gate_metadata(self):
        coco = _depth_gated_coco()
        coco["info"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coco_good.json"
            path.write_text(json.dumps(coco))

            with self.assertRaisesRegex(ValueError, "depth_gate"):
                load_tracking_coco_observations(path)

    def test_load_tracking_coco_observations_converts_bbox_and_keeps_depth_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coco_good_depth15.json"
            path.write_text(json.dumps(_depth_gated_coco()))

            tracking = load_tracking_coco_observations(path)

        self.assertTrue(tracking.summary["depth_gated"])
        self.assertEqual(tracking.summary["depth_gate"]["max_depth_m"], 1.5)
        self.assertEqual(tracking.summary["observation_count"], 3)
        self.assertEqual(tracking.summary["skipped_annotations"], {"unestablished_track": 1})
        self.assertEqual(set(tracking.tracks), {10})

        first = tracking.observations[0]
        self.assertEqual(first.frame_id, 1)
        self.assertEqual(first.track_id, 10)
        self.assertEqual(first.bbox_xywh, (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(first.bbox_xyxy, (10.0, 20.0, 40.0, 60.0))
        self.assertEqual(first.score, 0.91)

    def test_build_observations_dataset_drops_unregistered_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coco_good_depth15.json"
            path.write_text(json.dumps(_depth_gated_coco()))
            tracking = load_tracking_coco_observations(path)

            dataset = build_observations_dataset(tracking, registered_frame_ids={1, 3})

        self.assertTrue(dataset["summary"]["depth_gated"])
        self.assertEqual(dataset["summary"]["input_measurements"], 3)
        self.assertEqual(dataset["summary"]["measurements"], 2)
        self.assertEqual(dataset["summary"]["dropped_unregistered_measurements"], 1)
        self.assertEqual([frame["frame_id"] for frame in dataset["frames"]], [1, 3])
        self.assertEqual(
            dataset["tracks"],
            [
                {
                    "track_id": 10,
                    "measurement_count": 2,
                    "representative_depth_m": 0.85,
                    "depth_gate_max_m": 1.5,
                }
            ],
        )
        self.assertEqual(
            [measurement["frame_id"] for measurement in dataset["measurements"]],
            [1, 3],
        )

    def test_build_observations_dataset_includes_colmap_camera_and_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coco_good_depth15.json"
            path.write_text(json.dumps(_depth_gated_coco()))
            tracking = load_tracking_coco_observations(path)
            colmap_model = ColmapTextModel(
                cameras={
                    1: CameraModel(
                        camera_id=1,
                        model="PINHOLE",
                        width=800,
                        height=600,
                        params=(553.0, 560.0, 412.0, 294.0),
                        fx=553.0,
                        fy=560.0,
                        cx=412.0,
                        cy=294.0,
                    )
                },
                images=[
                    ColmapImage(1, 1, "000001.jpg", 1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                    ColmapImage(3, 3, "000003.jpg", 1, (1.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0)),
                ],
            )

            dataset = build_observations_dataset(tracking, colmap_model=colmap_model)

        self.assertEqual(dataset["schema"], "quadric_slam_pipeline/observations/v1")
        self.assertEqual(dataset["camera"]["params"], [553.0, 560.0, 412.0, 294.0])
        self.assertEqual([frame["frame_id"] for frame in dataset["frames"]], [1, 3])
        self.assertEqual(dataset["frames"][0]["cam_from_world"]["qvec"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(dataset["frames"][0]["pose_source"], "colmap")
        self.assertEqual(dataset["summary"]["pose_source_counts"], {"colmap": 2})
        self.assertEqual(dataset["tracks"], [{"track_id": 10, "measurement_count": 2, "representative_depth_m": 0.85, "depth_gate_max_m": 1.5}])


if __name__ == "__main__":
    unittest.main()
