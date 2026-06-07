import unittest

from scripts.quadric_slam_pipeline.bytetrack_json_to_tracking_coco import (
    convert_bytetrack_json_to_coco,
)


class ByteTrackJsonToTrackingCocoTest(unittest.TestCase):
    def test_convert_bytetrack_json_to_coco_recovers_score_and_xywh(self):
        tracking = {
            "tracker": "bytetrack",
            "frames": [
                {
                    "frame_index": 0,
                    "file_name": "000001.jpg",
                    "image_size_wh": [800, 600],
                    "tracks": [
                        {
                            "track_id": 42,
                            "source_detection_id": 7,
                            "bbox_xyxy": [10.0, 20.0, 40.0, 65.0],
                        }
                    ],
                }
            ],
        }
        prediction = {
            "model": "model.onnx",
            "conf": 0.1,
            "iou": 0.5,
            "nms": "lsnms",
            "frames": [
                {
                    "frame_index": 0,
                    "file_name": "000001.jpg",
                    "image_size_wh": [800, 600],
                    "detections": [
                        {
                            "id": 7,
                            "bbox_xyxy": [10.0, 20.0, 40.0, 65.0],
                            "score": 0.91,
                            "label": 0,
                            "label_name": "tomato",
                        }
                    ],
                }
            ],
        }

        coco = convert_bytetrack_json_to_coco(tracking=tracking, prediction=prediction)

        self.assertEqual(
            coco["images"],
            [{"id": 1, "file_name": "000001.jpg", "width": 800, "height": 600}],
        )
        self.assertEqual(
            coco["info"]["conversion_summary"],
            {"images": 1, "annotations": 1, "skipped": {}},
        )
        ann = coco["annotations"][0]
        self.assertEqual(ann["bbox"], [10.0, 20.0, 30.0, 45.0])
        self.assertEqual(ann["area"], 1350.0)
        self.assertEqual(
            ann["attributes"],
            {
                "track_id": 42,
                "source_detection_id": 7,
                "score": 0.91,
                "source_label": 0,
                "source_label_name": "tomato",
            },
        )


if __name__ == "__main__":
    unittest.main()
