import unittest

from scripts.quadric_slam_pipeline.reconstruct_quadrics import run_quadric_reconstruction


def _observations():
    return {
        "schema": "quadric_slam_pipeline/observations/v1",
        "camera": {
            "model": "PINHOLE",
            "width": 800,
            "height": 600,
            "params": [553.0, 560.0, 412.0, 294.0],
        },
        "frames": [
            {"frame_id": 1, "file_name": "000001.jpg", "cam_from_world": {"qvec": [1, 0, 0, 0], "tvec": [0, 0, 0]}},
            {
                "frame_id": 2,
                "file_name": "000002.jpg",
                "cam_from_world": {"qvec": [1, 0, 0, 0], "tvec": [0.02, 0, 0]},
            },
            {
                "frame_id": 3,
                "file_name": "000003.jpg",
                "cam_from_world": {"qvec": [1, 0, 0, 0], "tvec": [0.04, 0, 0]},
            },
        ],
        "tracks": [{"track_id": 10, "measurement_count": 3, "representative_depth_m": 0.85}],
        "measurements": [
            {"frame_id": 1, "track_id": 10, "bbox_xyxy": [300, 220, 350, 280], "score": 0.9},
            {"frame_id": 2, "track_id": 10, "bbox_xyxy": [302, 221, 352, 281], "score": 0.91},
            {"frame_id": 3, "track_id": 10, "bbox_xyxy": [304, 222, 354, 282], "score": 0.92},
        ],
    }


class QuadricReconstructionTest(unittest.TestCase):
    def test_run_quadric_reconstruction_builds_factor_graph_ready_result(self):
        result = run_quadric_reconstruction(_observations(), optimize=False)

        self.assertEqual(result["schema"], "quadric_slam_pipeline/quadrics/v1")
        self.assertEqual(result["global_scale"], "not_metric_non_goal")
        self.assertEqual(result["summary"]["input_tracks"], 1)
        self.assertEqual(result["summary"]["output_quadrics"], 1)
        self.assertEqual(result["summary"]["factor_count"], 4)
        self.assertEqual(result["summary"]["pose_count"], 3)
        self.assertEqual(result["summary"]["optimizer_status"], "not_requested")
        self.assertEqual(result["summary"]["reprojection_error_proxy"], "bbox_center_jitter_px_nonmetric_keyframe_pose")
        self.assertIsNotNone(result["summary"]["mean_bbox_center_jitter_px"])

        quadric = result["quadrics"][0]
        self.assertEqual(quadric["track_id"], 10)
        self.assertEqual(quadric["measurement_count"], 3)
        self.assertEqual(quadric["factor_count"], 3)
        self.assertAlmostEqual(quadric["representative_depth_m"], 0.85)
        self.assertEqual(len(quadric["mean_bbox_xyxy"]), 4)
        self.assertGreaterEqual(quadric["bbox_center_jitter_px"], 0.0)
        self.assertGreater(quadric["radii_xyz"][0], 0.0)
        self.assertGreater(quadric["radii_xyz"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
