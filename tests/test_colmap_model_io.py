import tempfile
import unittest
from pathlib import Path

from scripts.quadric_slam_pipeline.colmap_model_io import (
    CameraModel,
    ColmapImage,
    ColmapTextModel,
    densify_colmap_poses,
    load_colmap_text_model,
    write_colmap_text_model,
)


class ColmapModelIoTest(unittest.TestCase):
    def test_load_colmap_text_model_parses_camera_and_registered_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "cameras.txt").write_text(
                "\n".join(
                    [
                        "# Camera list",
                        "1 PINHOLE 640 480 500 510 320 240",
                    ]
                )
            )
            (model_dir / "images.txt").write_text(
                "\n".join(
                    [
                        "# Image list",
                        "7 1 0 0 0 0.1 0.2 0.3 1 000001.jpg",
                        "",
                        "8 0.707 0 0.707 0 1 2 3 1 images/000002.jpg",
                        "0 0 -1",
                    ]
                )
            )

            model = load_colmap_text_model(model_dir)

        self.assertEqual(set(model.cameras), {1})
        camera = model.cameras[1]
        self.assertEqual(camera.model, "PINHOLE")
        self.assertEqual(camera.width, 640)
        self.assertEqual(camera.height, 480)
        self.assertEqual(camera.fx, 500)
        self.assertEqual(camera.fy, 510)
        self.assertEqual(camera.cx, 320)
        self.assertEqual(camera.cy, 240)

        self.assertEqual([image.frame_id for image in model.images], [1, 2])
        self.assertEqual(model.images[0].qvec_wxyz, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(model.images[0].tvec, (0.1, 0.2, 0.3))
        self.assertEqual(model.images[1].file_name, "images/000002.jpg")

    def test_densify_colmap_poses_interpolates_between_keyframes(self):
        model = ColmapTextModel(
            cameras={
                1: CameraModel(
                    camera_id=1,
                    model="PINHOLE",
                    width=800,
                    height=600,
                    params=(500.0, 510.0, 320.0, 240.0),
                    fx=500.0,
                    fy=510.0,
                    cx=320.0,
                    cy=240.0,
                )
            },
            images=[
                ColmapImage(1, 1, "000001.jpg", 1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                ColmapImage(21, 21, "000021.jpg", 1, (1.0, 0.0, 0.0, 0.0), (2.0, 4.0, 6.0)),
            ],
        )

        dense = densify_colmap_poses(model, [1, 11, 21])

        self.assertEqual([image.frame_id for image in dense.images], [1, 11, 21])
        self.assertEqual(dense.images[0].pose_source, "colmap_keyframe")
        self.assertEqual(dense.images[1].pose_source, "interpolated_se3_slerp_lerp")
        self.assertEqual(dense.images[1].file_name, "000011.jpg")
        self.assertEqual(dense.images[1].tvec, (1.0, 2.0, 3.0))
        self.assertEqual(dense.images[2].pose_source, "colmap_keyframe")

    def test_write_and_load_colmap_text_model_preserves_pose_source(self):
        model = ColmapTextModel(
            cameras={
                1: CameraModel(
                    camera_id=1,
                    model="PINHOLE",
                    width=800,
                    height=600,
                    params=(500.0, 510.0, 320.0, 240.0),
                    fx=500.0,
                    fy=510.0,
                    cx=320.0,
                    cy=240.0,
                )
            },
            images=[
                ColmapImage(
                    11,
                    11,
                    "000011.jpg",
                    1,
                    (1.0, 0.0, 0.0, 0.0),
                    (1.0, 2.0, 3.0),
                    pose_source="interpolated_se3_slerp_lerp",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_colmap_text_model(model, tmp)
            loaded = load_colmap_text_model(tmp)

        self.assertEqual(loaded.images[0].pose_source, "interpolated_se3_slerp_lerp")
        self.assertEqual(loaded.images[0].frame_id, 11)


if __name__ == "__main__":
    unittest.main()
