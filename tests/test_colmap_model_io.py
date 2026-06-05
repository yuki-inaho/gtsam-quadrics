import tempfile
import unittest
from pathlib import Path

from scripts.quadric_slam_pipeline.colmap_model_io import load_colmap_text_model


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


if __name__ == "__main__":
    unittest.main()
