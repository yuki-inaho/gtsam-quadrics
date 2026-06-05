import unittest

import numpy as np

from scripts.quadric_slam_pipeline.render_quadric_video import (
    COLOR_QUADRIC,
    COLOR_TRACK,
    draw_box,
    render_frame_array,
)


class QuadricVideoRenderTest(unittest.TestCase):
    def test_draw_box_marks_rectangle_edges(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)

        rendered = draw_box(image, [5, 4, 15, 12], COLOR_TRACK, thickness=2)

        self.assertTrue(np.all(rendered[4, 5] == COLOR_TRACK))
        self.assertTrue(np.all(rendered[12, 15] == COLOR_TRACK))
        self.assertTrue(np.all(rendered[8, 10] == [0, 0, 0]))
        self.assertTrue(np.all(image == 0), "draw_box must not mutate input")

    def test_render_frame_array_draws_measurement_and_quadric_boxes(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)

        rendered = render_frame_array(
            image,
            measurements=[{"bbox_xyxy": [2, 3, 10, 12]}],
            quadric_boxes=[{"bbox_xyxy": [14, 5, 22, 15]}],
        )

        self.assertTrue(np.all(rendered[3, 2] == COLOR_TRACK))
        self.assertTrue(np.all(rendered[5, 14] == COLOR_QUADRIC))
        self.assertTrue(np.all(image == 0), "render_frame_array must not mutate input")


if __name__ == "__main__":
    unittest.main()
