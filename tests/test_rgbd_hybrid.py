from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recon_pipeline.foundation_stereo import (
    ColmapCamera,
    ColmapImage,
    DepthView,
    RectifiedPair,
)
from recon_pipeline.rgbd_hybrid import (
    arbitrate_depth,
    build_parser,
    project_rectified_depth_to_camera,
    mask_depth_to_world_sphere,
    mask_depth_to_screen_target,
    grabcut_foreground_mask,
    validate_arguments,
    write_metric_colmap_model,
    warp_depth_to_camera,
)


try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


@unittest.skipUnless(np is not None and cv2 is not None, "hybrid optional dependencies missing")
class RGBDHybridGeometryTests(unittest.TestCase):
    def test_tof_wins_and_only_anchored_agreeing_stereo_fills(self) -> None:
        tof = np.zeros((5, 5), dtype=np.float32)
        tof[2, 2] = 0.5
        first = np.zeros_like(tof)
        second = np.zeros_like(tof)
        first[2, 2] = 0.7
        second[2, 2] = 0.7
        first[2, 3] = 0.505
        second[2, 3] = 0.506
        first[1, 2] = 0.4
        second[1, 2] = 0.6
        first[0, 0] = 0.5
        second[0, 0] = 0.5

        result, stats = arbitrate_depth(
            tof,
            (first, second),
            absolute_tolerance=0.01,
            relative_tolerance=0.0,
            anchor_radius=1,
            np=np,
            cv2=cv2,
        )

        self.assertAlmostEqual(float(result[2, 2]), 0.5)
        self.assertAlmostEqual(float(result[2, 3]), 0.5055, places=4)
        self.assertEqual(float(result[1, 2]), 0.0)
        self.assertEqual(float(result[0, 0]), 0.0)
        self.assertEqual(stats["stereo_fill_pixels"], 1)
        self.assertEqual(stats["stereo_conflict_pixels"], 1)

    def test_identity_rectification_preserves_camera_depth(self) -> None:
        camera = ColmapCamera(1, "PINHOLE", 4, 3, (2.0, 2.0, 1.5, 1.0))
        image = ColmapImage(
            1,
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            1,
            "frame.jpg",
            frozenset(),
            (),
        )
        projection = np.asarray(
            [[2.0, 0.0, 1.5, 0.0], [0.0, 2.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        pair = RectifiedPair(
            left=image,
            right=image,
            left_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            right_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            left_valid=np.ones((3, 4), dtype=bool),
            right_valid=np.ones((3, 4), dtype=bool),
            model_left_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            model_right_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            model_left_valid=np.ones((3, 4), dtype=bool),
            model_right_valid=np.ones((3, 4), dtype=bool),
            left_rectification=np.identity(3),
            left_projection=projection,
            right_projection=projection,
            reprojection=np.identity(4),
            vertical=False,
        )
        depth = np.zeros((3, 4), dtype=np.float32)
        depth[1, 2] = 0.5
        view = DepthView(pair, depth, depth > 0, 0, None)

        projected = project_rectified_depth_to_camera(view, camera, 4, 3, np)

        self.assertAlmostEqual(float(projected[1, 2]), 0.5)
        self.assertEqual(int((projected > 0).sum()), 1)

    def test_depth_warp_maps_between_pinhole_intrinsics(self) -> None:
        source_intrinsic = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        target = ColmapCamera(1, "PINHOLE", 5, 5, (2.0, 2.0, 0.0, 0.0))
        depth = np.zeros((5, 5), dtype=np.uint16)
        depth[1, 1] = 500

        warped = warp_depth_to_camera(depth, source_intrinsic, target, np, cv2)

        self.assertEqual(int(warped[2, 2]), 500)

    def test_world_sphere_removes_background_depth(self) -> None:
        camera = ColmapCamera(1, "PINHOLE", 3, 3, (1.0, 1.0, 1.0, 1.0))
        image = ColmapImage(
            1,
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            1,
            "frame.jpg",
            frozenset(),
            (),
        )
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 0.5
        depth[1, 2] = 1.0

        masked = mask_depth_to_world_sphere(
            depth, image, camera, np.asarray([0.0, 0.0, 0.5]), 0.2, np
        )

        self.assertAlmostEqual(float(masked[1, 1]), 0.5)
        self.assertEqual(float(masked[1, 2]), 0.0)

    def test_screen_target_selects_seed_component(self) -> None:
        depth = np.zeros((9, 9), dtype=np.float32)
        depth[2:7, 3:6] = 0.5
        depth[7:, :] = 0.55

        masked, info = mask_depth_to_screen_target(
            depth,
            seed_x_fraction=0.5,
            seed_y_fraction=0.4,
            seed_fraction=0.2,
            radius_x_fraction=0.3,
            radius_y_fraction=0.4,
            depth_band=0.1,
            np=np,
            cv2=cv2,
        )

        self.assertGreater(info["target_pixels"], 0)
        self.assertAlmostEqual(float(masked[3, 4]), 0.5)
        self.assertEqual(float(masked[8, 0]), 0.0)

    def test_skin_pixels_can_override_background_seed_depth(self) -> None:
        depth = np.full((20, 20), 0.7, dtype=np.float32)
        depth[4:14, 7:13] = 0.4
        color = np.zeros((20, 20, 3), dtype=np.uint8)
        color[4:14, 7:13] = (120, 160, 210)

        masked, info = mask_depth_to_screen_target(
            depth,
            color_bgr=color,
            seed_x_fraction=0.5,
            seed_y_fraction=0.8,
            seed_fraction=0.1,
            radius_x_fraction=0.45,
            radius_y_fraction=0.45,
            depth_band=0.05,
            np=np,
            cv2=cv2,
        )

        self.assertGreater(info["skin_seed_pixels"], 0)
        self.assertAlmostEqual(info["seed_depth"], 0.4)
        self.assertAlmostEqual(float(masked[8, 10]), 0.4)

    def test_grabcut_foreground_is_limited_to_initial_rectangle(self) -> None:
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        image[8:34, 20:40] = (180, 180, 220)

        foreground = grabcut_foreground_mask(
            image,
            x_min_fraction=0.25,
            x_max_fraction=0.75,
            y_min_fraction=0.1,
            y_max_fraction=0.9,
            iterations=1,
            np=np,
            cv2=cv2,
        )

        self.assertTrue(bool(foreground[20, 30]))
        self.assertFalse(bool(foreground[20, 5]))


class RGBDHybridModelTests(unittest.TestCase):
    def test_metric_model_scales_translations_and_points(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source_model = root / "source"
            source_images = root / "images"
            destination = root / "metric"
            source_model.mkdir()
            source_images.mkdir()
            (source_model / "cameras.txt").write_text(
                "1 PINHOLE 4 3 2 2 1.5 1\n", encoding="utf-8"
            )
            (source_model / "images.txt").write_text(
                "1 1 0 0 0 1 2 3 1 frame.jpg\n1 1 7\n", encoding="utf-8"
            )
            (source_model / "points3D.txt").write_text(
                "7 4 5 6 1 2 3 0 1 0\n", encoding="utf-8"
            )
            (source_images / "frame.jpg").write_bytes(b"image")

            intrinsic = ((2.1, 0.0, 1.4), (0.0, 2.2, 0.9), (0.0, 0.0, 1.0))
            model = write_metric_colmap_model(
                source_model,
                source_images,
                destination,
                0.1,
                intrinsic=intrinsic,
                translations={1: (0.11, 0.22, 0.33)},
            )

            image_fields = (model / "images.txt").read_text().splitlines()[0].split()
            point_fields = (model / "points3D.txt").read_text().split()
            for actual, expected in zip(map(float, image_fields[5:8]), (0.11, 0.22, 0.33)):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(map(float, point_fields[1:4]), (0.4, 0.5, 0.6)):
                self.assertAlmostEqual(actual, expected)
            self.assertTrue((destination / "images" / "frame.jpg").is_file())
            camera_fields = (model / "cameras.txt").read_text().split()
            self.assertEqual(camera_fields[1], "PINHOLE")
            self.assertEqual(tuple(map(float, camera_fields[4:])), (2.1, 2.2, 1.4, 0.9))

    def test_parser_defaults_to_landscape_foundation_input(self) -> None:
        args = build_parser().parse_args(["input.mkv", "output"])
        validate_arguments(args)
        self.assertEqual((args.inference_width, args.inference_height), (960, 544))
        self.assertEqual((args.fusion_width, args.fusion_height), (960, 540))


if __name__ == "__main__":
    unittest.main()
