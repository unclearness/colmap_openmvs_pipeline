from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from recon_pipeline.foundation_stereo import (
    ColmapCamera,
    ColmapImage,
    DepthView,
    RectifiedPair,
    StereoPair,
    _select_diverse_source_candidates,
    build_parser,
    camera_matrix_and_distortion,
    estimate_world_normals_from_depth,
    multiview_consistency_mask,
    quaternion_to_rotation,
    read_colmap_images,
    spatially_uniform_reference_order,
    validate_arguments,
)
from recon_pipeline.foundation_stereo_icp_refusion import (
    build_parser as build_icp_refusion_parser,
    clamp_pose_correction,
    select_icp_edges,
    validate_arguments as validate_icp_refusion_arguments,
)


class FoundationStereoTextModelTests(unittest.TestCase):
    def test_images_parser_collects_only_valid_point_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "images.txt"
            path.write_text(
                "# images\n"
                "7 1 0 0 0 1 2 3 4 image 7.jpg\n"
                "10 20 100 30 40 -1 50 60 101\n",
                encoding="utf-8",
            )
            image = read_colmap_images(path)[7]
        self.assertEqual(image.name, "image 7.jpg")
        self.assertEqual(image.point3d_ids, frozenset({100, 101}))

    def test_invalid_voxel_size_is_rejected_before_inference(self) -> None:
        args = build_parser().parse_args(
            ["model", "images", "output", "--voxel-size", "-1"]
        )
        with self.assertRaisesRegex(ValueError, "voxel-size"):
            validate_arguments(args)

    def test_depth_only_flag_is_parsed(self) -> None:
        args = build_parser().parse_args(
            ["model", "images", "output", "--depth-only"]
        )
        self.assertTrue(args.depth_only)

    def test_icp_refusion_defaults_are_slightly_relaxed(self) -> None:
        args = build_icp_refusion_parser().parse_args(
            [
                "source",
                "output",
                "--optimizer",
                "leave-one-out",
                "--loo-tukey-k",
                "0.075",
            ]
        )
        validate_icp_refusion_arguments(args)
        self.assertEqual(args.optimizer, "leave-one-out")
        self.assertEqual(args.loo_tukey_k, 0.075)
        self.assertEqual(args.min_consistent_views, 4)
        self.assertEqual(args.consistency_relative_tolerance, 0.005)
        self.assertEqual(args.consistency_absolute_voxel_multiplier, 0.5)


@unittest.skipIf(np is None, "FoundationStereo optional dependencies are not installed")
class FoundationStereoGeometryTests(unittest.TestCase):
    @staticmethod
    def _image(image_id: int, name: str) -> ColmapImage:
        return ColmapImage(
            image_id=image_id,
            quaternion=(1.0, 0.0, 0.0, 0.0),
            translation=(0.0, 0.0, 0.0),
            camera_id=1,
            name=name,
            point3d_ids=frozenset(range(image_id + 1)),
            observations=(),
        )

    def test_identity_quaternion_is_identity_rotation(self) -> None:
        np.testing.assert_allclose(
            quaternion_to_rotation((1.0, 0.0, 0.0, 0.0), np), np.eye(3)
        )

    def test_simple_radial_maps_to_opencv_coefficients(self) -> None:
        intrinsic, distortion = camera_matrix_and_distortion(
            ColmapCamera(1, "SIMPLE_RADIAL", 100, 200, (80.0, 50.0, 100.0, 0.1)),
            np,
        )
        np.testing.assert_allclose(
            intrinsic, [[80.0, 0.0, 50.0], [0.0, 80.0, 100.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(distortion, [0.1, 0.0, 0.0, 0.0])

    def test_multiview_consistency_keeps_matching_depth(self) -> None:
        import cv2

        image = ColmapImage(
            image_id=1,
            quaternion=(1.0, 0.0, 0.0, 0.0),
            translation=(0.0, 0.0, 0.0),
            camera_id=1,
            name="image.png",
            point3d_ids=frozenset(),
            observations=(),
        )
        projection = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        rectified = RectifiedPair(
            left=image,
            right=image,
            left_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            right_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            left_valid=np.ones((2, 2), dtype=bool),
            right_valid=np.ones((2, 2), dtype=bool),
            model_left_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            model_right_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            model_left_valid=np.ones((2, 2), dtype=bool),
            model_right_valid=np.ones((2, 2), dtype=bool),
            left_rectification=np.eye(3),
            left_projection=projection,
            right_projection=projection,
            reprojection=np.eye(4),
            vertical=False,
        )
        reference = DepthView(
            rectified, np.full((2, 2), 2.0, np.float32), np.ones((2, 2), bool), 0, None
        )
        matching = DepthView(
            rectified, np.full((2, 2), 2.0, np.float32), np.ones((2, 2), bool), 1, None
        )
        mismatching = DepthView(
            rectified, np.full((2, 2), 3.0, np.float32), np.ones((2, 2), bool), 2, None
        )

        accepted = multiview_consistency_mask(
            reference,
            [reference, matching],
            min_support_views=1,
            relative_tolerance=0.01,
            absolute_tolerance=0.01,
            np=np,
            cv2=cv2,
        )
        rejected = multiview_consistency_mask(
            reference,
            [reference, mismatching],
            min_support_views=1,
            relative_tolerance=0.01,
            absolute_tolerance=0.01,
            np=np,
            cv2=cv2,
        )

        self.assertTrue(accepted.all())
        self.assertFalse(rejected.any())

        reference.normal_world = np.zeros((2, 2, 3), np.float32)
        reference.normal_world[..., 2] = -1.0
        matching.normal_world = reference.normal_world.copy()
        mismatching.normal_world = np.zeros((2, 2, 3), np.float32)
        mismatching.normal_world[..., 0] = 1.0
        normal_accepted = multiview_consistency_mask(
            reference,
            [reference, matching],
            min_support_views=1,
            relative_tolerance=0.01,
            absolute_tolerance=0.01,
            normal_cosine_threshold=0.9,
            np=np,
            cv2=cv2,
        )
        normal_rejected = multiview_consistency_mask(
            reference,
            [reference, mismatching],
            min_support_views=1,
            relative_tolerance=0.01,
            absolute_tolerance=0.01,
            normal_cosine_threshold=0.9,
            np=np,
            cv2=cv2,
        )
        self.assertTrue(normal_accepted.all())
        self.assertFalse(normal_rejected.any())

    def test_depth_plane_normal_is_oriented_toward_camera(self) -> None:
        image = self._image(1, "plane.png")
        projection = np.asarray(
            [[10.0, 0.0, 2.0, 0.0], [0.0, 10.0, 2.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        rectified = RectifiedPair(
            left=image,
            right=image,
            left_bgr=np.zeros((5, 5, 3), dtype=np.uint8),
            right_bgr=np.zeros((5, 5, 3), dtype=np.uint8),
            left_valid=np.ones((5, 5), dtype=bool),
            right_valid=np.ones((5, 5), dtype=bool),
            model_left_bgr=np.zeros((5, 5, 3), dtype=np.uint8),
            model_right_bgr=np.zeros((5, 5, 3), dtype=np.uint8),
            model_left_valid=np.ones((5, 5), dtype=bool),
            model_right_valid=np.ones((5, 5), dtype=bool),
            left_rectification=np.eye(3),
            left_projection=projection,
            right_projection=projection,
            reprojection=np.eye(4),
            vertical=False,
        )
        view = DepthView(
            rectified,
            np.full((5, 5), 2.0, np.float32),
            np.ones((5, 5), bool),
            0,
            None,
        )

        normals = estimate_world_normals_from_depth(
            view,
            relative_discontinuity=0.01,
            absolute_discontinuity=0.01,
            np=np,
        )

        np.testing.assert_allclose(normals[2, 2], [0.0, 0.0, -1.0], atol=1e-6)

    def test_spatial_reference_order_visits_every_bin_before_second_round(self) -> None:
        images = [self._image(index, f"image_{index}.jpg") for index in range(8)]
        angles = {
            image.image_id: (image.image_id // 2) * (np.pi / 2) + 0.1
            for image in images
        }

        selected = spatially_uniform_reference_order(
            images, angles, bin_count=4, references_per_bin=2
        )

        first_round_bins = [int(angles[image.image_id] // (np.pi / 2)) for image in selected[:4]]
        self.assertEqual(first_round_bins, [0, 1, 2, 3])

    def test_two_sources_bracket_reference_azimuth_when_available(self) -> None:
        reference = self._image(10, "reference.jpg")
        negative_best = StereoPair(reference, self._image(11, "negative.jpg"), 50, 1.0, 10.0, 0.1, 2.0)
        negative_second = StereoPair(reference, self._image(12, "negative2.jpg"), 40, 1.0, 10.0, 0.1, 2.0)
        positive = StereoPair(reference, self._image(13, "positive.jpg"), 30, 1.0, 10.0, 0.1, 2.0)

        selected = _select_diverse_source_candidates(
            [
                (10.0, -0.1, negative_best),
                (9.0, -0.2, negative_second),
                (8.0, 0.1, positive),
            ],
            2,
        )

        self.assertEqual({candidate[2] for candidate in selected}, {negative_best, positive})

    def test_icp_edges_connect_nearby_similar_views(self) -> None:
        poses = []
        for x in (0.0, 1.0, 2.0):
            pose = np.identity(4)
            pose[0, 3] = x
            poses.append(pose)

        edges = select_icp_edges(
            poses, neighbors=1, max_view_angle_degrees=10.0, np=np
        )

        pairs = {(source, target) for source, target, _ in edges}
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 2), pairs)

    def test_pose_correction_is_clamped(self) -> None:
        import cv2

        initial = np.identity(4)
        optimized = np.identity(4)
        optimized[0, 3] = 2.0
        angle = np.radians(20.0)
        optimized[:3, :3] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        result, translation, rotation, clamped = clamp_pose_correction(
            initial,
            optimized,
            max_translation=1.0,
            max_rotation_degrees=5.0,
            np=np,
            cv2=cv2,
        )

        self.assertTrue(clamped)
        self.assertAlmostEqual(translation, 2.0)
        self.assertAlmostEqual(rotation, 20.0)
        self.assertAlmostEqual(float(np.linalg.norm(result[:3, 3])), 1.0)


if __name__ == "__main__":
    unittest.main()
