from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from recon_pipeline.rgbd_fusion import (
    PoseRecord,
    _rotation_to_quaternion,
    derive_color_calibration,
    select_texture_poses,
)


@unittest.skipIf(np is None, "RGB-D optional dependencies are not installed")
class RGBDFusionTests(unittest.TestCase):
    def test_k4a_1080p_mode_specific_intrinsics(self) -> None:
        payload = {
            "CalibrationInformation": {
                "Cameras": [
                    {
                        "Purpose": "CALIBRATION_CameraPurposePhotoVideo",
                        "Intrinsics": {
                            "ModelType": "CALIBRATION_LensDistortionModelBrownConrady",
                            "ModelParameters": [
                                0.49425735473632815,
                                0.49793917338053384,
                                0.5848361253738403,
                                0.7792156934738159,
                                0.0873513743,
                                -0.1150452122,
                                0.0480436012,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0002995548,
                                -0.0002088123,
                            ],
                        },
                    }
                ]
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = derive_color_calibration(path, 1920, 1080)
        intrinsic = np.asarray(calibration.intrinsic)
        self.assertAlmostEqual(intrinsic[0, 0], 1122.88536, places=4)
        self.assertAlmostEqual(intrinsic[1, 1], 1122.07060, places=4)
        self.assertAlmostEqual(intrinsic[0, 2], 948.47412, places=4)
        self.assertAlmostEqual(intrinsic[1, 2], 536.53241, places=4)

    def test_identity_rotation_becomes_colmap_identity_quaternion(self) -> None:
        self.assertEqual(
            _rotation_to_quaternion(np.eye(3), np),
            (1.0, 0.0, 0.0, 0.0),
        )

    def test_texture_views_are_limited_and_directionally_distributed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_root = Path(temp)
            poses = []
            for index, x in enumerate((-0.3, -0.1, 0.1, 0.3)):
                image_name = f"frame_{index}.png"
                image = np.zeros((64, 64), dtype=np.uint8)
                image[:, :: index + 2] = 255
                self.assertTrue(cv2.imwrite(str(image_root / image_name), image))
                transform = np.eye(4)
                transform[:3, 3] = (x, 0.0, -1.0)
                poses.append(
                    PoseRecord(
                        source_frame=index,
                        timestamp_seconds=float(index),
                        camera_to_world=transform.tolist(),
                        fitness=1.0,
                        inlier_rmse=0.0,
                        texture_image=image_name,
                    )
                )

            selected = select_texture_poses(
                poses, image_root, (0.0, 0.0, 0.0), 2, 0.5, np, cv2
            )

        self.assertEqual(len(selected), 2)
        self.assertEqual(len({pose.source_frame for pose in selected}), 2)


if __name__ == "__main__":
    unittest.main()
