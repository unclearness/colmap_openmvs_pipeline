from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recon_pipeline.artifacts import (
    copy_colmap_text_model,
    validate_mesh,
    validate_colmap_text_model,
)


class ColmapArtifactTests(unittest.TestCase):
    def test_model_with_empty_observation_line_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            model.mkdir()
            (model / "cameras.txt").write_text(
                "1 PINHOLE 16 16 10 10 8 8\n", encoding="utf-8"
            )
            (model / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 image.jpg\n\n", encoding="utf-8"
            )
            (model / "points3D.txt").write_text("", encoding="utf-8")
            stats = validate_colmap_text_model(model)
            self.assertEqual(stats["images"], 1)

    def test_missing_points_can_be_normalized_to_valid_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            destination = Path(temp) / "destination"
            source.mkdir()
            (source / "cameras.txt").write_text(
                "1 PINHOLE 16 16 10 10 8 8\n", encoding="utf-8"
            )
            (source / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 image.jpg\n\n", encoding="utf-8"
            )
            details = copy_colmap_text_model(
                source, destination, allow_missing_points=True
            )
            self.assertTrue(details["generated_empty_points3D"])
            self.assertTrue((destination / "points3D.txt").is_file())

    def test_empty_ply_mesh_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / "mesh.ply"
            mesh.write_text(
                "ply\nformat ascii 1.0\nelement vertex 0\n"
                "element face 0\nend_header\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(RuntimeError, "no surface geometry"):
                validate_mesh(mesh)

    def test_obj_mesh_counts_vertices_and_faces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / "mesh.obj"
            mesh.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii"
            )
            stats = validate_mesh(mesh)
            self.assertEqual(stats["vertices"], 3)
            self.assertEqual(stats["faces"], 1)


if __name__ == "__main__":
    unittest.main()
