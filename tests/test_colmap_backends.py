from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recon_pipeline.backends.colmap import ColmapBackend
from recon_pipeline.backends.openmvs import OpenMVSBackend
from recon_pipeline.models import BackendName, Mesher, PipelineConfig, Preset, Target
from recon_pipeline.process import CommandRunner


class ColmapBackendCommandTests(unittest.TestCase):
    def _config(
        self, root: Path, *, backend: BackendName, target: Target
    ) -> PipelineConfig:
        image_dir = root / "images"
        image_dir.mkdir()
        (image_dir / "a.jpg").write_bytes(b"a")
        (image_dir / "b.jpg").write_bytes(b"b")
        return PipelineConfig(
            image_dir=image_dir,
            output_dir=root / "output",
            backend=backend,
            target=target,
            preset=Preset.FAST,
            dry_run=True,
        ).normalized()

    def test_colmap_411_matcher_name_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root, backend=BackendName.COLMAP, target=Target.SFM)
            runner = CommandRunner(root / "pipeline.log", dry_run=True, echo=False)
            ColmapBackend(root / "colmap.exe").run(config, runner)
            matching = next(
                record for record in runner.records if record.label == "colmap.sequential_matcher"
            )
            index = matching.argv.index("--FeatureMatching.type")
            self.assertEqual(matching.argv[index + 1], "SIFT_BRUTEFORCE")

    def test_colmap_fusion_stays_in_workspace_for_delaunay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root, backend=BackendName.COLMAP, target=Target.MESH)
            runner = CommandRunner(root / "pipeline.log", dry_run=True, echo=False)
            ColmapBackend(root / "colmap.exe").run(config, runner)
            fusion = next(
                record for record in runner.records if record.label == "colmap.stereo_fusion.0"
            )
            output_index = fusion.argv.index("--output_path")
            self.assertTrue(
                fusion.argv[output_index + 1].endswith(
                    str(Path("native") / "colmap" / "dense" / "0" / "fused.ply")
                )
            )

    def test_auto_mesher_falls_back_when_delaunay_is_not_built(self) -> None:
        self.assertEqual(
            ColmapBackend.select_mesher(Mesher.AUTO, delaunay_available=False),
            Mesher.POISSON,
        )
        with self.assertRaisesRegex(RuntimeError, "without CGAL/Delaunay"):
            ColmapBackend.select_mesher(
                Mesher.DELAUNAY, delaunay_available=False
            )

    def test_colmap_texture_target_returns_textured_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root, backend=BackendName.COLMAP, target=Target.MESH)
            config.texture = True
            runner = CommandRunner(root / "pipeline.log", dry_run=True, echo=False)
            result = ColmapBackend(root / "colmap.exe").run(config, runner)
            texturer = next(
                record
                for record in runner.records
                if record.label == "colmap.mesh_texturer.0"
            )
            self.assertIn("--workspace_path", texturer.argv)
            self.assertEqual(
                result.meshes,
                [root / "output" / "mesh" / "0" / "textured" / "mesh.obj"],
            )

    def test_openmvs_mesh_uses_external_dense_point_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(
                root, backend=BackendName.OPENMVS, target=Target.MESH
            )
            runner = CommandRunner(root / "pipeline.log", dry_run=True, echo=False)
            OpenMVSBackend(
                root / "colmap.exe", root / "openmvs", variant="cuda"
            ).run(config, runner)
            reconstruct = next(
                record
                for record in runner.records
                if record.label == "openmvs.reconstruct_mesh.0"
            )
            pointcloud_index = reconstruct.argv.index("-p")
            self.assertTrue(
                reconstruct.argv[pointcloud_index + 1].endswith("scene_dense.ply")
            )
            output_index = reconstruct.argv.index("-o")
            self.assertTrue(
                reconstruct.argv[output_index + 1].endswith("scene_mesh.ply")
            )

    def test_openmvs_textured_mesh_is_standard_obj(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(
                root, backend=BackendName.OPENMVS, target=Target.MESH
            )
            config.texture = True
            runner = CommandRunner(root / "pipeline.log", dry_run=True, echo=False)
            result = OpenMVSBackend(
                root / "colmap.exe", root / "openmvs", variant="cuda"
            ).run(config, runner)
            texturer = next(
                record
                for record in runner.records
                if record.label == "openmvs.texture_mesh.0"
            )
            export_index = texturer.argv.index("--export-type")
            self.assertEqual(texturer.argv[export_index + 1], "obj")
            self.assertEqual(
                result.meshes,
                [root / "output" / "mesh" / "0" / "mesh.obj"],
            )


if __name__ == "__main__":
    unittest.main()
