from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from recon_pipeline.backends.metashape import (
    MetashapeBackend,
    build_metashape_command,
)
from recon_pipeline.models import BackendName, PipelineConfig, Preset, Target
from recon_pipeline.process import CommandRunner
from recon_pipeline.workers.metashape_worker import MetashapeJob, run_job


class MetashapeCommandTests(unittest.TestCase):
    def test_metashape_executable_command_uses_dash_r_and_json(self) -> None:
        command = build_metashape_command(
            executable=Path(r"C:\Program Files\Agisoft\Metashape Pro\metashape.exe"),
            worker_script=Path(r"C:\pipeline\metashape_worker.py"),
            config_path=Path(r"C:\output\job.json"),
        )
        self.assertEqual(
            command,
            [
                r"C:\Program Files\Agisoft\Metashape Pro\metashape.exe",
                "-r",
                r"C:\pipeline\metashape_worker.py",
                "--config",
                r"C:\output\job.json",
            ],
        )

    def test_explicit_python_runner_does_not_use_dash_r(self) -> None:
        command = build_metashape_command(
            python_executable=Path(r"C:\Python311\python.exe"),
            worker_script=Path(r"C:\pipeline\metashape_worker.py"),
            config_path=Path(r"C:\output\job.json"),
        )
        self.assertEqual(command[0], r"C:\Python311\python.exe")
        self.assertNotIn("-r", command)
        self.assertEqual(command[-2:], ["--config", r"C:\output\job.json"])

    def test_command_rejects_ambiguous_runner(self) -> None:
        with self.assertRaisesRegex(ValueError, "Choose one Metashape runner"):
            build_metashape_command(
                executable=Path("metashape.exe"),
                python_executable=Path("python.exe"),
                config_path=Path("job.json"),
            )

    def test_backend_dry_run_records_command_without_writing_job(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"a")
            (images / "b.jpg").write_bytes(b"b")
            config = PipelineConfig(
                image_dir=images,
                output_dir=root / "output",
                backend=BackendName.METASHAPE,
                target=Target.MESH,
                preset=Preset.FAST,
                dry_run=True,
            ).normalized()
            runner = CommandRunner(root / "run.log", dry_run=True, echo=False)

            result = MetashapeBackend(root / "metashape.exe").run(config, runner)

            self.assertEqual(len(runner.records), 1)
            command = runner.records[0].argv
            self.assertEqual(command[0], str(root / "metashape.exe"))
            self.assertEqual(command[1], "-r")
            self.assertEqual(command[-2], "--config")
            self.assertFalse((root / "output" / "native" / "metashape" / "job.json").exists())
            self.assertEqual(result.backend, BackendName.METASHAPE)
            self.assertEqual(result.meshes, [root / "output" / "mesh" / "0" / "mesh.obj"])

    def test_backend_rejects_dense_target_before_launch(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            config = PipelineConfig(
                image_dir=root,
                output_dir=root / "output",
                backend=BackendName.METASHAPE,
                target=Target.DENSE,
                dry_run=True,
            )
            runner = CommandRunner(root / "run.log", dry_run=True, echo=False)
            with self.assertRaisesRegex(ValueError, "no standalone dense"):
                MetashapeBackend(root / "metashape.exe").run(config, runner)
            self.assertEqual(runner.records, [])

    def test_standard_install_has_explicit_professional_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Professional-edition"):
            MetashapeBackend._preflight_runner(
                executable=Path(
                    r"C:\Program Files\Agisoft\Metashape Standard\metashape.exe"
                ),
                python_executable=None,
                dry_run=True,
            )


class _FakeDocument:
    def __init__(self, chunk: _FakeChunk, events: list[tuple[str, object]]) -> None:
        self.chunk = chunk
        self.events = events
        self.path: Path | None = None

    def addChunk(self) -> _FakeChunk:
        self.events.append(("addChunk", None))
        return self.chunk

    def save(self, path: str | None = None) -> None:
        self.events.append(("save", path))
        if path is not None:
            self.path = Path(path)
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("fake project\n", encoding="utf-8")


class _FakeChunk:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events
        self.cameras = [
            SimpleNamespace(transform=object()),
            SimpleNamespace(transform=object()),
        ]

    def addPhotos(self, photos: list[str]) -> None:
        self.events.append(("addPhotos", photos))

    def matchPhotos(self, **kwargs: object) -> None:
        self.events.append(("matchPhotos", kwargs))

    def alignCameras(self) -> None:
        self.events.append(("alignCameras", None))

    def exportCameras(self, **kwargs: object) -> None:
        self.events.append(("exportCameras", kwargs))
        anchor = Path(str(kwargs["path"]))
        self.assert_anchor_name(anchor)
        destination = anchor.parent / "sparse" / "0"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "cameras.txt").write_text(
            "1 PINHOLE 10 10 1 1 5 5\n", encoding="utf-8"
        )
        (destination / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 a.jpg\n\n"
            "2 1 0 0 0 0 0 0 1 b.jpg\n\n",
            encoding="utf-8",
        )
        (destination / "points3D.txt").write_text(
            "# no points in fake model\n", encoding="utf-8"
        )
        if kwargs["save_images"]:
            images = anchor.parent / "images"
            images.mkdir(parents=True, exist_ok=True)
            (images / "a.jpg").write_bytes(b"a")
            (images / "b.jpg").write_bytes(b"b")

    @staticmethod
    def assert_anchor_name(anchor: Path) -> None:
        if anchor.name != "colmap.txt":
            raise AssertionError(f"unexpected COLMAP anchor: {anchor}")

    def buildDepthMaps(self, **kwargs: object) -> None:
        self.events.append(("buildDepthMaps", kwargs))

    def buildModel(self, **kwargs: object) -> None:
        self.events.append(("buildModel", kwargs))

    def exportModel(self, **kwargs: object) -> None:
        self.events.append(("exportModel", kwargs))
        path = Path(str(kwargs["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake obj\n", encoding="utf-8")


def _fake_api(events: list[tuple[str, object]]) -> SimpleNamespace:
    chunk = _FakeChunk(events)
    return SimpleNamespace(
        app=SimpleNamespace(activated=True, version="2.3.1"),
        CamerasFormatColmap=object(),
        MildFiltering=object(),
        DepthMapsData=object(),
        Arbitrary=object(),
        EnabledInterpolation=object(),
        HighFaceCount=object(),
        ModelFormatOBJ=object(),
        Document=lambda: _FakeDocument(chunk, events),
    )


class MetashapeWorkerTests(unittest.TestCase):
    def _job(self, root: Path, target: str) -> MetashapeJob:
        image_dir = root / "input"
        image_dir.mkdir()
        (image_dir / "a.jpg").write_bytes(b"a")
        (image_dir / "b.jpg").write_bytes(b"b")
        return MetashapeJob(
            image_dir=image_dir,
            target=target,
            preset="normal",
            texture=False,
            copy_images=True,
            project_path=root / "native" / "metashape" / "project.psx",
            mesh_path=root / "mesh" / "0" / "mesh.obj",
            colmap_model_path=root / "colmap" / "sparse" / "0",
            colmap_images_path=root / "colmap" / "images",
            result_path=root / "native" / "metashape" / "result.json",
        )

    def test_mesh_is_built_directly_from_depth_maps(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            events: list[tuple[str, object]] = []
            api = _fake_api(events)
            job = self._job(root, "mesh")

            result = run_job(job, metashape=api)

            operation_names = [name for name, _ in events]
            self.assertLess(
                operation_names.index("buildDepthMaps"),
                operation_names.index("buildModel"),
            )
            build_model = next(value for name, value in events if name == "buildModel")
            self.assertIs(build_model["source_data"], api.DepthMapsData)
            export_cameras = next(
                value for name, value in events if name == "exportCameras"
            )
            self.assertIs(export_cameras["format"], api.CamerasFormatColmap)
            self.assertFalse(export_cameras["binary"])
            self.assertTrue(export_cameras["save_images"])
            self.assertTrue(export_cameras["convert_to_pinhole"])
            self.assertEqual(
                Path(export_cameras["path"]), root / "colmap" / "colmap.txt"
            )
            self.assertEqual(result["aligned_cameras"], 2)
            self.assertTrue(job.project_path.is_file())
            self.assertTrue(job.mesh_path.is_file())
            self.assertTrue((job.colmap_model_path / "cameras.txt").is_file())
            self.assertTrue((job.colmap_images_path / "a.jpg").is_file())

    def test_sfm_stops_before_depth_and_mesh_operations(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            events: list[tuple[str, object]] = []
            job = self._job(Path(temp), "sfm")
            run_job(job, metashape=_fake_api(events))
            operation_names = [name for name, _ in events]
            self.assertNotIn("buildDepthMaps", operation_names)
            self.assertNotIn("buildModel", operation_names)
            self.assertNotIn("exportModel", operation_names)

    def test_unactivated_api_has_actionable_license_error(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            job = self._job(Path(temp), "sfm")
            api = SimpleNamespace(
                app=SimpleNamespace(activated=False, version="2.3.1"),
                CamerasFormatColmap=object(),
                Document=object(),
            )
            with self.assertRaisesRegex(
                RuntimeError, "Professional is not activated.*Standard"
            ):
                run_job(job, metashape=api)

    def test_dense_job_is_rejected_during_json_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "intentionally unsupported"):
            MetashapeJob.from_mapping(
                {
                    "schema_version": 1,
                    "image_dir": "images",
                    "target": "dense",
                    "preset": "normal",
                    "project_path": "project.psx",
                    "mesh_path": "mesh.obj",
                    "colmap_model_path": "sparse/0",
                    "colmap_images_path": "images-out",
                    "result_path": "result.json",
                }
            )


if __name__ == "__main__":
    unittest.main()
