from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from recon_pipeline.models import BackendName, PipelineConfig, Target
from recon_pipeline.process import CommandRunner
from recon_pipeline.backends.realityscan import (
    RealityScanBackend,
    build_command,
    flatten_images,
    plan_flattened_images,
    resolve_realityscan_executable,
)


ASSETS = (
    Path(__file__).resolve().parents[1]
    / "recon_pipeline"
    / "assets"
    / "realityscan"
)


def _position(argv: list[str], option: str) -> int:
    return argv.index(option)


class ProducingRunner:
    dry_run = False

    def __init__(self, *, omit_second_image: bool = False) -> None:
        self.omit_second_image = omit_second_image
        self.argv: list[str] | None = None

    def run(self, label: str, argv: list[str], *, cwd: Path | None = None) -> None:
        self.argv = list(argv)
        sparse = Path(argv[_position(argv, "-exportSparsePointCloud") + 1])
        registration = Path(argv[_position(argv, "-exportRegistration") + 1])
        project = Path(argv[_position(argv, "-save") + 1])
        sparse.write_text("ply\n", encoding="utf-8")
        project.write_text("RealityScan project\n", encoding="utf-8")

        export_dir = registration.parent
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "cameras.txt").write_text(
            "# cameras\n1 PINHOLE 16 16 10 10 8 8\n",
            encoding="utf-8",
        )
        (export_dir / "images.txt").write_text(
            "# images\n"
            "1 1 0 0 0 0 0 0 1 a.jpg\n\n"
            "2 1 0 0 0 1 0 0 1 b.jpg\n\n",
            encoding="utf-8",
        )
        (export_dir / "a.jpg").write_bytes(b"a")
        if not self.omit_second_image:
            (export_dir / "b.jpg").write_bytes(b"b")

        if "-exportSelectedModel" in argv:
            mesh = Path(argv[_position(argv, "-exportSelectedModel") + 1])
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_text("o mesh\n", encoding="utf-8")


class RealityScanDiscoveryTests(unittest.TestCase):
    def test_explicit_executable_has_priority(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            explicit = root / "explicit" / "RealityScan.exe"
            fallback = root / "fallback" / "RealityScan.exe"
            explicit.parent.mkdir()
            fallback.parent.mkdir()
            explicit.write_bytes(b"exe")
            fallback.write_bytes(b"exe")

            resolved = resolve_realityscan_executable(
                explicit,
                environ={"REALITYSCAN_EXE": str(fallback)},
                known_paths=[fallback],
            )
            self.assertEqual(resolved, explicit.resolve())

    def test_environment_directory_is_supported(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            install = Path(temporary) / "RealityScan_2.1"
            install.mkdir()
            executable = install / "RealityScan.exe"
            executable.write_bytes(b"exe")
            resolved = resolve_realityscan_executable(
                environ={"REALITYSCAN_HOME": str(install)}, known_paths=[]
            )
            self.assertEqual(resolved, executable.resolve())

    def test_invalid_explicit_path_does_not_silently_fall_back(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            fallback = root / "RealityScan.exe"
            fallback.write_bytes(b"exe")
            with self.assertRaises(FileNotFoundError):
                resolve_realityscan_executable(
                    root / "missing.exe", known_paths=[fallback]
                )


class RealityScanFlattenTests(unittest.TestCase):
    def test_nested_paths_are_flattened_deterministically(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "left").mkdir(parents=True)
            (source / "right").mkdir()
            (source / "left" / "image.jpg").write_bytes(b"left")
            (source / "right" / "image.jpg").write_bytes(b"right")

            plan = plan_flattened_images(source)
            self.assertEqual(
                [name for _, name in plan],
                ["left__image.jpg", "right__image.jpg"],
            )
            copied = flatten_images(source, root / "flat")
            self.assertEqual([path.name for path in copied], [name for _, name in plan])

    def test_generated_name_collision_is_rejected_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            source = Path(temporary)
            (source / "a").mkdir()
            (source / "a" / "photo.JPG").write_bytes(b"nested")
            (source / "A__PHOTO.jpg").write_bytes(b"flat")
            with self.assertRaisesRegex(ValueError, "collision"):
                plan_flattened_images(source)


class RealityScanCommandTests(unittest.TestCase):
    def test_sfm_command_has_safety_flags_and_colmap_exports(self) -> None:
        argv = build_command(
            Path("RealityScan.exe"),
            Path("flat-images"),
            Path("output"),
            target=Target.SFM,
            assets_dir=ASSETS,
        )
        self.assertEqual(argv[0], "RealityScan.exe")
        self.assertEqual(
            argv[1:7],
            [
                "-stdConsole",
                "-headless",
                "-silent",
                str(Path("output/native/realityscan/crash_reports")),
                "-set",
                "appQuitOnError=true",
            ],
        )
        self.assertLess(_position(argv, "-align"), _position(argv, "-exportRegistration"))
        self.assertLess(_position(argv, "-exportRegistration"), _position(argv, "-save"))
        self.assertNotIn("-calculateNormalModel", argv)
        self.assertNotIn("-exportSelectedModel", argv)
        self.assertEqual(argv[-1], "-quit")

    def test_high_mesh_with_texture_uses_official_cli_sequence(self) -> None:
        argv = build_command(
            Path("RealityScan.exe"),
            Path("flat-images"),
            Path("output"),
            target="mesh",
            quality="high",
            texture=True,
            no_distortion=True,
            assets_dir=ASSETS,
        )
        expected = [
            "-editInputSelection",
            "-align",
            "-setReconstructionRegionAuto",
            "-calculateHighModel",
            "-unwrap",
            "-calculateTexture",
            "-exportSelectedModel",
            "-save",
            "-quit",
        ]
        positions = [_position(argv, option) for option in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("inpDistortionModel=0", argv)
        self.assertNotIn("-calculateNormalModel", argv)

    def test_shared_intrinsics_groups_calibration_and_lens_before_alignment(self) -> None:
        argv = build_command(
            Path("RealityScan.exe"),
            Path("flat-images"),
            Path("output"),
            target=Target.SFM,
            shared_intrinsics=True,
            assets_dir=ASSETS,
        )
        self.assertLess(
            _position(argv, "-setConstantCalibrationGroups"),
            _position(argv, "-align"),
        )
        lens_position = _position(argv, "-setPriorLensGroup")
        self.assertEqual(argv[lens_position + 1], "0")
        self.assertLess(lens_position, _position(argv, "-align"))

    def test_brown3_unknown_distortion_is_explicit_before_alignment(self) -> None:
        argv = build_command(
            Path("RealityScan.exe"),
            Path("flat-images"),
            Path("output"),
            target=Target.SFM,
            distortion_model="brown3",
            distortion_prior="unknown",
            assets_dir=ASSETS,
        )
        alignment = _position(argv, "-align")
        self.assertIn("sfmDistortionModel=Brown3", argv[:alignment])
        self.assertIn("inpDistortionModel=2", argv[:alignment])
        self.assertIn("inpDistortion=0", argv[:alignment])

    def test_sensitive_alignment_settings_precede_alignment(self) -> None:
        argv = build_command(
            Path("RealityScan.exe"),
            Path("flat-images"),
            Path("output"),
            target=Target.SFM,
            sensitive_alignment=True,
            assets_dir=ASSETS,
        )
        alignment = _position(argv, "-align")
        for value in (
            "sfmMaxFeaturesPerMpx=20000",
            "sfmMaxFeaturesPerImage=80000",
            "sfmImagesOverlap=High",
            "sfmDetectorSensitivity=Ultra",
            "sfmPreselectorFeatures=20000",
            "sfmForceComponentRematch=true",
        ):
            self.assertIn(value, argv)
            self.assertLess(argv.index(value), alignment)

    def test_dense_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no standalone dense"):
            build_command(
                Path("RealityScan.exe"),
                Path("flat-images"),
                Path("output"),
                target=Target.DENSE,
                assets_dir=ASSETS,
            )


class RealityScanBackendTests(unittest.TestCase):
    def _input(self, root: Path) -> Path:
        image_dir = root / "input"
        image_dir.mkdir()
        (image_dir / "a.jpg").write_bytes(b"a")
        (image_dir / "b.jpg").write_bytes(b"b")
        return image_dir

    def test_dry_run_records_command_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            image_dir = self._input(root)
            output_dir = root / "output"
            runner = CommandRunner(root / "run.log", dry_run=True, echo=False)
            config = PipelineConfig(
                image_dir=image_dir,
                output_dir=output_dir,
                backend=BackendName.REALITYSCAN,
                target=Target.MESH,
                dry_run=True,
                realityscan_exe=root / "not-installed" / "RealityScan.exe",
            )

            result = RealityScanBackend(assets_dir=ASSETS).run(config, runner)

            self.assertEqual(len(runner.records), 1)
            self.assertTrue(runner.records[0].dry_run)
            self.assertFalse(output_dir.exists())
            self.assertEqual(result.meshes, [output_dir / "mesh" / "0" / "mesh.obj"])
            self.assertEqual(result.dense_clouds, [])

    def test_run_normalizes_and_validates_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            image_dir = self._input(root)
            output_dir = root / "output"
            executable = root / "RealityScan.exe"
            executable.write_bytes(b"exe")
            runner = ProducingRunner()
            config = PipelineConfig(
                image_dir=image_dir,
                output_dir=output_dir,
                backend=BackendName.REALITYSCAN,
                target=Target.MESH,
                realityscan_exe=executable,
            )

            result = RealityScanBackend(assets_dir=ASSETS).run(config, runner)  # type: ignore[arg-type]

            model = output_dir / "colmap" / "sparse" / "0"
            self.assertEqual(result.colmap_models, [model])
            self.assertTrue((model / "points3D.txt").is_file())
            self.assertTrue((output_dir / "colmap" / "images" / "a.jpg").is_file())
            self.assertEqual(result.metadata["registered_images"], 2)
            self.assertEqual(result.dense_clouds, [])
            self.assertTrue(result.meshes[0].is_file())

    def test_missing_registered_image_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            image_dir = self._input(root)
            executable = root / "RealityScan.exe"
            executable.write_bytes(b"exe")
            config = PipelineConfig(
                image_dir=image_dir,
                output_dir=root / "output",
                backend=BackendName.REALITYSCAN,
                target=Target.SFM,
                realityscan_exe=executable,
            )
            with self.assertRaisesRegex(FileNotFoundError, "registered COLMAP"):
                RealityScanBackend(assets_dir=ASSETS).run(
                    config, ProducingRunner(omit_second_image=True)  # type: ignore[arg-type]
                )


class RealityScanAssetTests(unittest.TestCase):
    def test_reference_xml_files_are_byte_exact(self) -> None:
        expected = {
            "colmap_undistorted.xml": "d31f5b35e2708891ea3fad2c585708b2014a7bbb4fbb3888c15606bc05ded530",
            "sparse_point_cloud.xml": "cdbdf0e0c335924eb39bd086a3c518d03a110bddafdfe07864825bd8bd0d089a",
            "export_obj.xml": "adddfac58d51612a67a4f93376cc40998a82388f07ceede6601e85e5afbdf641",
        }
        for name, digest in expected.items():
            with self.subTest(name=name):
                actual = hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
                self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
