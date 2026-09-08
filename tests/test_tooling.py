from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from recon_pipeline.tooling import (
    ArchiveSafetyError,
    ManifestError,
    ToolResolver,
    find_7z,
    install_asset,
    load_manifest,
    safe_extract_zip,
    verify_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tools" / "manifest.json"
OPENMVS_COMMANDS = (
    "InterfaceCOLMAP.exe",
    "DensifyPointCloud.exe",
    "ReconstructMesh.exe",
    "RefineMesh.exe",
    "TextureMesh.exe",
)


def touch_file(path: Path, contents: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    """Create test scratch space inside the sandbox-writable repository."""

    return tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)


class ManifestTests(unittest.TestCase):
    def test_manifest_pins_official_release_hashes(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)

        colmap = manifest["tools"]["colmap"]
        self.assertEqual(colmap["version"], "4.1.1")
        self.assertEqual(
            colmap["variants"]["cuda"]["sha256"],
            "b06064e7e4bd34f5b4ef71b442d3537d95d57c666dbec5a3b475902ccd832b9b",
        )
        self.assertEqual(
            colmap["variants"]["nocuda"]["sha256"],
            "faf1247d2ec90933aa8bd003709790abf0211cdc132cceec4c831718f2e0895a",
        )
        openmvs = manifest["tools"]["openmvs"]
        self.assertEqual(openmvs["version"], "2.4.0")
        self.assertEqual(
            openmvs["variants"]["cuda"]["sha256"],
            "6aac6b14ef478e501d2514cd1d74ed20e659b53b3bc2835d45a473ddfb921621",
        )
        self.assertEqual(
            openmvs["variants"]["cpu"]["sha256"],
            "0c31660c15c9ebc4c106873cf67564d9570d404aef7a6403451da1b6178b2167",
        )
        foundationstereo = manifest["tools"]["foundationstereo"]
        self.assertEqual(foundationstereo["version"], "2.0")
        self.assertEqual(
            foundationstereo["variants"]["dynamic"]["sha256"],
            "a001a7bc0512a0bc3b3218194e924784e58b20656c6f1ea2c151024e555cfd64",
        )

    def test_manifest_rejects_install_directory_traversal(self) -> None:
        with temporary_directory() as temporary:
            path = Path(temporary) / "manifest.json"
            payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            payload["tools"]["colmap"]["variants"]["cuda"]["install_dir"] = "../escape"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ArchiveSafetyError):
                load_manifest(path)


class ArchiveTests(unittest.TestCase):
    def test_safe_extract_zip_extracts_regular_files(self) -> None:
        with temporary_directory() as temporary:
            root = Path(temporary)
            archive = root / "safe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("bin/tool.exe", b"tool")

            destination = root / "output"
            safe_extract_zip(archive, destination)

            self.assertEqual((destination / "bin" / "tool.exe").read_bytes(), b"tool")

    def test_safe_extract_zip_rejects_forward_and_backslash_traversal(self) -> None:
        for member in ("../escape.txt", "..\\escape.txt", "C:\\escape.txt", "/escape.txt"):
            with self.subTest(member=member), temporary_directory() as temporary:
                root = Path(temporary)
                archive = root / "unsafe.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(member, b"escape")

                with self.assertRaises(ArchiveSafetyError):
                    safe_extract_zip(archive, root / "output")
                self.assertFalse((root / "escape.txt").exists())

    def test_safe_extract_zip_rejects_symbolic_links(self) -> None:
        with temporary_directory() as temporary:
            root = Path(temporary)
            archive = root / "link.zip"
            link = zipfile.ZipInfo("link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(link, "outside")

            with self.assertRaises(ArchiveSafetyError):
                safe_extract_zip(archive, root / "output")

    def test_verify_sha256_reports_mismatch_without_changing_file(self) -> None:
        with temporary_directory() as temporary:
            path = Path(temporary) / "archive.zip"
            path.write_bytes(b"original")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_sha256(path, "0" * 64)
            self.assertEqual(path.read_bytes(), b"original")

    def test_find_7z_accepts_explicit_executable_or_directory(self) -> None:
        with temporary_directory() as temporary:
            executable = touch_file(Path(temporary) / "7-Zip" / "7z.exe")

            self.assertEqual(find_7z(executable), executable.resolve())
            self.assertEqual(find_7z(executable.parent), executable.resolve())

    def test_install_asset_uses_verified_cache_and_versioned_directory(self) -> None:
        with temporary_directory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            archive = cache / "demo.zip"
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("bin/demo.exe", b"demo")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tools": {
                            "demo": {
                                "version": "1.2.3",
                                "default_variant": "cpu",
                                "variants": {
                                    "cpu": {
                                        "url": "https://example.invalid/demo.zip",
                                        "sha256": digest,
                                        "archive": "demo.zip",
                                        "install_dir": "Demo-1.2.3",
                                        "probe": "bin/demo.exe",
                                    }
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            installed = install_asset(
                root,
                "demo",
                "cpu",
                manifest_path=manifest_path,
                download_dir=cache,
            )

            self.assertEqual(installed, root / "tools" / "Demo-1.2.3")
            self.assertEqual((installed / "bin" / "demo.exe").read_bytes(), b"demo")
            self.assertEqual(install_asset(
                root,
                "demo",
                "cpu",
                manifest_path=manifest_path,
                download_dir=cache,
            ), installed)

    def test_install_asset_supports_verified_direct_file(self) -> None:
        with temporary_directory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            model = cache / "model.onnx"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"onnx")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "tools": {
                            "model": {
                                "version": "1.0",
                                "default_variant": "dynamic",
                                "variants": {
                                    "dynamic": {
                                        "url": "https://example.invalid/model.onnx",
                                        "sha256": digest,
                                        "archive": "model.onnx",
                                        "install_dir": "Model-1.0",
                                        "probe": "model.onnx",
                                        "kind": "file",
                                    }
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            installed = install_asset(
                root,
                "model",
                "dynamic",
                manifest_path=manifest_path,
                download_dir=cache,
            )

            self.assertEqual((installed / "model.onnx").read_bytes(), b"onnx")


class ResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = temporary_directory()
        self.root = Path(self.temporary.name)
        manifest_target = self.root / "tools" / "manifest.json"
        manifest_target.parent.mkdir(parents=True)
        manifest_target.write_text(MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def resolver(self, **kwargs: object) -> ToolResolver:
        return ToolResolver(
            self.root,
            environ={},
            registry_entries=[],
            system_detection=False,
            **kwargs,
        )

    def test_manifest_defaults_resolve_colmap_and_openmvs(self) -> None:
        colmap = touch_file(self.root / "tools" / "COLMAP-4.1.1" / "bin" / "colmap.exe")
        openmvs_bin = self.root / "tools" / "OpenMVS-2.4.0" / "vc17" / "x64" / "Release"
        for command in OPENMVS_COMMANDS:
            touch_file(openmvs_bin / command)
        resolver = self.resolver()

        colmap_info = resolver.resolve_colmap()
        openmvs_info = resolver.resolve_openmvs(variant="cpu")

        self.assertTrue(colmap_info.available)
        self.assertEqual(colmap_info.path, colmap.resolve())
        self.assertEqual(colmap_info.version, "4.1.1")
        self.assertEqual(colmap_info.variant, "cuda")
        self.assertTrue(openmvs_info.available)
        self.assertEqual(openmvs_info.path, openmvs_bin.resolve())
        self.assertEqual(openmvs_info.version, "2.4.0")
        self.assertEqual(openmvs_info.variant, "cpu")

    def test_explicit_path_wins_and_invalid_explicit_does_not_fall_back(self) -> None:
        manifest_exe = touch_file(
            self.root / "tools" / "COLMAP-4.1.1" / "bin" / "colmap.exe"
        )
        explicit_exe = touch_file(self.root / "alternate" / "COLMAP-9.8.7-CPU" / "colmap.exe")
        resolver = self.resolver()

        info = resolver.resolve_colmap(explicit_exe)
        missing = resolver.resolve_colmap(self.root / "missing")

        self.assertEqual(info.path, explicit_exe.resolve())
        self.assertEqual(info.version, "9.8.7")
        self.assertEqual(info.variant, "nocuda")
        self.assertEqual(info.details["source"], "explicit")
        self.assertFalse(missing.available)
        self.assertNotEqual(missing.path, manifest_exe)

    def test_environment_path_precedes_manifest_default(self) -> None:
        touch_file(self.root / "tools" / "COLMAP-4.1.1" / "bin" / "colmap.exe")
        environment_exe = touch_file(self.root / "environment" / "colmap.exe")
        resolver = ToolResolver(
            self.root,
            environ={"colmap_path": str(environment_exe)},
            registry_entries=[],
            system_detection=False,
        )

        info = resolver.resolve_colmap()

        self.assertEqual(info.path, environment_exe.resolve())
        self.assertEqual(info.details["source"], "env:COLMAP_PATH")

    def test_realityscan_is_discovered_from_epic_manifest(self) -> None:
        manifest_dir = self.root / "Epic" / "Manifests"
        install_dir = self.root / "Apps" / "RealityScan_2.1"
        executable = touch_file(install_dir / "RealityScan.exe")
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "realityscan.item").write_text(
            json.dumps(
                {
                    "DisplayName": "RealityScan 2.1.1",
                    "InstallLocation": str(install_dir),
                    "LaunchExecutable": "RealityScan.exe",
                    "AppVersionString": "2.1.1.119166",
                    "AppName": "test-app",
                    "CatalogItemId": "test-catalog",
                    "bIsIncompleteInstall": False,
                }
            ),
            encoding="utf-8",
        )
        resolver = self.resolver(epic_manifest_dirs=[manifest_dir])

        info = resolver.resolve_realityscan()

        self.assertTrue(info.available)
        self.assertEqual(info.path, executable.resolve())
        self.assertEqual(info.version, "2.1.1.119166")
        self.assertEqual(info.details["source"], "epic_manifest")

    def test_metashape_version_and_edition_come_from_registry(self) -> None:
        install_dir = self.root / "Agisoft" / "Metashape"
        executable = touch_file(install_dir / "metashape.exe")
        resolver = ToolResolver(
            self.root,
            environ={},
            registry_entries=[
                {
                    "RegistryKey": "test-key",
                    "DisplayName": "Agisoft Metashape Professional",
                    "DisplayVersion": "2.3.0",
                    "InstallLocation": str(install_dir),
                    "Publisher": "Agisoft",
                }
            ],
            system_detection=False,
        )

        info = resolver.resolve_metashape()

        self.assertTrue(info.available)
        self.assertEqual(info.path, executable.resolve())
        self.assertEqual(info.version, "2.3.0")
        self.assertEqual(info.variant, "professional")
        self.assertEqual(info.details["edition"], "Professional")
        self.assertEqual(info.details["source"], "registry")

    def test_collect_tool_metadata_has_stable_keys(self) -> None:
        metadata = self.resolver().collect_tool_metadata()

        self.assertEqual(
            tuple(metadata),
            ("colmap", "openmvs", "realityscan", "metashape"),
        )
        self.assertTrue(all(not info.available for info in metadata.values()))
        self.assertIsInstance(metadata["colmap"].as_dict(), dict)


if __name__ == "__main__":
    unittest.main()
