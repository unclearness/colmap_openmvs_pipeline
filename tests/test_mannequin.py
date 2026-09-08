from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recon_pipeline.mannequin import (
    TEXTURE_SETTINGS, backend_for_platform, build_parser, collect_images,
    main, resolve, run, select_model, stereo_settings, texture_command,
)
from recon_pipeline.mannequin_ply import export_obj


class MannequinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images with spaces"
        self.images.mkdir()
        for name in ("a.JPG", "b.png", "c.jpeg"):
            (self.images / name).write_bytes(b"image placeholder")

    def config(self, *options):
        return resolve(build_parser().parse_args([str(self.images), str(self.root / "result"), *options]))

    def test_platform_selection(self):
        self.assertEqual(backend_for_platform("auto", "win32"), "realityscan")
        self.assertEqual(backend_for_platform("auto", "linux"), "colmap")
        with self.assertRaisesRegex(ValueError, "Windows"):
            backend_for_platform("realityscan", "linux")

    def test_images_ignore_mask_and_support_formats(self):
        (self.images / "a.JPG.mask.png").write_bytes(b"mask")
        self.assertEqual(len(collect_images(self.images)), 3)

    def test_image_matching_defaults_exhaustive(self):
        self.assertEqual(self.config()["matcher"], "exhaustive")

    def test_video_matching_defaults_sequential(self):
        video = self.root / "clip.mp4"
        video.write_bytes(b"video")
        args = build_parser().parse_args([str(video), str(self.root / "out")])
        self.assertEqual(resolve(args)["matcher"], "sequential")

    def test_reject_nested_output(self):
        args = build_parser().parse_args([str(self.images), str(self.images / "out")])
        with self.assertRaisesRegex(ValueError, "inside"):
            resolve(args)

    def test_dry_run_creates_no_files_or_processes(self):
        config = self.config("--dry-run")
        with patch("recon_pipeline.mannequin.CommandRunner") as runner, patch("builtins.print"):
            result = run(config)
        runner.assert_not_called()
        self.assertEqual(result["status"], "planned-not-executed")
        self.assertFalse(Path(config["output"]).exists())

    def test_existing_output_is_preserved(self):
        config = self.config("--dry-run")
        output = Path(config["output"])
        output.mkdir()
        (output / "user.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            run(config)
        self.assertEqual((output / "user.txt").read_text(), "keep")

    def test_explicit_texture_parameters_and_argv_spaces(self):
        config = self.config()
        command = texture_command(config, self.root / "scene.mvs", self.root / "mesh input.ply", self.root / "mesh.ply", self.root)
        self.assertIn(str(self.root / "mesh input.ply"), command)
        for key, value in TEXTURE_SETTINGS.items():
            self.assertEqual(command[command.index("--" + key) + 1], value)
        config["texture_settings"].pop("cuda-device")
        command = texture_command(config, "scene", "mesh", "out", "native")
        self.assertNotIn("--cuda-device", command)

    def test_invalid_numeric_settings(self):
        for options in [("--fps", "nan"), ("--mask-kernel", "2"), ("--max-pairs", "0"), ("--poisson-trim", "1")]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.config(*options)

    def test_protected_stereo_paths(self):
        overrides = self.root / "stereo.json"
        overrides.write_text(json.dumps({"output": "elsewhere"}))
        config = self.config("--stereo-config", str(overrides))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            stereo_settings(config, self.root, self.root, self.root)

    def test_stereo_requires_consistent_full_fov(self):
        overrides = self.root / "stereo.json"
        overrides.write_text(json.dumps({"rectification_alpha": 0}))
        config = self.config("--stereo-config", str(overrides))
        with self.assertRaisesRegex(ValueError, "full FOV"):
            stereo_settings(config, self.root, self.root, self.root)

    def test_recipe_cli_override(self):
        recipe = self.root / "recipe.json"
        recipe.write_text(json.dumps({"fps": 4.0, "max_pairs": 24}))
        with patch("recon_pipeline.mannequin.run", return_value={}) as execute:
            result = main([str(self.images), str(self.root / "out"), "--config", str(recipe), "--fps", "3", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertEqual(execute.call_args.args[0]["fps"], 3.0)
        self.assertEqual(execute.call_args.args[0]["max_pairs"], 24)

    def test_failed_preflight_records_failure_without_reconstruction(self):
        config = self.config("--sfm-backend", "colmap")
        config["model"] = str(self.root / "missing.onnx")
        with patch("recon_pipeline.mannequin.CommandRunner.run") as execute:
            with self.assertRaises(FileNotFoundError):
                run(config)
        execute.assert_not_called()
        manifest = json.loads((Path(config["output"]) / "run.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertNotIn("artifacts", manifest)

    def test_largest_component_selection(self):
        for name in ("0", "1"):
            model = self.root / "sparse" / name
            model.mkdir(parents=True)
            (model / "cameras.txt").write_text("placeholder")
        with patch("recon_pipeline.mannequin.validate_colmap_text_model", side_effect=lambda path: {"images": 10 if path.name == "0" else 50}):
            self.assertEqual(select_model(self.root).name, "1")


class PlyExportTests(unittest.TestCase):
    def fixture(self, root, encoding, texture="atlas.png"):
        source = root / "source.ply"
        header = f"ply\nformat {encoding} 1.0\ncomment TextureFile {texture}\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nproperty list uchar float texcoord\nend_header\n".encode()
        if encoding == "ascii":
            body = b"0 0 0\n1 0 0\n0 1 0\n3 0 1 2 6 0 0 1 0 0 1\n"
        else:
            endian = "<" if encoding == "binary_little_endian" else ">"
            body = struct.pack(endian + "9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
            body += struct.pack(endian + "B3iB6f", 3, 0, 1, 2, 6, 0, 0, 1, 0, 0, 1)
        source.write_bytes(header + body)
        (root / "atlas.png").write_bytes(b"image")
        return source

    def test_ascii_and_both_endian_formats_preserve_uvs(self):
        for encoding in ("ascii", "binary_little_endian", "binary_big_endian"):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = self.fixture(root, encoding)
                target = root / "publish/mesh.obj"
                result = export_obj(source, target)
                text = target.read_text()
                self.assertEqual(result["faces"], 1)
                self.assertIn("vt 0 1\n", text)
                self.assertIn("f 1/1 2/2 3/3", text)
                self.assertEqual((target.parent / "atlas.png").read_bytes(), b"image")
                with self.assertRaises(FileExistsError):
                    export_obj(source, target)

    def test_reject_texture_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root, "ascii", "../atlas.png")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                export_obj(source, root / "mesh.obj")


class ImageNormalizationTests(unittest.TestCase):
    def test_nested_duplicate_names_and_exif_rotation(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is an optional worker dependency")
        from recon_pipeline.mannequin_worker import normalize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            for folder in ("a", "b"):
                (source / folder).mkdir(parents=True)
                exif = Image.Exif()
                exif[274] = 6
                Image.new("RGB", (4, 8), "red").save(source / folder / "same.jpg", exif=exif)
            Image.new("L", (8, 4), 128).save(source / "third.tiff")
            target = root / "normalized"
            normalize(source, target, {"max_image_size": 20, "shared_intrinsics": True})
            self.assertEqual(len(list(target.glob("*.png"))), 3)
            for path in target.glob("*.png"):
                with Image.open(path) as image:
                    self.assertEqual(image.size, (8, 4))
                    self.assertEqual(image.mode, "RGB")
            with Image.open(source / "a/same.jpg") as image:
                self.assertEqual(image.size, (4, 8))


class MeshTopologyTests(unittest.TestCase):
    def test_nonmanifold_edges_are_removed_before_texturing(self):
        try:
            import open3d as o3d
        except ImportError:
            self.skipTest("Open3D is an optional worker dependency")
        from recon_pipeline.mannequin_worker import mesh
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            surface = o3d.geometry.TriangleMesh()
            surface.vertices = o3d.utility.Vector3dVector([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]])
            surface.triangles = o3d.utility.Vector3iVector([[0,1,2],[1,0,3],[0,1,4]])
            source = root / "input.ply"
            o3d.io.write_triangle_mesh(str(source), surface)
            (root / "run.json").write_text(json.dumps({"mesh": str(source)}))
            mesh(root, root / "result", {"mesher":"tsdf", "largest_component":False, "mesh_faces":100})
            result = o3d.io.read_triangle_mesh(str(root / "result/mesh.ply"))
            self.assertTrue(result.is_edge_manifold(allow_boundary_edges=True))
            self.assertTrue(result.is_vertex_manifold())
            self.assertGreater(len(result.triangles), 0)
            self.assertEqual(len(o3d.io.read_triangle_mesh(str(root / "result/mesh_full.ply")).triangles), 3)


if __name__ == "__main__":
    unittest.main()
