from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recon_pipeline.textured_mesh import convert_colmap_textured_ply_to_obj


class ColmapTexturedMeshTests(unittest.TestCase):
    def test_per_face_uv_ply_is_converted_to_standard_obj(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "mesh.ply"
            source.write_text(
                "ply\n"
                "format ascii 1.0\n"
                "comment TextureFile texture.png\n"
                "element vertex 3\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "element face 1\n"
                "property list uchar int vertex_indices\n"
                "property list uchar float texcoord\n"
                "end_header\n"
                "0 0 0\n1 0 0\n0 1 0\n"
                "3 0 1 2 6 0 0 1 0 0 1\n",
                encoding="ascii",
            )
            (root / "texture.png").write_bytes(b"png")

            obj, mtl, texture = convert_colmap_textured_ply_to_obj(
                source, root / "mesh.obj"
            )

            obj_text = obj.read_text(encoding="utf-8")
            self.assertIn("mtllib mesh.mtl\n", obj_text)
            self.assertIn("vt 1 0\n", obj_text)
            self.assertIn("f 1/1 2/2 3/3\n", obj_text)
            self.assertIn("map_Kd texture.png", mtl.read_text(encoding="utf-8"))
            self.assertEqual(texture, root / "texture.png")


if __name__ == "__main__":
    unittest.main()
