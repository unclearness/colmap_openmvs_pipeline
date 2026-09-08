"""Lossless face-corner UV export for OpenMVS ASCII/binary PLY files."""
from __future__ import annotations

import shutil
import struct
from pathlib import Path

from recon_pipeline.artifacts import require_file, validate_mesh


TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def export_obj(source: Path, destination: Path) -> dict:
    """Preserve topology, normalized UVs and texture indices; never rebake."""
    require_file(source, "textured PLY")
    if destination.exists() or destination.with_suffix(".mtl").exists():
        raise FileExistsError(f"OBJ/MTL already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        elements, textures = [], []
        encoding = None
        for _ in range(10000):
            line = stream.readline().decode("ascii").strip()
            fields = line.split()
            if not fields:
                raise ValueError("Invalid PLY header")
            if fields[0] == "format":
                encoding = fields[1]
            elif fields[0] == "element":
                elements.append((fields[1], int(fields[2]), []))
            elif fields[0] == "property":
                elements[-1][2].append(fields[1:])
            elif fields[:2] == ["comment", "TextureFile"]:
                textures.append(line.split(maxsplit=2)[2])
            elif fields[0] == "end_header":
                break
        else:
            raise ValueError("PLY header too long")
        if encoding not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format: {encoding}")
        if not textures:
            raise ValueError("PLY contains no TextureFile comments")
        atlas_paths = []
        for name in textures:
            if Path(name).name != name or ":" in name or "\\" in name:
                raise ValueError(f"Unsafe texture name: {name}")
            atlas = require_file(source.parent / name, "texture atlas")
            target = destination.parent / name
            if atlas.resolve() != target.resolve():
                if target.exists():
                    raise FileExistsError(target)
                shutil.copy2(atlas, target)
            atlas_paths.append(str(target))
        endian = "<" if encoding == "binary_little_endian" else ">"

        def read_value(kind, tokens):
            if encoding == "ascii":
                return float(next(tokens)) if TYPES[kind] in "fd" else int(next(tokens))
            fmt = endian + TYPES[kind]
            return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]

        vertices = 0
        faces = 0
        uv_index = 1
        has_normals = False
        with destination.open("w", encoding="utf-8", newline="\n") as out:
            out.write(f"mtllib {destination.with_suffix('.mtl').name}\no mesh\ns 1\n")
            for element, count, properties in elements:
                for _ in range(count):
                    tokens = iter(stream.readline().decode("ascii").split()) if encoding == "ascii" else None
                    record = {}
                    for prop in properties:
                        if prop[0] == "list":
                            size = read_value(prop[1], tokens)
                            if not 0 <= size <= 1000000:
                                raise ValueError("Invalid PLY list length")
                            record[prop[-1]] = [read_value(prop[2], tokens) for _ in range(size)]
                        else:
                            record[prop[-1]] = read_value(prop[0], tokens)
                    if element == "vertex":
                        out.write(f"v {record['x']:.17g} {record['y']:.17g} {record['z']:.17g}\n")
                        has_normals = all(key in record for key in ("nx", "ny", "nz"))
                        if has_normals:
                            out.write(f"vn {record['nx']:.17g} {record['ny']:.17g} {record['nz']:.17g}\n")
                        vertices += 1
                    elif element == "face":
                        indices = record["vertex_indices"]
                        uv = record["texcoord"]
                        texture = int(record.get("texnumber", 0))
                        if len(indices) < 3 or len(uv) != 2 * len(indices):
                            raise ValueError("Face UV count mismatch")
                        if not 0 <= texture < len(textures):
                            raise ValueError("Invalid texture index")
                        if any(index < 0 or index >= vertices for index in indices):
                            raise ValueError("Invalid vertex index")
                        for index in range(0, len(uv), 2):
                            out.write(f"vt {uv[index]:.17g} {uv[index+1]:.17g}\n")
                        out.write(f"usemtl material_{texture}\n")
                        out.write("f " + " ".join(f"{v+1}/{uv_index+i}" + (f"/{v+1}" if has_normals else "") for i, v in enumerate(indices)) + "\n")
                        uv_index += len(indices)
                        faces += 1
    destination.with_suffix(".mtl").write_text(
        "".join(f"newmtl material_{i}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {name}\n\n" for i, name in enumerate(textures)),
        encoding="utf-8",
    )
    validate_mesh(destination)
    return {"obj": str(destination), "mtl": str(destination.with_suffix('.mtl')),
            "textures": atlas_paths, "vertices": vertices, "faces": faces}
