from __future__ import annotations

from pathlib import Path

from recon_pipeline.artifacts import require_file, validate_mesh


def convert_colmap_textured_ply_to_obj(
    source: Path, destination: Path
) -> tuple[Path, Path, Path]:
    """Convert COLMAP's ASCII per-face-UV PLY to standard OBJ + MTL.

    COLMAP's mesh_texturer stores one UV pair per face corner in a PLY list
    property. Many common readers ignore that extension, whereas OBJ represents
    the same face-corner mapping directly with independent texture indices.
    """

    source = require_file(source, "COLMAP textured PLY")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    material_path = destination.with_suffix(".mtl")

    with source.open("r", encoding="ascii", errors="strict", newline="") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError(f"Not a PLY file: {source}")

        current_element: str | None = None
        vertex_count = 0
        face_count = 0
        vertex_properties: list[str] = []
        face_properties: list[tuple[str, bool]] = []
        texture_name: str | None = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {source}")
            fields = line.strip().split()
            if not fields:
                continue
            if fields[:2] == ["format", "ascii"]:
                continue
            if fields[0] == "format":
                raise ValueError(
                    f"COLMAP OBJ conversion requires ASCII PLY output: {source}"
                )
            if fields[:2] == ["comment", "TextureFile"] and len(fields) >= 3:
                texture_name = " ".join(fields[2:])
            elif fields[0] == "element" and len(fields) == 3:
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
                elif current_element == "face":
                    face_count = int(fields[2])
            elif fields[0] == "property":
                is_list = len(fields) >= 5 and fields[1] == "list"
                name = fields[-1]
                if current_element == "vertex":
                    if is_list:
                        raise ValueError(f"Unsupported vertex list property: {name}")
                    vertex_properties.append(name)
                elif current_element == "face":
                    face_properties.append((name, is_list))
            elif fields[0] == "end_header":
                break

        if vertex_count < 1 or face_count < 1:
            raise ValueError(f"Textured PLY has no surface geometry: {source}")
        if texture_name is None:
            raise ValueError(f"Textured PLY has no TextureFile comment: {source}")
        try:
            x_index = vertex_properties.index("x")
            y_index = vertex_properties.index("y")
            z_index = vertex_properties.index("z")
        except ValueError as exc:
            raise ValueError(f"Textured PLY has no XYZ vertex properties: {source}") from exc

        material_name = "material_0"
        with destination.open(
            "w", encoding="utf-8", newline="\n"
        ) as output:
            output.write(f"mtllib {material_path.name}\n")
            output.write("o mesh\n")
            for _ in range(vertex_count):
                values = stream.readline().split()
                if len(values) < len(vertex_properties):
                    raise ValueError(f"Truncated vertex data in {source}")
                output.write(
                    f"v {values[x_index]} {values[y_index]} {values[z_index]}\n"
                )

            output.write(f"usemtl {material_name}\n")
            texture_index = 1
            for _ in range(face_count):
                fields = stream.readline().split()
                if not fields:
                    raise ValueError(f"Truncated face data in {source}")
                cursor = 0
                properties: dict[str, list[str] | str] = {}
                for name, is_list in face_properties:
                    if is_list:
                        count = int(fields[cursor])
                        cursor += 1
                        properties[name] = fields[cursor : cursor + count]
                        cursor += count
                    else:
                        properties[name] = fields[cursor]
                        cursor += 1
                indices = properties.get("vertex_indices")
                texcoords = properties.get("texcoord")
                if not isinstance(indices, list) or not isinstance(texcoords, list):
                    raise ValueError(
                        f"Face has no vertex_indices/texcoord lists in {source}"
                    )
                if len(texcoords) != 2 * len(indices):
                    raise ValueError(f"Face UV count does not match its vertices in {source}")
                face_texture_indices: list[int] = []
                for corner in range(len(indices)):
                    u = texcoords[2 * corner]
                    v = texcoords[2 * corner + 1]
                    output.write(f"vt {u} {v}\n")
                    face_texture_indices.append(texture_index)
                    texture_index += 1
                output.write(
                    "f "
                    + " ".join(
                        f"{int(vertex) + 1}/{uv}"
                        for vertex, uv in zip(indices, face_texture_indices)
                    )
                    + "\n"
                )

    texture_path = require_file(source.parent / texture_name, "COLMAP texture atlas")
    material_path.write_text(
        "newmtl material_0\n"
        "Ka 1.000000 1.000000 1.000000\n"
        "Kd 1.000000 1.000000 1.000000\n"
        "Ks 0.000000 0.000000 0.000000\n"
        "d 1.0\n"
        "illum 1\n"
        f"map_Kd {texture_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_mesh(destination)
    return destination, material_path, texture_path
