from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True, slots=True)
class OutputLayout:
    root: Path

    @property
    def native(self) -> Path:
        return self.root / "native"

    @property
    def colmap(self) -> Path:
        return self.root / "colmap"

    @property
    def colmap_sparse(self) -> Path:
        return self.colmap / "sparse"

    @property
    def colmap_images(self) -> Path:
        return self.colmap / "images"

    @property
    def dense(self) -> Path:
        return self.root / "dense"

    @property
    def mesh(self) -> Path:
        return self.root / "mesh"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def manifest(self) -> Path:
        return self.root / "run.json"

    def create(self) -> None:
        for path in (self.native, self.colmap_sparse, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def ensure_new_output(path: Path, *, dry_run: bool = False) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new output path."
        )
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def iter_images(root: Path) -> list[Path]:
    images = [
        path
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.as_posix().lower())


def require_images(root: Path, *, minimum: int = 2) -> list[Path]:
    images = iter_images(root)
    if len(images) < minimum:
        raise ValueError(f"Expected at least {minimum} images in {root}, found {len(images)}")
    return images


def copy_image_tree(source: Path, destination: Path) -> list[Path]:
    source = Path(source)
    destination = Path(destination)
    copied: list[Path] = []
    for image in iter_images(source):
        target = destination / image.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
        copied.append(target)
    return copied


def numeric_model_dirs(root: Path) -> list[Path]:
    models = [path for path in Path(root).iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(models, key=lambda path: int(path.name))


def require_file(path: Path, label: str, *, allow_empty: bool = False) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not allow_empty and path.stat().st_size == 0:
        raise RuntimeError(f"Empty {label}: {path}")
    return path


def validate_mesh(path: Path) -> dict[str, int | str]:
    path = require_file(path, "mesh")
    suffix = path.suffix.lower()
    if suffix == ".ply":
        header = bytearray()
        with path.open("rb") as stream:
            while len(header) < 1024 * 1024:
                line = stream.readline()
                if not line:
                    break
                header.extend(line)
                if line.strip() == b"end_header":
                    break
        text = header.decode("ascii", errors="replace")
        vertex_match = re.search(r"^element vertex\s+(\d+)\s*$", text, re.MULTILINE)
        face_match = re.search(r"^element face\s+(\d+)\s*$", text, re.MULTILINE)
        vertices = int(vertex_match.group(1)) if vertex_match else 0
        faces = int(face_match.group(1)) if face_match else 0
        if vertices < 1 or faces < 1:
            raise RuntimeError(
                f"Mesh has no surface geometry: {path} "
                f"(vertices={vertices}, faces={faces})"
            )
        return {"format": "ply", "vertices": vertices, "faces": faces}
    if suffix == ".obj":
        vertices = 0
        faces = 0
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.startswith("v "):
                    vertices += 1
                elif line.startswith("f "):
                    faces += 1
        if vertices < 1 or faces < 1:
            raise RuntimeError(
                f"Mesh has no surface geometry: {path} "
                f"(vertices={vertices}, faces={faces})"
            )
        return {"format": "obj", "vertices": vertices, "faces": faces}
    return {"format": suffix.removeprefix(".") or "unknown", "vertices": -1, "faces": -1}


def _data_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _colmap_image_count(path: Path) -> int:
    records: list[str] = []
    started = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            continue
        if not started and not line.strip():
            continue
        started = True
        records.append(line)
    if not records:
        return 0
    # images.txt stores two physical lines per image. The second line may be
    # empty when there are no 2D observations, and splitlines may omit a final
    # empty line, so round up rather than counting only non-empty lines.
    return (len(records) + 1) // 2


def validate_colmap_text_model(model_dir: Path) -> dict[str, int]:
    model_dir = Path(model_dir)
    cameras = require_file(model_dir / "cameras.txt", "COLMAP cameras.txt")
    images = require_file(model_dir / "images.txt", "COLMAP images.txt")
    points = require_file(
        model_dir / "points3D.txt", "COLMAP points3D.txt", allow_empty=True
    )
    camera_lines = _data_lines(cameras)
    image_count = _colmap_image_count(images)
    point_lines = _data_lines(points)
    if not camera_lines:
        raise RuntimeError(f"COLMAP model contains no cameras: {model_dir}")
    if image_count < 1:
        raise RuntimeError(f"COLMAP model contains no registered images: {model_dir}")
    return {
        "cameras": len(camera_lines),
        "images": image_count,
        "points3D": len(point_lines),
    }


def write_empty_points3d(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
        "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0\n",
        encoding="utf-8",
        newline="\n",
    )


def find_colmap_text_model(root: Path) -> Path:
    root = Path(root)
    candidates = sorted(root.rglob("cameras.txt"), key=lambda path: len(path.parts))
    for cameras in candidates:
        parent = cameras.parent
        if (parent / "images.txt").is_file():
            return parent
    raise FileNotFoundError(f"No COLMAP text model found below {root}")


def copy_colmap_text_model(
    source: Path,
    destination: Path,
    *,
    allow_missing_points: bool = False,
) -> dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    generated_empty = False
    for name in ("cameras.txt", "images.txt"):
        require_file(source / name, f"source {name}")
        shutil.copy2(source / name, destination / name)
    points = source / "points3D.txt"
    if points.is_file():
        shutil.copy2(points, destination / "points3D.txt")
    elif allow_missing_points:
        write_empty_points3d(destination / "points3D.txt")
        generated_empty = True
    else:
        raise FileNotFoundError(f"Missing source points3D.txt: {points}")
    stats = validate_colmap_text_model(destination)
    return {"stats": stats, "generated_empty_points3D": generated_empty}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def existing_files(paths: Iterable[Path]) -> list[Path]:
    return [Path(path) for path in paths if Path(path).is_file()]
