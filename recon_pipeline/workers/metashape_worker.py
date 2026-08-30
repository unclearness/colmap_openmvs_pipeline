"""Standalone worker executed by Metashape Professional or its Python module.

The proprietary module is intentionally imported only after argument and JSON
validation.  This keeps the main pipeline importable on machines without
Metashape and avoids consuming a license during command construction.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_TARGETS = {"sfm", "mesh"}
_PRESETS = {"fast", "normal", "high"}
_MATCH_DOWNSCALE = {"fast": 4, "normal": 2, "high": 1}
_DEPTH_DOWNSCALE = {"fast": 8, "normal": 4, "high": 2}


@dataclass(frozen=True, slots=True)
class MetashapeJob:
    image_dir: Path
    target: str
    preset: str
    texture: bool
    copy_images: bool
    project_path: Path
    mesh_path: Path
    colmap_model_path: Path
    colmap_images_path: Path
    result_path: Path

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MetashapeJob:
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported Metashape job schema_version (expected 1)")

        required = (
            "image_dir",
            "target",
            "preset",
            "project_path",
            "mesh_path",
            "colmap_model_path",
            "colmap_images_path",
            "result_path",
        )
        missing = [name for name in required if not data.get(name)]
        if missing:
            raise ValueError(
                "Metashape job is missing required fields: " + ", ".join(missing)
            )

        target = str(data["target"]).lower()
        if target == "dense":
            raise ValueError(
                "Metashape dense point-cloud output is intentionally unsupported; "
                "choose target 'sfm' or 'mesh'"
            )
        if target not in _TARGETS:
            raise ValueError(f"Unsupported Metashape target: {target}")

        preset = str(data["preset"]).lower()
        if preset not in _PRESETS:
            raise ValueError(
                f"Unsupported Metashape preset: {preset}; choose fast, normal, or high"
            )

        return cls(
            image_dir=Path(str(data["image_dir"])),
            target=target,
            preset=preset,
            texture=bool(data.get("texture", False)),
            copy_images=bool(data.get("copy_images", True)),
            project_path=Path(str(data["project_path"])),
            mesh_path=Path(str(data["mesh_path"])),
            colmap_model_path=Path(str(data["colmap_model_path"])),
            colmap_images_path=Path(str(data["colmap_images_path"])),
            result_path=Path(str(data["result_path"])),
        )


def load_job(path: Path) -> MetashapeJob:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Metashape job JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Metashape job JSON root must be an object")
    return MetashapeJob.from_mapping(data)


def _load_metashape() -> ModuleType:
    try:
        return importlib.import_module("Metashape")
    except Exception as exc:
        raise RuntimeError(
            "Could not import the Metashape Python API. Python scripting requires "
            "Metashape Professional; the Standard edition is not supported. Use "
            "metashape.exe -r, or an explicit Python interpreter with the matching "
            "Metashape module and license configuration."
        ) from exc


def _preflight_professional_api(metashape: ModuleType | Any) -> str:
    app = getattr(metashape, "app", None)
    if app is None:
        raise RuntimeError(
            "The imported module is not a complete Metashape Professional API "
            "(Metashape.app is missing)."
        )
    if getattr(app, "activated", False) is not True:
        raise RuntimeError(
            "Metashape Professional is not activated for this Python process. "
            "Metashape Standard does not provide Python scripting. Activate a "
            "Professional/trial license, or configure the standalone module to find "
            "the Professional license used by the GUI installation."
        )
    if not hasattr(metashape, "CamerasFormatColmap"):
        raise RuntimeError(
            "This Metashape Python API has no CamerasFormatColmap exporter. "
            "Metashape 2.1.3 or newer is required."
        )
    if not hasattr(metashape, "Document"):
        raise RuntimeError("The imported Metashape module has no Document API")
    return str(getattr(app, "version", "unknown"))


def _find_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )
    if len(images) < 2:
        raise ValueError(
            f"Metashape requires at least two input images; found {len(images)} in "
            f"{image_dir}"
        )
    return images


def _save_document(document: Any, project_path: Path | None = None) -> None:
    if project_path is None:
        document.save()
    else:
        document.save(str(project_path))


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run_job(job: MetashapeJob, metashape: ModuleType | Any | None = None) -> dict[str, Any]:
    api = _load_metashape() if metashape is None else metashape
    version = _preflight_professional_api(api)
    images = _find_images(job.image_dir)

    job.project_path.parent.mkdir(parents=True, exist_ok=True)
    # Metashape's COLMAP exporter takes an anchor file, then creates the
    # standard sparse/0 and images layout below the anchor's parent directory.
    colmap_root = job.colmap_model_path.parents[1]
    colmap_root.mkdir(parents=True, exist_ok=True)
    colmap_anchor = colmap_root / "colmap.txt"

    document = api.Document()
    _save_document(document, job.project_path)
    chunk = document.addChunk()
    chunk.addPhotos([str(path) for path in images])
    _save_document(document)

    chunk.matchPhotos(
        downscale=_MATCH_DOWNSCALE[job.preset],
        generic_preselection=True,
        reference_preselection=False,
    )
    _save_document(document)
    chunk.alignCameras()
    _save_document(document)

    aligned = sum(
        1 for camera in chunk.cameras if getattr(camera, "transform", None) is not None
    )
    if aligned == 0:
        raise RuntimeError(
            "Metashape aligned no cameras. Check image overlap, focus, and exposure."
        )

    # Use Metashape's official COLMAP exporter (introduced in Metashape 2.1.3).
    chunk.exportCameras(
        path=str(colmap_anchor),
        format=api.CamerasFormatColmap,
        save_points=True,
        save_images=job.copy_images,
        use_labels=False,
        convert_to_pinhole=job.copy_images,
        binary=False,
    )

    if job.target == "mesh":
        job.mesh_path.parent.mkdir(parents=True, exist_ok=True)
        chunk.buildDepthMaps(
            downscale=_DEPTH_DOWNSCALE[job.preset],
            filter_mode=api.MildFiltering,
            reuse_depth=False,
        )
        _save_document(document)
        chunk.buildModel(
            source_data=api.DepthMapsData,
            surface_type=api.Arbitrary,
            interpolation=api.EnabledInterpolation,
            face_count=api.HighFaceCount,
            vertex_colors=True,
            keep_depth=True,
            build_texture=job.texture,
        )
        _save_document(document)
        chunk.exportModel(
            path=str(job.mesh_path),
            format=api.ModelFormatOBJ,
            save_texture=job.texture,
            save_uv=job.texture,
            save_colors=True,
            save_normals=True,
        )
        _save_document(document)

    result = {
        "metashape_version": version,
        "input_images": len(images),
        "aligned_cameras": aligned,
        "target": job.target,
        "project": str(job.project_path),
        "colmap_model": str(job.colmap_model_path),
        "colmap_images": str(job.colmap_images_path) if job.copy_images else None,
        "mesh": str(job.mesh_path) if job.target == "mesh" else None,
    }
    _write_result(job.result_path, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a JSON-configured Metashape job")
    parser.add_argument("--config", type=Path, required=True, help="Metashape job JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        job = load_job(args.config)
        result = run_job(job)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"Metashape worker failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
