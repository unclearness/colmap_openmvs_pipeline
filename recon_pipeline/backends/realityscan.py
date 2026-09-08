"""RealityScan command-line backend.

RealityScan can align images and reconstruct a mesh, but it does not expose a
standalone dense point-cloud stage.  This backend therefore supports only the
``sfm`` and ``mesh`` targets.  Every successful run also exports a COLMAP text
model and its undistorted images.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from recon_pipeline.artifacts import (
    IMAGE_EXTENSIONS,
    OutputLayout,
    copy_colmap_text_model,
    find_colmap_text_model,
    iter_images,
    require_file,
)
from recon_pipeline.models import BackendName, BackendResult, PipelineConfig, Target
from recon_pipeline.process import CommandRunner


REALITYSCAN_ENV_VARS = (
    "REALITYSCAN_EXE",
    "REALITYSCAN_PATH",
    "REALITYSCAN_HOME",
)


def _version_key(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", str(path.parent))
    return tuple(int(number) for number in numbers)


def _as_executable(candidate: str | Path) -> Path:
    path = Path(candidate).expanduser()
    if path.is_dir() or path.suffix.lower() != ".exe":
        path /= "RealityScan.exe"
    return path.resolve()


def _known_realityscan_candidates(
    environ: Mapping[str, str],
) -> list[Path]:
    """Return installed RealityScan candidates, newest version first."""

    roots: list[Path] = []
    for variable in ("ProgramW6432", "ProgramFiles"):
        value = environ.get(variable)
        if value:
            root = Path(value)
            if root not in roots:
                roots.append(root)
    if not roots:
        roots.append(Path(r"C:\Program Files"))

    discovered: list[Path] = []
    for root in roots:
        epic_games = root / "Epic Games"
        if epic_games.is_dir():
            discovered.extend(epic_games.glob("RealityScan_*/RealityScan.exe"))
            discovered.extend(epic_games.glob("RealityScan*/RealityScan.exe"))
        discovered.extend(
            (
                epic_games / "RealityScan" / "RealityScan.exe",
                epic_games / "RealityScan_2.1" / "RealityScan.exe",
            )
        )

    on_path = shutil.which("RealityScan.exe")
    if on_path:
        discovered.append(Path(on_path))

    unique: dict[str, Path] = {}
    for path in discovered:
        resolved = path.expanduser().resolve()
        unique.setdefault(str(resolved).casefold(), resolved)
    return sorted(unique.values(), key=_version_key, reverse=True)


def resolve_realityscan_executable(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    known_paths: Sequence[str | Path] | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve RealityScan from an explicit value, environment, then installs.

    Environment variables are checked in :data:`REALITYSCAN_ENV_VARS` order.
    A value may name either ``RealityScan.exe`` itself or its containing
    directory.  An invalid explicit or environment override is reported rather
    than silently replaced by a different installation.
    """

    environment = os.environ if environ is None else environ
    if explicit is not None:
        path = _as_executable(explicit)
        if must_exist and not path.is_file():
            raise FileNotFoundError(f"RealityScan executable not found: {path}")
        return path

    for variable in REALITYSCAN_ENV_VARS:
        value = environment.get(variable)
        if not value:
            continue
        path = _as_executable(value)
        if must_exist and not path.is_file():
            raise FileNotFoundError(
                f"RealityScan executable from {variable} was not found: {path}"
            )
        return path

    candidates = (
        [_as_executable(path) for path in known_paths]
        if known_paths is not None
        else _known_realityscan_candidates(environment)
    )
    for path in candidates:
        if path.is_file():
            return path
    if not must_exist and candidates:
        return candidates[0]
    if not must_exist:
        return Path(
            r"C:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe"
        )

    attempted = ", ".join(str(path) for path in candidates) or "no known paths"
    raise FileNotFoundError(
        "RealityScan.exe was not found. Pass --realityscan-exe or set "
        f"REALITYSCAN_EXE. Checked: {attempted}"
    )


def _flat_name(relative: Path) -> str:
    if len(relative.parts) == 1:
        return relative.name
    return "__".join(relative.parts)


def plan_flattened_images(image_dir: Path) -> list[tuple[Path, str]]:
    """Plan a deterministic flat RealityScan input directory.

    Directory components are joined with ``__``.  The preflight comparison is
    case-insensitive because RealityScan is a Windows application.  This also
    detects ambiguous inputs such as ``a/photo.jpg`` and ``a__photo.jpg``.
    """

    root = Path(image_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")

    sources = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    sources.sort(key=lambda path: path.relative_to(root).as_posix().casefold())
    if len(sources) < 2:
        raise ValueError(f"Expected at least 2 images in {root}, found {len(sources)}")

    plan: list[tuple[Path, str]] = []
    destinations: dict[str, Path] = {}
    for source in sources:
        relative = source.relative_to(root)
        name = _flat_name(relative)
        key = name.casefold()
        previous = destinations.get(key)
        if previous is not None:
            raise ValueError(
                "Flattened image-name collision for "
                f"{name!r}: {previous.relative_to(root)} and {relative}"
            )
        destinations[key] = source
        plan.append((source, name))
    return plan


def flatten_images(image_dir: Path, destination: Path) -> list[Path]:
    """Copy images into a flat directory after a complete collision preflight."""

    plan = plan_flattened_images(image_dir)
    destination = Path(destination)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(
            f"Flattened image destination is not a directory: {destination}"
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Flattened image destination is not empty: {destination}"
        )
    targets = [destination / name for _, name in plan]

    destination.mkdir(parents=True, exist_ok=True)
    for (source, _), target in zip(plan, targets, strict=True):
        shutil.copy2(source, target)
    return targets


def _asset_paths(assets_dir: Path | None = None) -> dict[str, Path]:
    root = (
        Path(assets_dir)
        if assets_dir is not None
        else Path(__file__).resolve().parents[1] / "assets" / "realityscan"
    )
    assets = {
        "sparse": root / "sparse_point_cloud.xml",
        "colmap": root / "colmap_undistorted.xml",
        "mesh": root / "export_obj.xml",
    }
    for label, path in assets.items():
        require_file(path, f"RealityScan {label} XML")
    return assets


def _coerce_target(target: Target | str) -> Target:
    try:
        result = Target(target)
    except ValueError as exc:
        raise ValueError(f"Unsupported RealityScan target: {target}") from exc
    if result is Target.DENSE:
        raise ValueError(
            "RealityScan has no standalone dense point-cloud target; "
            "choose sfm or mesh"
        )
    return result


def build_command(
    executable: str | Path,
    image_dir: Path,
    output_dir: Path,
    *,
    target: Target | str = Target.SFM,
    quality: str = "normal",
    texture: bool = False,
    no_distortion: bool = False,
    distortion_model: str | None = None,
    distortion_prior: str | None = None,
    shared_intrinsics: bool = False,
    sensitive_alignment: bool = False,
    assets_dir: Path | None = None,
) -> list[str]:
    """Build the official RealityScan CLI argv without filesystem mutations."""

    selected_target = _coerce_target(target)
    quality = quality.lower()
    if quality not in {"normal", "high"}:
        raise ValueError("RealityScan mesh quality must be normal or high")
    distortion_models = {
        "division": ("1", "Division"),
        "brown3": ("2", "Brown3"),
        "brown4": ("3", "Brown4"),
        "brown3-tangential2": ("4", "Brown3WithTangential2"),
        "brown4-tangential2": ("5", "Brown4WithTangential2"),
    }
    distortion_priors = {"unknown": "0", "approximate": "1", "fixed": "2"}
    if distortion_model is not None and distortion_model not in distortion_models:
        raise ValueError(f"Unsupported RealityScan distortion model: {distortion_model}")
    if distortion_prior is not None and distortion_prior not in distortion_priors:
        raise ValueError(f"Unsupported RealityScan distortion prior: {distortion_prior}")
    if no_distortion and (distortion_model is not None or distortion_prior is not None):
        raise ValueError(
            "no_distortion cannot be combined with an explicit distortion model or prior"
        )

    assets = _asset_paths(assets_dir)
    layout = OutputLayout(Path(output_dir))
    native = layout.native / "realityscan"
    crash_dir = native / "crash_reports"
    export_dir = native / "colmap_export"
    project = native / "project.rsproj"
    sparse_cloud = native / "sparse.ply"
    registration_anchor = export_dir / "colmap.txt"
    mesh = layout.mesh / "0" / "mesh.obj"

    argv = [
        str(executable),
        "-stdConsole",
        "-headless",
        "-silent",
        str(crash_dir),
        "-set",
        "appQuitOnError=true",
        "-newScene",
        "-addFolder",
        str(image_dir),
        "-selectAllImages",
    ]
    if shared_intrinsics:
        argv.extend(("-setConstantCalibrationGroups", "-setPriorLensGroup", "0"))
    if sensitive_alignment:
        for key, value in (
            ("sfmFeatureDetectionQuality", "High"),
            ("sfmMaxFeaturesPerMpx", "20000"),
            ("sfmMaxFeaturesPerImage", "80000"),
            ("sfmImagesOverlap", "High"),
            ("sfmDetectorSensitivity", "Ultra"),
            ("sfmPreselectorFeatures", "20000"),
            ("sfmForceComponentRematch", "true"),
        ):
            argv.extend(("-set", f"{key}={value}"))
    if no_distortion:
        argv.extend(("-editInputSelection", "inpDistortionModel=0"))
    elif distortion_model is not None:
        input_value, alignment_value = distortion_models[distortion_model]
        argv.extend(("-set", f"sfmDistortionModel={alignment_value}"))
        argv.extend(("-editInputSelection", f"inpDistortionModel={input_value}"))
    if distortion_prior is not None:
        argv.extend(
            ("-editInputSelection", f"inpDistortion={distortion_priors[distortion_prior]}")
        )
    argv.extend(
        (
            "-align",
            "-selectMaximalComponent",
            "-exportSparsePointCloud",
            str(sparse_cloud),
            str(assets["sparse"]),
            "-exportRegistration",
            str(registration_anchor),
            str(assets["colmap"]),
        )
    )

    if selected_target is Target.MESH:
        argv.append("-setReconstructionRegionAuto")
        argv.append(
            "-calculateHighModel" if quality == "high" else "-calculateNormalModel"
        )
        if texture:
            argv.extend(("-unwrap", "-calculateTexture"))
        argv.extend(("-exportSelectedModel", str(mesh), str(assets["mesh"])))

    argv.extend(("-save", str(project), "-quit"))
    return argv


def _copy_exported_images(export_dir: Path, destination: Path) -> list[Path]:
    images = iter_images(export_dir)
    if not images:
        raise FileNotFoundError(
            f"RealityScan exported no undistorted COLMAP images below {export_dir}"
        )

    names: dict[str, Path] = {}
    for image in images:
        key = image.name.casefold()
        previous = names.get(key)
        if previous is not None:
            raise RuntimeError(
                "RealityScan produced colliding flat image names: "
                f"{previous} and {image}"
            )
        names[key] = image

    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for image in images:
        target = destination / image.name
        if target.exists():
            raise FileExistsError(f"COLMAP image already exists: {target}")
        shutil.copy2(image, target)
        copied.append(target)
    return copied


def _registered_image_names(images_txt: Path) -> list[str]:
    """Read image names from the two-line COLMAP text representation."""

    lines = images_txt.read_text(encoding="utf-8", errors="replace").splitlines()
    names: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise RuntimeError(f"Invalid COLMAP image record in {images_txt}: {line}")
        names.append(fields[9])
        if index < len(lines):
            index += 1
    return names


def _validate_registered_images(model_dir: Path, image_dir: Path) -> int:
    names = _registered_image_names(model_dir / "images.txt")
    if not names:
        raise RuntimeError(f"COLMAP model contains no registered images: {model_dir}")

    missing: list[str] = []
    for name in names:
        relative = PurePosixPath(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe COLMAP image path exported by RealityScan: {name}")
        if not (image_dir.joinpath(*relative.parts)).is_file():
            missing.append(name)
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} registered COLMAP image(s) in {image_dir}: {preview}"
        )
    return len(names)


class RealityScanBackend:
    """RealityScan SfM/mesh backend with mandatory COLMAP export."""

    def __init__(
        self,
        executable: Path | None = None,
        assets_dir: Path | None = None,
    ) -> None:
        self.executable = Path(executable) if executable is not None else None
        self.assets_dir = Path(assets_dir) if assets_dir is not None else None

    def build_command(
        self,
        config: PipelineConfig,
        *,
        executable: Path | None = None,
        image_dir: Path | None = None,
    ) -> list[str]:
        selected_executable = executable or config.realityscan_exe or self.executable
        if selected_executable is None:
            selected_executable = resolve_realityscan_executable(
                must_exist=not config.dry_run
            )
        flat_input = image_dir or (
            OutputLayout(config.output_dir).native / "realityscan" / "input"
        )
        return build_command(
            selected_executable,
            flat_input,
            config.output_dir,
            target=config.target,
            quality=config.realityscan_quality,
            texture=config.texture,
            no_distortion=config.realityscan_no_distortion,
            distortion_model=config.realityscan_distortion_model,
            distortion_prior=config.realityscan_distortion_prior,
            shared_intrinsics=config.realityscan_shared_intrinsics,
            sensitive_alignment=config.realityscan_sensitive_alignment,
            assets_dir=self.assets_dir,
        )

    def run(self, config: PipelineConfig, runner: CommandRunner) -> BackendResult:
        target = _coerce_target(config.target)
        quality = config.realityscan_quality.lower()
        if quality not in {"normal", "high"}:
            raise ValueError("RealityScan mesh quality must be normal or high")

        image_dir = Path(config.image_dir).expanduser().resolve()
        output_dir = Path(config.output_dir).expanduser().resolve()
        plan = plan_flattened_images(image_dir)
        dry_run = bool(config.dry_run or runner.dry_run)
        executable = resolve_realityscan_executable(
            config.realityscan_exe or self.executable,
            must_exist=not dry_run,
        )

        layout = OutputLayout(output_dir)
        native = layout.native / "realityscan"
        flat_input = native / "input"
        export_dir = native / "colmap_export"
        project = native / "project.rsproj"
        sparse_cloud = native / "sparse.ply"
        model_dir = layout.colmap_sparse / "0"
        mesh = layout.mesh / "0" / "mesh.obj"

        argv = build_command(
            executable,
            flat_input,
            output_dir,
            target=target,
            quality=quality,
            texture=config.texture,
            no_distortion=config.realityscan_no_distortion,
            distortion_model=config.realityscan_distortion_model,
            distortion_prior=config.realityscan_distortion_prior,
            shared_intrinsics=config.realityscan_shared_intrinsics,
            sensitive_alignment=config.realityscan_sensitive_alignment,
            assets_dir=self.assets_dir,
        )

        metadata: dict[str, object] = {
            "executable": str(executable),
            "quality": quality,
            "texture": bool(config.texture and target is Target.MESH),
            "distortion_model": config.realityscan_distortion_model,
            "distortion_prior": config.realityscan_distortion_prior,
            "shared_intrinsics": config.realityscan_shared_intrinsics,
            "sensitive_alignment": config.realityscan_sensitive_alignment,
            "input_images": len(plan),
            "sparse_point_cloud": str(sparse_cloud),
            "command": argv,
            "dry_run": dry_run,
        }
        predicted_meshes = [mesh] if target is Target.MESH else []
        if dry_run:
            if runner.dry_run:
                runner.run("RealityScan", argv)
            return BackendResult(
                backend=BackendName.REALITYSCAN,
                target=target,
                output_dir=output_dir,
                colmap_models=[model_dir],
                meshes=predicted_meshes,
                native_project=project,
                metadata=metadata,
            )

        for directory in (
            native,
            export_dir,
            native / "crash_reports",
            model_dir,
            layout.colmap_images,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if target is Target.MESH:
            mesh.parent.mkdir(parents=True, exist_ok=True)

        flatten_images(image_dir, flat_input)
        runner.run("RealityScan", argv, cwd=native)

        require_file(sparse_cloud, "RealityScan sparse point cloud")
        require_file(project, "RealityScan project")
        source_model = find_colmap_text_model(export_dir)
        colmap_info = copy_colmap_text_model(
            source_model,
            model_dir,
            allow_missing_points=True,
        )
        exported_images = _copy_exported_images(export_dir, layout.colmap_images)
        registered_images = _validate_registered_images(
            model_dir, layout.colmap_images
        )
        metadata.update(
            {
                "colmap": colmap_info,
                "exported_images": len(exported_images),
                "registered_images": registered_images,
            }
        )

        meshes: list[Path] = []
        if target is Target.MESH:
            meshes.append(require_file(mesh, "RealityScan OBJ mesh"))

        return BackendResult(
            backend=BackendName.REALITYSCAN,
            target=target,
            output_dir=output_dir,
            colmap_models=[model_dir],
            meshes=meshes,
            native_project=project,
            metadata=metadata,
        )


__all__ = [
    "REALITYSCAN_ENV_VARS",
    "RealityScanBackend",
    "build_command",
    "flatten_images",
    "plan_flattened_images",
    "resolve_realityscan_executable",
]
