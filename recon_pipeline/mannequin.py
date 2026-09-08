"""Video/images -> masked SfM -> FoundationStereo -> OpenMVS 2.3 texture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Sequence

from recon_pipeline.artifacts import ensure_new_output, require_file, validate_colmap_text_model, validate_mesh, write_json
from recon_pipeline.foundation_stereo import DEFAULT_MODEL, DEFAULT_MODEL_SHA256, build_parser as stereo_parser, validate_arguments as validate_stereo, read_colmap_images
from recon_pipeline.mannequin_ply import export_obj
from recon_pipeline.process import CommandRunner


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
TEXTURE_SETTINGS = {
    "resolution-level": "0", "cost-smoothness-ratio": "0.1",
    "global-seam-leveling": "1", "local-seam-leveling": "1",
    "sharpness-weight": "0", "cuda-device": "-2",
    "outlier-threshold": "0.006", "virtual-face-images": "0",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_images(root: Path) -> list[Path]:
    paths = sorted((p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                    and not p.name.lower().endswith(".mask.png")), key=lambda p: p.relative_to(root).as_posix().casefold())
    if len(paths) < 3:
        raise ValueError(f"At least three overlapping images are required: {root}")
    return paths


def backend_for_platform(requested: str, platform: str) -> str:
    selected = ("realityscan" if platform == "win32" else "colmap") if requested == "auto" else requested
    if selected == "realityscan" and platform != "win32":
        raise ValueError("RealityScan automation requires Windows; use --sfm-backend colmap or --sfm-root on Ubuntu")
    return selected


def default_python(environment: str) -> str:
    candidate = ROOT / environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate if candidate.is_file() else Path(sys.executable))


def executable(value: str | Path) -> Path:
    found = shutil.which(str(value))
    return Path(found or value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mannequin-pipeline", description=__doc__)
    parser.add_argument("input", type=Path, help="Video or image folder; with --sfm-root, the images referenced by that model")
    parser.add_argument("output", type=Path, help="New or empty output directory")
    parser.add_argument("--config", type=Path, help="Replay recipe.json defaults; explicit command-line flags take precedence")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-image-size", type=int, default=1920)
    parser.add_argument("--sfm-backend", choices=["auto", "realityscan", "colmap"], default="auto")
    parser.add_argument("--sfm-root", type=Path, help="Reuse COLMAP root containing sparse/<id>/*.txt; input must be its image folder")
    parser.add_argument("--input-mesh", type=Path, help="Texture an existing mesh in the --sfm-root coordinate system; skips SAM3/stereo")
    parser.add_argument("--shared-intrinsics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--matcher", choices=["auto", "sequential", "exhaustive"], default="auto")
    parser.add_argument("--min-registration-ratio", type=float, default=0.8)
    parser.add_argument("--sam-python", default=default_python(".venv-sam3"))
    parser.add_argument("--stereo-python", default=default_python(".venv-open3d"))
    parser.add_argument("--sam-model", default="facebook/sam3")
    parser.add_argument("--sam-revision", default="main", help="Cached Hugging Face commit; resolved snapshot recorded, no automatic download")
    parser.add_argument("--prompts", default="mannequin|mannequin head|cosmetology mannequin head")
    parser.add_argument("--sam-threshold", type=float, default=0.15)
    parser.add_argument("--mask-threshold", type=float, default=0.45)
    parser.add_argument("--mask-kernel", type=int, default=9)
    parser.add_argument("--min-mask-fraction", type=float, default=0.005)
    parser.add_argument("--max-mask-fraction", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--max-pairs", type=int, default=96)
    parser.add_argument("--stereo-config", type=Path, help="JSON overrides of FoundationStereo argparse keys; resolved full settings are recorded")
    parser.add_argument("--mesher", choices=["poisson", "tsdf"], default="poisson")
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--poisson-trim", type=float, default=0.01)
    parser.add_argument("--mesh-faces", type=int, default=250000)
    parser.add_argument("--largest-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--colmap-exe", default=str(ROOT / "tools/COLMAP-4.1.1/bin/colmap.exe") if os.name == "nt" else "colmap")
    parser.add_argument("--realityscan-exe", type=Path)
    parser.add_argument("--openmvs-dir", type=Path, default=ROOT / "tools/OpenMVS-2.3.0" if os.name == "nt" else None)
    parser.add_argument("--timeout", type=float, default=1800, help="Per-command timeout; not a total runtime guarantee")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print resolved plan without loading models or running reconstruction")
    return parser


def resolve(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    config.pop("config", None)
    for name in ("input", "output", "sfm_root", "input_mesh", "model", "realityscan_exe", "openmvs_dir", "stereo_config"):
        if config[name] is not None:
            config[name] = str(config[name].expanduser().resolve())
    source, output = Path(config["input"]), Path(config["output"])
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        collect_images(source)
        if output == source or output.is_relative_to(source):
            raise ValueError("Output must not be inside the input image tree")
    elif source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Input must be a video or folder containing overlapping images")
    for name in ("fps", "timeout"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("max_image_size", "max_pairs", "mesh_faces", "threads", "mask_kernel"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if config["device_id"] < 0 or config["mask_kernel"] % 2 != 1 or not 4 <= config["poisson_depth"] <= 12:
        raise ValueError("Invalid device, mask kernel (must be odd) or Poisson depth (4..12)")
    for name in ("sam_threshold", "mask_threshold", "poisson_trim", "min_registration_ratio"):
        if not 0 <= config[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not 0 < config["min_mask_fraction"] < config["max_mask_fraction"] <= 1:
        raise ValueError("Invalid mask area bounds")
    if config["poisson_trim"] == 1:
        raise ValueError("poisson-trim must be less than 1")
    if config["input_mesh"] and not config["sfm_root"]:
        raise ValueError("--input-mesh requires --sfm-root")
    if config["sfm_root"] and not source.is_dir():
        raise ValueError("--sfm-root requires its image folder as input")
    config["sfm_backend"] = "imported" if config["sfm_root"] else backend_for_platform(args.sfm_backend, sys.platform)
    config["matcher"] = ("exhaustive" if source.is_dir() else "sequential") if args.matcher == "auto" else args.matcher
    config["prompts"] = [p.strip() for p in args.prompts.split("|") if p.strip()]
    if not config["prompts"]:
        raise ValueError("At least one SAM3 prompt is required")
    for name in ("ffmpeg", "colmap_exe", "sam_python", "stereo_python"):
        config[name] = str(executable(config[name]))
    suffix = ".exe" if os.name == "nt" else ""
    for name in ("TextureMesh", "InterfaceCOLMAP"):
        config[name] = str(executable(Path(config["openmvs_dir"]) / (name + suffix) if config["openmvs_dir"] else name))
    config["texture_settings"] = dict(TEXTURE_SETTINGS)
    return config


def texture_command(config: dict, scene: str | Path, mesh: str | Path, output: str | Path, native: str | Path) -> list[str]:
    command = [config["TextureMesh"], "-i", str(scene), "-m", str(mesh), "-o", str(output), "-w", str(native)]
    for key, value in config["texture_settings"].items():
        command.extend(["--" + key, value])
    return command + ["--max-threads", str(config["threads"]), "-v", "3"]


def select_model(root: Path) -> Path:
    models = list((root / "sparse").glob("*/cameras.txt"))
    if not models:
        raise FileNotFoundError(f"Expected COLMAP text model under {root}/sparse/<id>")
    ranked = [(validate_colmap_text_model(p.parent)["images"], p.parent) for p in models]
    return sorted(ranked, key=lambda item: (-item[0], str(item[1])))[0][1]


def stereo_settings(config: dict, model: Path, images: Path, destination: Path) -> argparse.Namespace:
    args = stereo_parser().parse_args([str(model), str(images), str(destination)])
    args.model = Path(config["model"])
    args.model_sha256 = config["model_sha256"]
    args.max_pairs = config["max_pairs"]
    args.device_id = config["device_id"]
    # Spatial coverage of the accepted mannequin experiment, with LR checking enabled.
    args.spatial_bins = 24
    args.references_per_bin = 4
    args.sources_per_reference = 4
    args.max_baseline_ratio = 0.5
    args.target_baseline_ratio = 0.25
    args.max_view_angle = 25.0
    if config["stereo_config"]:
        changes = json.loads(Path(config["stereo_config"]).read_text(encoding="utf-8"))
        protected = {"colmap_model", "images", "output", "colmap_fused", "model", "model_sha256", "depth_only", "provider"}
        for key, value in changes.items():
            if key not in vars(args) or key in protected:
                raise ValueError(f"Unsupported stereo override: {key}")
            setattr(args, key, value)
    validate_stereo(args)
    if args.rectification_alpha != 1 or not args.multiview_consistency or args.min_consistent_views < 2 or args.normal_consistency_angle <= 0:
        raise ValueError("Mannequin fusion requires full FOV, multiple independent supports and normal consistency")
    return args


def run(config: dict) -> dict:
    output, source = Path(config["output"]), Path(config["input"])
    ensure_new_output(output, dry_run=config["dry_run"])
    native = output / "native"
    planned = {"schema": 1, "settings": config, "stages": ["normalize", "SAM3", config["sfm_backend"], "undistort", "SAM3-undistorted", "FoundationStereo", config["mesher"], "OpenMVS-2.3.0-CPU", "OBJ"]}
    if config["input_mesh"]:
        validate_mesh(Path(config["input_mesh"]))
        planned["stages"] = ["import-SfM", "undistort", "OpenMVS-2.3.0-CPU", "OBJ"]
    if config["sfm_root"]:
        select_model(Path(config["sfm_root"]))
        if not config["input_mesh"]:
            planned["stages"] = ["import-SfM", "undistort", "SAM3-undistorted", "FoundationStereo", config["mesher"], "OpenMVS-2.3.0-CPU", "OBJ"]
    stereo = stereo_settings(config, output / "colmap/sparse/0", native / "masked_undistorted/images", native / "foundationstereo")
    planned["stereo_settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(stereo).items()}
    if config["dry_run"]:
        planned["status"] = "planned-not-executed"
        print(json.dumps(planned, indent=2))
        return planned
    native.mkdir(parents=True)
    if "recipe" in config:
        write_json(output / "recipe.json", config["recipe"])
    manifest = output / "run.json"
    config_path = output / "settings.json"
    write_json(config_path, config)
    runner = CommandRunner(output / "logs/pipeline.log", timeout_seconds=config["timeout"])
    started = time.monotonic()
    planned.update(status="running", completed_stages=[], hashes={})

    def checkpoint(stage):
        planned["completed_stages"].append(stage)
        planned["commands"] = [r.to_dict() for r in runner.records]
        write_json(manifest, planned)

    def worker(operation, src, dest, python):
        runner.run(operation, [python, "-m", "recon_pipeline.mannequin_worker", operation, config_path, src, dest], cwd=ROOT)

    try:
        checkpoint("created")
        planned["hashes"]["source_code"] = {
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in (ROOT / "recon_pipeline").rglob("*.py")
        }
        needed = ["TextureMesh", "InterfaceCOLMAP", "colmap_exe"]
        if config["sfm_backend"] == "realityscan":
            from recon_pipeline.backends.realityscan import resolve_realityscan_executable

            config["realityscan_exe"] = str(resolve_realityscan_executable(config["realityscan_exe"]))
            needed.append("realityscan_exe")
        if not config["input_mesh"]:
            needed += ["sam_python", "stereo_python"]
            require_file(Path(config["model"]), "FoundationStereo ONNX model")
            actual = sha256(config["model"])
            if actual != config["model_sha256"]:
                raise ValueError("FoundationStereo model SHA-256 mismatch")
            planned["hashes"]["foundationstereo"] = actual
        if source.is_file():
            needed.append("ffmpeg")
            planned["hashes"]["video"] = sha256(source)
        for tool in needed:
            require_file(Path(config[tool]), tool)
            planned["hashes"][tool] = sha256(config[tool])
        runner.run("OpenMVS version", [config["TextureMesh"], "--help"], cwd=native, expected_exit_codes=(0, 1))
        version_text = runner.log_path.read_text(encoding="utf-8")
        version_text += "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in native.glob("TextureMesh*.log"))
        if not re.search(r"OpenMVS[^\n]*v2\.3\.0\b", version_text):
            raise RuntimeError("This recipe requires OpenMVS 2.3.0; pass --openmvs-dir (no silent 2.4 substitution)")
        if "cuda-device" not in version_text:
            # CPU-only Linux builds do not define this command-line option.
            config["texture_settings"].pop("cuda-device", None)
        write_json(config_path, config)
        if not config["input_mesh"]:
            worker("preflight-sam", source, native / "sam_environment.json", config["sam_python"])
            worker("preflight-stereo", source, native / "stereo_environment.json", config["stereo_python"])
            sam_environment = json.loads((native / "sam_environment.json").read_text(encoding="utf-8"))
            config["sam_model"] = sam_environment["snapshot"]
            planned["hashes"]["sam_snapshot"] = {
                p.relative_to(config["sam_model"]).as_posix(): sha256(p)
                for p in Path(config["sam_model"]).rglob("*") if p.is_file()
            }
            write_json(config_path, config)
            if "recipe" in config:
                snapshot = Path(config["sam_model"])
                if re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
                    config["recipe"]["sam_revision"] = snapshot.name
                protected = {"colmap_model", "images", "output", "colmap_fused", "model", "model_sha256", "depth_only", "provider"}
                overrides = {k: v for k, v in planned["stereo_settings"].items() if k not in protected}
                write_json(output / "stereo_overrides.json", overrides)
                config["recipe"]["stereo_config"] = str(output / "stereo_overrides.json")
                write_json(output / "recipe.json", config["recipe"])
        checkpoint("preflight")
        if config["sfm_root"]:
            sfm_model, sfm_images = select_model(Path(config["sfm_root"])), source
        else:
            image_source = source
            if source.is_file():
                image_source = native / "frames"
                image_source.mkdir()
                runner.run("video frames", [config["ffmpeg"], "-nostdin", "-n", "-i", source, "-vf", f"fps={config['fps']}", image_source / "frame_%06d.png"])
            normalized = native / "normalized"
            worker("normalize", image_source, normalized, config["sam_python"])
            worker("mask", normalized, native / "masked", config["sam_python"])
            checkpoint("input masks")
            sfm_images = native / "masked/images"
            command = [sys.executable, "-m", "recon_pipeline", "run", str(sfm_images), str(native / "sfm"), "--backend", config["sfm_backend"], "--target", "sfm", "--timeout", str(config["timeout"])]
            if config["sfm_backend"] == "realityscan":
                command += ["--realityscan-distortion-model", "brown3", "--realityscan-distortion-prior", "unknown", "--realityscan-sensitive-alignment"]
                if config["shared_intrinsics"]:
                    command += ["--realityscan-shared-intrinsics"]
                if config["realityscan_exe"]:
                    command += ["--realityscan-exe", config["realityscan_exe"]]
            else:
                command += ["--colmap-exe", config["colmap_exe"], "--camera-model", "FULL_OPENCV", "--camera-mode", "1" if config["shared_intrinsics"] else "3", "--matcher", config["matcher"]]
            runner.run("SfM", command, cwd=ROOT)
            sfm_model = select_model(native / "sfm/colmap")
            # RealityScan exports undistorted images; COLMAP SfM exports original images.
            sfm_images = native / "sfm/colmap/images"
        stats = validate_colmap_text_model(sfm_model)
        image_records = read_colmap_images(sfm_model / "images.txt")
        for record in image_records.values():
            relative = PurePosixPath(record.name.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or ":" in record.name:
                raise ValueError(f"Unsafe registered image path: {record.name}")
            require_file(sfm_images / Path(*relative.parts), "registered image")
        planned["hashes"]["sfm"] = {p.name: sha256(p) for p in sfm_model.glob("*.txt")}
        planned["hashes"]["registered_images"] = {r.name: sha256(sfm_images / r.name) for r in image_records.values()}
        input_count = len(collect_images(sfm_images))
        if not config["sfm_root"]:
            input_count = len(collect_images(native / "normalized"))
        ratio = stats["images"] / input_count
        planned["registration"] = {**stats, "input_images": input_count, "ratio": ratio}
        if stats["images"] < 3 or ratio < config["min_registration_ratio"]:
            raise RuntimeError(f"Only {stats['images']}/{input_count} images registered; inspect SfM, try higher FPS or change matching before dense processing")
        checkpoint("SfM")
        undistorted = native / "undistorted"
        runner.run("undistort", [config["colmap_exe"], "image_undistorter", "--image_path", sfm_images, "--input_path", sfm_model,
                   "--output_path", undistorted, "--output_type", "COLMAP", "--copy_policy", "copy"])
        model = output / "colmap/sparse/0"
        model.mkdir(parents=True)
        runner.run("export cameras", [config["colmap_exe"], "model_converter", "--input_path", undistorted / "sparse", "--output_path", model, "--output_type", "TXT"])
        validate_colmap_text_model(model)
        (output / "sparse").mkdir()
        runner.run("sparse PLY", [config["colmap_exe"], "model_converter", "--input_path", model, "--output_path", output / "sparse/sparse.ply", "--output_type", "PLY"])
        shutil.copytree(undistorted / "images", output / "colmap/images")
        checkpoint("undistorted COLMAP export")
        mesh_path = Path(config["input_mesh"]) if config["input_mesh"] else native / "geometry/mesh.ply"
        if not config["input_mesh"]:
            worker("mask", output / "colmap/images", native / "masked_undistorted", config["sam_python"])
            # Resolve all FoundationStereo defaults before invoking its existing worker.
            stereo_config = native / "stereo_settings.json"
            write_json(stereo_config, planned["stereo_settings"])
            runner.run("FoundationStereo", [config["stereo_python"], "-m", "recon_pipeline.mannequin_stereo", stereo_config], cwd=ROOT)
            checkpoint("FoundationStereo")
            worker("mesh", native / "foundationstereo", native / "geometry", config["stereo_python"])
            checkpoint("geometry")
        planned["hashes"]["input_mesh"] = sha256(mesh_path)
        validate_mesh(mesh_path)
        interface_input = native / "openmvs_input"
        shutil.copytree(model, interface_input / "sparse")
        mvs = native / "openmvs"
        mvs.mkdir()
        scene = mvs / "scene.mvs"
        runner.run("InterfaceCOLMAP", [config["InterfaceCOLMAP"], "-i", interface_input, "--image-folder", output / "colmap/images", "-o", scene, "-w", mvs], cwd=mvs)
        textured = output / "mesh"
        textured.mkdir()
        runner.run("TextureMesh", texture_command(config, scene, mesh_path, textured / "mesh.ply", mvs), cwd=mvs)
        planned["artifacts"] = export_obj(textured / "mesh.ply", textured / "mesh.obj")
        planned["artifacts"].update(colmap=str(model), sparse=str(output / "sparse/sparse.ply"))
        planned["hashes"]["obj"] = sha256(textured / "mesh.obj")
        planned["status"] = "complete"
        checkpoint("textured OBJ")
        return planned
    except Exception as exc:
        planned.update(status="failed", error=str(exc))
        raise
    finally:
        planned["elapsed_seconds"] = time.monotonic() - started
        planned["commands"] = [r.to_dict() for r in runner.records]
        write_json(manifest, planned)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--config", type=Path)
        first, _ = probe.parse_known_args(argv)
        if first.config:
            values = json.loads(first.config.read_text(encoding="utf-8"))
            allowed = {action.dest for action in parser._actions} - {"input", "output", "config", "help", "dry_run"}
            if not isinstance(values, dict) or set(values) - allowed:
                raise ValueError("Recipe must contain only supported CLI option names")
            parser.set_defaults(**values)
        args = parser.parse_args(argv)
        for action in parser._actions:
            if action.choices is not None and getattr(args, action.dest) not in action.choices:
                raise ValueError(f"Invalid recipe option {action.dest}")
        config = resolve(args)
        if not args.dry_run:
            ensure_new_output(Path(config["output"]))
            recipe = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                      if key not in {"input", "output", "config", "dry_run"}}
            # run() owns output creation; write recipe after it has reserved the directory.
            config["recipe"] = recipe
        result = run(config)
        if not args.dry_run:
            print(json.dumps(result.get("artifacts", {}), indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
