from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from recon_pipeline import __version__
from recon_pipeline.artifacts import iter_images
from recon_pipeline.models import (
    BackendName,
    Matcher,
    Mesher,
    PipelineConfig,
    Preset,
    Target,
)


def _path(value: str) -> Path:
    return Path(value)


def _intrinsics(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "intrinsics must be comma-separated numbers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("intrinsics cannot be empty")
    return result


def _add_pipeline_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=[value.value for value in BackendName],
        default=BackendName.COLMAP.value,
        help="Reconstruction engine. openmvs uses COLMAP for SfM.",
    )
    parser.add_argument(
        "--target",
        choices=[value.value for value in Target],
        default=Target.MESH.value,
        help="Last stage to run. RealityScan/Metashape support sfm and mesh only.",
    )
    parser.add_argument(
        "--preset",
        choices=[value.value for value in Preset],
        default=Preset.NORMAL.value,
    )
    parser.add_argument(
        "--matcher",
        choices=[value.value for value in Matcher],
        default=Matcher.SEQUENTIAL.value,
    )
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=(0, 1, 2, 3),
        default=1,
        help="COLMAP camera mode: 0 auto, 1 single, 2 per-folder, 3 per-image.",
    )
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--intrinsics", type=_intrinsics)
    parser.add_argument("--gpu-index", default="-1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--forward-motion", action="store_true")
    parser.add_argument("--fixed-intrinsics", action="store_true")
    parser.add_argument(
        "--mesher",
        choices=[value.value for value in Mesher],
        default=Mesher.AUTO.value,
        help="COLMAP mesher. auto uses Delaunay when present, otherwise Poisson.",
    )
    parser.add_argument("--refine-mesh", action="store_true")
    parser.add_argument("--texture", action="store_true")
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument("--timeout", type=float, dest="timeout_seconds")
    parser.add_argument("--dry-run", action="store_true")

    tools = parser.add_argument_group("tool overrides")
    tools.add_argument("--colmap-exe", type=_path)
    tools.add_argument("--openmvs-dir", type=_path)
    tools.add_argument(
        "--openmvs-variant", choices=("auto", "cuda", "cpu"), default="auto"
    )
    tools.add_argument("--realityscan-exe", type=_path)
    tools.add_argument(
        "--realityscan-quality", choices=("normal", "high"), default="normal"
    )
    tools.add_argument("--realityscan-no-distortion", action="store_true")
    tools.add_argument(
        "--realityscan-distortion-model",
        choices=(
            "division",
            "brown3",
            "brown4",
            "brown3-tangential2",
            "brown4-tangential2",
        ),
        help="Explicit RealityScan lens model used for alignment.",
    )
    tools.add_argument(
        "--realityscan-distortion-prior",
        choices=("unknown", "approximate", "fixed"),
        help="Prior strength for the selected RealityScan lens model.",
    )
    tools.add_argument(
        "--realityscan-shared-intrinsics",
        action="store_true",
        help=(
            "Put all RealityScan inputs in one calibration and lens group. "
            "Use this for frames from one fixed camera/lens."
        ),
    )
    tools.add_argument(
        "--realityscan-sensitive-alignment",
        action="store_true",
        help="Increase RealityScan feature counts and use Ultra detector sensitivity.",
    )
    tools.add_argument("--metashape-exe", type=_path)
    tools.add_argument("--metashape-python", type=_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recon-pipeline",
        description=(
            "Run SfM, dense reconstruction, or meshing with COLMAP, OpenMVS, "
            "RealityScan, or Metashape."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Process one image directory")
    run_parser.add_argument("image_dir", type=_path)
    run_parser.add_argument("output_dir", type=_path)
    _add_pipeline_options(run_parser)

    batch_parser = subparsers.add_parser(
        "batch", help="Process each image-bearing child directory"
    )
    batch_parser.add_argument("data_dir", type=_path)
    batch_parser.add_argument("output_root", type=_path)
    batch_parser.add_argument("--continue-on-error", action="store_true")
    _add_pipeline_options(batch_parser)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Report installed engine paths, versions, and capabilities"
    )
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    doctor_parser.add_argument("--colmap-exe", type=_path)
    doctor_parser.add_argument("--openmvs-dir", type=_path)
    doctor_parser.add_argument(
        "--openmvs-variant", choices=("auto", "cuda", "cpu"), default="auto"
    )
    doctor_parser.add_argument("--realityscan-exe", type=_path)
    doctor_parser.add_argument("--metashape-exe", type=_path)
    return parser


def _config_from_args(
    args: argparse.Namespace, image_dir: Path, output_dir: Path
) -> PipelineConfig:
    return PipelineConfig(
        image_dir=image_dir,
        output_dir=output_dir,
        backend=BackendName(args.backend),
        target=Target(args.target),
        preset=Preset(args.preset),
        matcher=Matcher(args.matcher),
        camera_mode=args.camera_mode,
        camera_model=args.camera_model,
        intrinsic_prior=args.intrinsics,
        use_gpu=not args.cpu,
        gpu_index=args.gpu_index,
        forward_motion=args.forward_motion,
        fixed_intrinsics=args.fixed_intrinsics,
        mesher=Mesher(args.mesher),
        refine_mesh=args.refine_mesh,
        texture=args.texture,
        copy_images=not args.no_copy_images,
        dry_run=args.dry_run,
        timeout_seconds=args.timeout_seconds,
        colmap_exe=args.colmap_exe,
        openmvs_dir=args.openmvs_dir,
        openmvs_variant=args.openmvs_variant,
        realityscan_exe=args.realityscan_exe,
        realityscan_quality=args.realityscan_quality,
        realityscan_no_distortion=args.realityscan_no_distortion,
        realityscan_distortion_model=args.realityscan_distortion_model,
        realityscan_distortion_prior=args.realityscan_distortion_prior,
        realityscan_shared_intrinsics=args.realityscan_shared_intrinsics,
        realityscan_sensitive_alignment=args.realityscan_sensitive_alignment,
        metashape_exe=args.metashape_exe,
        metashape_python=args.metashape_python,
    )


def _print_result(result: object) -> None:
    payload = result.to_dict()  # type: ignore[attr-defined]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            from recon_pipeline.orchestrator import doctor

            report = doctor(
                colmap_exe=args.colmap_exe,
                openmvs_dir=args.openmvs_dir,
                openmvs_variant=args.openmvs_variant,
                realityscan_exe=args.realityscan_exe,
                metashape_exe=args.metashape_exe,
            )
            if args.as_json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                for name, details in report.items():
                    status = "available" if details.get("available") else "unavailable"
                    version = details.get("version") or "unknown version"
                    path = details.get("path") or "not found"
                    print(f"{name:12} {status:11} {version:16} {path}")
                    note = details.get("note")
                    if note:
                        print(f"  {note}")
            return 0

        from recon_pipeline.orchestrator import execute_pipeline

        if args.command == "run":
            result = execute_pipeline(
                _config_from_args(args, args.image_dir, args.output_dir)
            )
            _print_result(result)
            return 0

        data_dir = args.data_dir.expanduser().resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        datasets = [
            child
            for child in sorted(data_dir.iterdir(), key=lambda path: path.name.lower())
            if child.is_dir() and iter_images(child)
        ]
        if not datasets:
            raise ValueError(f"No image-bearing child directories found in {data_dir}")
        failures: list[tuple[Path, Exception]] = []
        for dataset in datasets:
            print(f"\n=== dataset: {dataset.name} ===", flush=True)
            try:
                result = execute_pipeline(
                    _config_from_args(
                        args, dataset, args.output_root / dataset.name
                    )
                )
                _print_result(result)
            except Exception as exc:  # continue mode needs per-dataset isolation
                if not args.continue_on_error:
                    raise
                failures.append((dataset, exc))
                print(f"FAILED {dataset.name}: {exc}", file=sys.stderr)
        if failures:
            print(f"{len(failures)} dataset(s) failed", file=sys.stderr)
            return 1
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
