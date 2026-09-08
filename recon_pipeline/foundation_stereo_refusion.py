from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from recon_pipeline.artifacts import ensure_new_output, write_json
from recon_pipeline.foundation_stereo import (
    ColmapImage,
    DepthView,
    StereoPair,
    _lazy_imports,
    _world_to_rectified_camera,
    estimate_world_normals_from_depth,
    multiview_consistency_mask,
    read_colmap_cameras,
    read_colmap_images,
    read_colmap_points,
    rectify_pair,
)


def _pair_artifact_index(pair_root: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for directory in pair_root.iterdir():
        if not directory.is_dir() or not (directory / "depth.npy").is_file():
            continue
        parts = directory.name.split("_", 1)
        if len(parts) != 2:
            continue
        result[(parts[1].casefold(), directory.name)] = directory
    return result


def _find_pair_artifact(
    index: dict[tuple[str, str], Path], reference: str, source: str
) -> Path:
    suffix = f"{Path(reference).stem}_{Path(source).stem}".casefold()
    matches = [
        directory
        for (stored_suffix, _), directory in index.items()
        if stored_suffix == suffix
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one saved depth pair for {reference} + {source}, found {len(matches)}"
        )
    return matches[0]


def load_saved_depth_views(
    source_root: Path,
    source_payload: dict[str, Any],
    cameras: dict[int, Any],
    images: dict[int, ColmapImage],
    image_root: Path,
    np: Any,
    cv2: Any,
) -> tuple[list[DepthView], list[dict[str, Any]]]:
    images_by_name = {image.name.casefold(): image for image in images.values()}
    artifact_index = _pair_artifact_index(source_root / "pairs")
    inference_width, inference_height = source_payload["inference_size"]
    rectification_alpha = float(source_payload.get("rectification_alpha", 0.0))
    views: list[DepthView] = []
    records: list[dict[str, Any]] = []
    for source_record in source_payload["pairs"]:
        if not source_record.get("accepted"):
            continue
        reference_name = source_record["reference"]
        source_name = source_record["source"]
        reference = images_by_name.get(reference_name.casefold())
        source = images_by_name.get(source_name.casefold())
        if reference is None or source is None:
            raise FileNotFoundError(
                f"Saved pair references an image absent from the COLMAP model: "
                f"{reference_name}, {source_name}"
            )
        selected_pair = StereoPair(
            reference=reference,
            source=source,
            shared_points=int(source_record["shared_points"]),
            baseline=float(source_record["baseline"]),
            median_depth=float(source_record["median_depth"]),
            baseline_ratio=float(source_record["baseline_ratio"]),
            view_angle_degrees=float(source_record["view_angle_degrees"]),
        )
        rectified = rectify_pair(
            selected_pair,
            cameras,
            image_root,
            (int(inference_width), int(inference_height)),
            np,
            cv2,
            rectification_alpha=rectification_alpha,
        )
        if (
            rectified.left.name.casefold() != reference_name.casefold()
            or rectified.right.name.casefold() != source_name.casefold()
        ):
            raise RuntimeError(
                f"Saved rectification order changed for {reference_name} + {source_name}"
            )
        artifact = _find_pair_artifact(
            artifact_index, reference_name, source_name
        )
        depth = np.load(artifact / "depth.npy").astype(np.float32, copy=False)
        valid_image = cv2.imread(str(artifact / "valid.png"), cv2.IMREAD_GRAYSCALE)
        if valid_image is None:
            raise FileNotFoundError(f"Saved valid mask not found: {artifact / 'valid.png'}")
        valid = valid_image > 0
        if depth.shape != rectified.left_bgr.shape[:2] or valid.shape != depth.shape:
            raise RuntimeError(
                f"Saved depth shape does not match reconstructed rectification: {artifact}"
            )
        record = dict(source_record)
        record["source_artifact"] = str(artifact)
        record["integrated"] = False
        records.append(record)
        views.append(
            DepthView(
                pair=rectified,
                depth=depth,
                valid=valid,
                record_index=len(records) - 1,
                artifact_dir=None,
            )
        )
    if not views:
        raise RuntimeError(f"No accepted saved depth views found in {source_root}")
    return views, records


def run_refusion(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source.expanduser().resolve()
    source_manifest = source_root / "run.json" if source_root.is_dir() else source_root
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source run.json not found: {source_manifest}")
    source_root = source_manifest.parent
    output_root = args.output.expanduser().resolve()
    ensure_new_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    model_dir = Path(source_payload["colmap_model"])
    image_root = Path(source_payload["images"])
    for required in (
        model_dir / "cameras.txt",
        model_dir / "images.txt",
        model_dir / "points3D.txt",
        source_root / "pairs",
    ):
        if not required.exists():
            raise FileNotFoundError(f"Refusion input not found: {required}")

    np, cv2, _, o3d = _lazy_imports()
    cameras = read_colmap_cameras(model_dir / "cameras.txt")
    images = read_colmap_images(model_dir / "images.txt")
    points = read_colmap_points(model_dir / "points3D.txt")
    views, records = load_saved_depth_views(
        source_root,
        source_payload,
        cameras,
        images,
        image_root,
        np,
        cv2,
    )
    voxel_size = args.voxel_size or float(source_payload["voxel_size"])
    for index, view in enumerate(views, start=1):
        view.normal_world = estimate_world_normals_from_depth(
            view,
            relative_discontinuity=args.normal_discontinuity_relative,
            absolute_discontinuity=voxel_size
            * args.normal_discontinuity_voxel_multiplier,
            np=np,
        )
        print(f"normal {index}/{len(views)} {view.pair.left.name}", flush=True)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * args.sdf_trunc_multiplier,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    mask_root = output_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    normal_cosine = math.cos(math.radians(args.normal_consistency_angle))
    integrated = 0
    for index, view in enumerate(views, start=1):
        consistent = multiview_consistency_mask(
            view,
            views,
            min_support_views=args.min_consistent_views,
            relative_tolerance=args.consistency_relative_tolerance,
            absolute_tolerance=voxel_size
            * args.consistency_absolute_voxel_multiplier,
            normal_cosine_threshold=normal_cosine,
            np=np,
            cv2=cv2,
        )
        consistent_pixels = int(consistent.sum())
        record = records[view.record_index]
        record["consistent_pixels"] = consistent_pixels
        record["consistent_fraction"] = consistent_pixels / consistent.size
        mask_path = mask_root / f"{index:04d}_{Path(view.pair.left.name).stem}.png"
        cv2.imwrite(str(mask_path), consistent.astype(np.uint8) * 255)
        record["consistent_mask"] = str(mask_path)
        if consistent_pixels < args.min_valid_pixels:
            record["skip_reason"] = "strong_depth_normal_consistency"
            print(
                f"refusion {index}/{len(views)} {view.pair.left.name}: "
                f"rejected {consistent_pixels / consistent.size:.1%}",
                flush=True,
            )
            continue
        filtered_depth = np.where(consistent, view.depth, 0).astype(np.float32)
        color_rgb = cv2.cvtColor(view.pair.left_bgr, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(filtered_depth),
            depth_scale=1.0,
            depth_trunc=float(filtered_depth.max()) + voxel_size,
            convert_rgb_to_intensity=False,
        )
        height, width = filtered_depth.shape
        projection = view.pair.left_projection
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(projection[0, 0]),
            float(projection[1, 1]),
            float(projection[0, 2]),
            float(projection[1, 2]),
        )
        volume.integrate(
            rgbd,
            intrinsic,
            _world_to_rectified_camera(
                view.pair.left, view.pair.left_rectification, np
            ),
        )
        record["integrated"] = True
        integrated += 1
        print(
            f"refusion {index}/{len(views)} {view.pair.left.name}: "
            f"accepted {consistent_pixels / consistent.size:.1%}",
            flush=True,
        )
    if integrated == 0:
        raise RuntimeError("No saved depth view passed strong consistency")

    dense_root = output_root / "dense"
    dense_root.mkdir(parents=True, exist_ok=True)
    cloud = volume.extract_point_cloud()
    cloud_path = dense_root / "foundationstereo_fused_depth_normal_consistent.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False):
        raise RuntimeError(f"Could not write refusion point cloud: {cloud_path}")
    result = {
        "status": "complete",
        "mode": "foundationstereo-saved-depth-normal-refusion",
        "source_run": str(source_manifest),
        "output": str(output_root),
        "input_depth_views": len(views),
        "integrated_views": integrated,
        "voxel_size": voxel_size,
        "min_consistent_views": args.min_consistent_views,
        "consistency_relative_tolerance": args.consistency_relative_tolerance,
        "consistency_absolute_voxel_multiplier": args.consistency_absolute_voxel_multiplier,
        "normal_consistency_angle": args.normal_consistency_angle,
        "normal_discontinuity_relative": args.normal_discontinuity_relative,
        "normal_discontinuity_voxel_multiplier": args.normal_discontinuity_voxel_multiplier,
        "cloud": str(cloud_path),
        "cloud_points": len(cloud.points),
        "cloud_has_colors": cloud.has_colors(),
        "cloud_has_normals": cloud.has_normals(),
        "views": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output_root / "run.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundation-stereo-refusion",
        description="Re-fuse saved FoundationStereo depths with strong depth and normal consistency",
    )
    parser.add_argument("source", type=Path, help="completed FoundationStereo run directory")
    parser.add_argument("output", type=Path, help="new refusion output directory")
    parser.add_argument("--min-consistent-views", type=int, default=3)
    parser.add_argument("--consistency-relative-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--consistency-absolute-voxel-multiplier", type=float, default=1.0
    )
    parser.add_argument("--normal-consistency-angle", type=float, default=25.0)
    parser.add_argument("--normal-discontinuity-relative", type=float, default=0.03)
    parser.add_argument(
        "--normal-discontinuity-voxel-multiplier", type=float, default=2.0
    )
    parser.add_argument("--min-valid-pixels", type=int, default=5000)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--sdf-trunc-multiplier", type=float, default=5.0)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.min_consistent_views < 1:
        raise ValueError("min-consistent-views must be positive")
    if args.consistency_relative_tolerance < 0:
        raise ValueError("consistency-relative-tolerance must not be negative")
    if args.consistency_absolute_voxel_multiplier < 0:
        raise ValueError("consistency-absolute-voxel-multiplier must not be negative")
    if not 0 < args.normal_consistency_angle <= 90:
        raise ValueError("normal-consistency-angle must be in (0, 90]")
    if args.normal_discontinuity_relative < 0:
        raise ValueError("normal-discontinuity-relative must not be negative")
    if args.normal_discontinuity_voxel_multiplier < 0:
        raise ValueError("normal-discontinuity-voxel-multiplier must not be negative")
    if args.min_valid_pixels < 1:
        raise ValueError("min-valid-pixels must be positive")
    if args.voxel_size is not None and args.voxel_size <= 0:
        raise ValueError("voxel-size must be positive")
    if args.sdf_trunc_multiplier <= 0:
        raise ValueError("sdf-trunc-multiplier must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_arguments(args)
        result = run_refusion(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
