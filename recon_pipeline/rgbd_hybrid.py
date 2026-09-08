from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from recon_pipeline.artifacts import ensure_new_output, validate_mesh, write_json
from recon_pipeline.foundation_stereo import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DepthView,
    _world_to_rectified_camera,
    build_parser as build_foundation_parser,
    camera_matrix_and_distortion,
    quaternion_to_rotation,
    read_colmap_cameras,
    read_colmap_images,
    read_colmap_points,
    run_experiment,
    validate_arguments as validate_foundation_arguments,
)
from recon_pipeline.foundation_stereo_refusion import load_saved_depth_views
from recon_pipeline.models import BackendName, PipelineConfig, Target
from recon_pipeline.orchestrator import execute_pipeline
from recon_pipeline.process import CommandRunner
from recon_pipeline.rgbd_fusion import (
    DEFAULT_K4A_BIN,
    DEFAULT_OPENMVS_DIR,
    _prepare_texturing_mesh,
    _rotation_to_quaternion,
    derive_color_calibration,
    extract_k4a_calibration,
    run_openmvs_texturing,
)


def _lazy_imports() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "RGB-D hybrid fusion requires numpy, opencv-python-headless, and open3d"
        ) from exc
    return np, cv2, o3d


def extract_common_rgbd_frames(
    mkv_path: Path,
    output_root: Path,
    k4a_bin: Path,
    *,
    frame_step: int,
    max_frames: int | None,
    source_fps: float,
    jpeg_quality: int,
    depth_min: float,
    depth_max: float,
    undistort_alpha: float,
) -> dict[str, Any]:
    rgb_root = output_root / "frames" / "rgb"
    depth_root = output_root / "frames" / "tof"
    rgb_root.mkdir(parents=True, exist_ok=True)
    depth_root.mkdir(parents=True, exist_ok=True)
    calibration_path = extract_k4a_calibration(
        mkv_path, output_root / "native" / "calibration.json"
    )

    os.environ["K4A_LIB_DIR"] = str(k4a_bin)
    os.environ["PATH"] = str(k4a_bin) + os.pathsep + os.environ.get("PATH", "")
    dll_handle = os.add_dll_directory(str(k4a_bin)) if os.name == "nt" else None
    np, cv2, o3d = _lazy_imports()
    calibration = derive_color_calibration(
        calibration_path, 1920, 1080, alpha=undistort_alpha
    )
    intrinsic = np.asarray(calibration.intrinsic, dtype=np.float64)
    distortion = np.asarray(calibration.distortion, dtype=np.float64)
    undistorted = np.asarray(calibration.undistorted_intrinsic, dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsic,
        distortion,
        None,
        undistorted,
        (calibration.width, calibration.height),
        cv2.CV_32FC1,
    )

    reader = o3d.io.AzureKinectMKVReader()
    if not reader.open(str(mkv_path)):
        raise RuntimeError(f"Open3D could not open MKV: {mkv_path}")
    records: list[dict[str, Any]] = []
    source_frame = -1
    try:
        while not reader.is_eof():
            rgbd = reader.next_frame()
            source_frame += 1
            if rgbd is None or source_frame % frame_step:
                continue
            if max_frames is not None and len(records) >= max_frames:
                break
            color = np.asarray(rgbd.color)
            depth = np.asarray(rgbd.depth)
            if color.shape[:2] != (calibration.height, calibration.width):
                raise RuntimeError(f"Unexpected aligned color shape: {color.shape}")
            if depth.shape != (calibration.height, calibration.width):
                raise RuntimeError(f"Unexpected aligned depth shape: {depth.shape}")
            color = cv2.remap(
                color,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            depth = cv2.remap(
                depth,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            ).astype(np.uint16, copy=False)
            valid = (
                (depth >= round(depth_min * 1000.0))
                & (depth <= round(depth_max * 1000.0))
            )
            depth = np.where(valid, depth, 0).astype(np.uint16)
            stem = f"frame_{source_frame:06d}"
            rgb_path = rgb_root / f"{stem}.jpg"
            depth_path = depth_root / f"{stem}.png"
            if not cv2.imwrite(
                str(rgb_path),
                cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise RuntimeError(f"Could not write RGB frame: {rgb_path}")
            if not cv2.imwrite(str(depth_path), depth):
                raise RuntimeError(f"Could not write ToF depth: {depth_path}")
            valid_depth = depth[depth > 0]
            records.append(
                {
                    "source_frame": source_frame,
                    "timestamp_seconds": source_frame / source_fps,
                    "image": rgb_path.name,
                    "depth": depth_path.name,
                    "valid_depth_pixels": int(len(valid_depth)),
                    "median_depth_m": (
                        float(np.median(valid_depth)) / 1000.0
                        if len(valid_depth)
                        else None
                    ),
                }
            )
            if len(records) % 50 == 0:
                print(
                    f"extract {len(records)} frames, source={source_frame}", flush=True
                )
    finally:
        reader.close()
        if dll_handle is not None:
            dll_handle.close()
    if len(records) < 2:
        raise RuntimeError("Fewer than two synchronized RGB-D frames were extracted")
    payload = {
        "input": str(mkv_path),
        "rgb_root": str(rgb_root),
        "depth_root": str(depth_root),
        "frame_step": frame_step,
        "source_fps": source_fps,
        "depth_range_m": [depth_min, depth_max],
        "calibration": asdict(calibration),
        "frames": records,
    }
    write_json(output_root / "frames" / "frames.json", payload)
    write_json(output_root / "calibration.json", asdict(calibration))
    return payload


def estimate_metric_scale(
    model_dir: Path,
    depth_root: Path,
    *,
    depth_scale: float,
    depth_min: float,
    depth_max: float,
    max_samples_per_image: int,
    source_intrinsic: Any,
    np: Any,
    cv2: Any,
) -> dict[str, Any]:
    cameras = read_colmap_cameras(model_dir / "cameras.txt")
    images = read_colmap_images(model_dir / "images.txt")
    points = read_colmap_points(model_dir / "points3D.txt")
    ratios: list[Any] = []
    frame_medians: list[dict[str, Any]] = []
    for image in sorted(images.values(), key=lambda item: item.name.casefold()):
        depth_path = depth_root / f"{Path(image.name).stem}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = warp_depth_to_camera(
            depth, source_intrinsic, cameras[image.camera_id], np, cv2
        )
        rotation = quaternion_to_rotation(image.quaternion, np)
        translation = np.asarray(image.translation, dtype=np.float64)
        observations = image.observations
        if len(observations) > max_samples_per_image:
            step = max(1, len(observations) // max_samples_per_image)
            observations = observations[::step][:max_samples_per_image]
        frame_ratios: list[float] = []
        for x, y, point_id in observations:
            point = points.get(point_id)
            if point is None:
                continue
            px = int(round(x))
            py = int(round(y))
            if not (1 <= px < depth.shape[1] - 1 and 1 <= py < depth.shape[0] - 1):
                continue
            patch = depth[py - 1 : py + 2, px - 1 : px + 2]
            valid_patch = patch[patch > 0]
            if len(valid_patch) < 3:
                continue
            sensor_depth = float(np.median(valid_patch)) / depth_scale
            if not depth_min <= sensor_depth <= depth_max:
                continue
            camera_point = rotation @ np.asarray(point, dtype=np.float64) + translation
            sfm_depth = float(camera_point[2])
            if sfm_depth <= 0 or not math.isfinite(sfm_depth):
                continue
            ratio = sensor_depth / sfm_depth
            if 1e-6 < ratio < 1e6 and math.isfinite(ratio):
                frame_ratios.append(ratio)
        if len(frame_ratios) >= 5:
            values = np.asarray(frame_ratios, dtype=np.float64)
            ratios.append(values)
            frame_medians.append(
                {
                    "image": image.name,
                    "samples": int(len(values)),
                    "scale": float(np.exp(np.median(np.log(values)))),
                }
            )
    if not ratios:
        raise RuntimeError("No SfM observations had valid synchronized ToF depth")
    values = np.concatenate(ratios)
    if len(values) < 100:
        raise RuntimeError(f"Too few metric scale samples: {len(values)}")
    logs = np.log(values)
    center = float(np.median(logs))
    mad = float(np.median(np.abs(logs - center)))
    limit = max(0.03, 4.0 * 1.4826 * mad)
    inliers = np.abs(logs - center) <= limit
    scale = float(np.exp(np.median(logs[inliers])))
    return {
        "scale": scale,
        "samples": int(len(values)),
        "inliers": int(inliers.sum()),
        "log_mad": mad,
        "p10": float(np.percentile(values[inliers], 10.0)),
        "p50": float(np.percentile(values[inliers], 50.0)),
        "p90": float(np.percentile(values[inliers], 90.0)),
        "frames": frame_medians,
        "camera_records": len(cameras),
        "registered_images": len(images),
    }


def write_metric_colmap_model(
    source_model: Path,
    source_images: Path,
    destination_root: Path,
    scale: float,
    *,
    intrinsic: Any | None = None,
    translations: dict[int, Sequence[float]] | None = None,
    poses: dict[int, tuple[Sequence[float], Sequence[float]]] | None = None,
) -> Path:
    model = destination_root / "sparse" / "0"
    images_root = destination_root / "images"
    model.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    if intrinsic is None:
        shutil.copy2(source_model / "cameras.txt", model / "cameras.txt")
    else:
        camera_lines: list[str] = []
        fx = float(intrinsic[0][0])
        fy = float(intrinsic[1][1])
        cx = float(intrinsic[0][2])
        cy = float(intrinsic[1][2])
        for line in (source_model / "cameras.txt").read_text(
            encoding="utf-8", errors="strict"
        ).splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                camera_lines.append(line)
                continue
            fields = line.split()
            camera_lines.append(
                f"{fields[0]} PINHOLE {fields[2]} {fields[3]} "
                f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}"
            )
        (model / "cameras.txt").write_text(
            "\n".join(camera_lines) + "\n", encoding="utf-8", newline="\n"
        )

    source_lines = (source_model / "images.txt").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()
    result: list[str] = []
    expect_observations = False
    image_names: list[str] = []
    for line in source_lines:
        if expect_observations:
            result.append(line)
            expect_observations = False
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            result.append(line)
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise RuntimeError(f"Invalid COLMAP image record: {line}")
        image_id = int(fields[0])
        if poses is not None and image_id in poses:
            quaternion, translation = poses[image_id]
            for index, value in zip((1, 2, 3, 4), quaternion):
                fields[index] = f"{float(value):.17g}"
            for index, value in zip((5, 6, 7), translation):
                fields[index] = f"{float(value):.17g}"
        elif translations is not None and image_id in translations:
            for index, value in zip((5, 6, 7), translations[image_id]):
                fields[index] = f"{float(value):.17g}"
        else:
            for index in (5, 6, 7):
                fields[index] = f"{float(fields[index]) * scale:.17g}"
        result.append(" ".join(fields))
        image_names.append(fields[9])
        expect_observations = True
    (model / "images.txt").write_text(
        "\n".join(result) + "\n", encoding="utf-8", newline="\n"
    )

    point_lines: list[str] = []
    for line in (source_model / "points3D.txt").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            point_lines.append(line)
            continue
        fields = line.split()
        for index in (1, 2, 3):
            fields[index] = f"{float(fields[index]) * scale:.17g}"
        point_lines.append(" ".join(fields))
    (model / "points3D.txt").write_text(
        "\n".join(point_lines) + "\n", encoding="utf-8", newline="\n"
    )

    for name in image_names:
        source = source_images / name
        target = images_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return model


def warp_depth_to_camera(
    depth: Any,
    source_intrinsic: Any,
    target_camera: Any,
    np: Any,
    cv2: Any,
) -> Any:
    target_intrinsic, target_distortion = camera_matrix_and_distortion(
        target_camera, np
    )
    if np.any(np.abs(target_distortion) > 1e-12):
        raise ValueError("Hybrid depth warping requires a distortion-free target camera")
    homography = target_intrinsic @ np.linalg.inv(
        np.asarray(source_intrinsic, dtype=np.float64)
    )
    return cv2.warpPerspective(
        depth,
        homography,
        (target_camera.width, target_camera.height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )


def refine_metric_translations(
    model_dir: Path,
    depth_root: Path,
    *,
    scale: float,
    source_intrinsic: Any,
    depth_scale: float,
    depth_min: float,
    depth_max: float,
    max_samples_per_image: int,
    max_correction: float,
    np: Any,
    cv2: Any,
) -> tuple[dict[int, tuple[float, float, float]], dict[str, Any]]:
    images = read_colmap_images(model_dir / "images.txt")
    points = read_colmap_points(model_dir / "points3D.txt")
    cameras = read_colmap_cameras(model_dir / "cameras.txt")
    translations: dict[int, tuple[float, float, float]] = {}
    records: list[dict[str, Any]] = []
    corrections: list[float] = []
    for image in sorted(images.values(), key=lambda item: item.name.casefold()):
        depth = cv2.imread(
            str(depth_root / f"{Path(image.name).stem}.png"), cv2.IMREAD_UNCHANGED
        )
        if depth is None:
            continue
        camera = cameras[image.camera_id]
        depth = warp_depth_to_camera(
            depth, source_intrinsic, camera, np, cv2
        )
        intrinsic_array, _ = camera_matrix_and_distortion(camera, np)
        fx = float(intrinsic_array[0, 0])
        fy = float(intrinsic_array[1, 1])
        cx = float(intrinsic_array[0, 2])
        cy = float(intrinsic_array[1, 2])
        observations = image.observations
        if len(observations) > max_samples_per_image:
            step = max(1, len(observations) // max_samples_per_image)
            observations = observations[::step][:max_samples_per_image]
        rotation = quaternion_to_rotation(image.quaternion, np)
        candidates: list[Any] = []
        for x, y, point_id in observations:
            point = points.get(point_id)
            if point is None:
                continue
            px = int(round(x))
            py = int(round(y))
            if not (1 <= px < depth.shape[1] - 1 and 1 <= py < depth.shape[0] - 1):
                continue
            patch = depth[py - 1 : py + 2, px - 1 : px + 2]
            valid_patch = patch[patch > 0]
            if len(valid_patch) < 3:
                continue
            z = float(np.median(valid_patch)) / depth_scale
            if not depth_min <= z <= depth_max:
                continue
            sensor = np.asarray([(x - cx) * z / fx, (y - cy) * z / fy, z])
            rotated = rotation @ (np.asarray(point, dtype=np.float64) * scale)
            candidates.append(sensor - rotated)
        if len(candidates) < 10:
            continue
        values = np.asarray(candidates, dtype=np.float64)
        translation = np.median(values, axis=0)
        residuals = np.linalg.norm(values - translation, axis=1)
        residual_median = float(np.median(residuals))
        residual_mad = float(np.median(np.abs(residuals - residual_median)))
        limit = residual_median + max(0.005, 3.0 * 1.4826 * residual_mad)
        inliers = residuals <= limit
        translation = np.median(values[inliers], axis=0)
        original = np.asarray(image.translation, dtype=np.float64) * scale
        correction = float(np.linalg.norm(translation - original))
        accepted = int(inliers.sum()) >= 10 and correction <= max_correction
        if accepted:
            translations[image.image_id] = tuple(float(value) for value in translation)
            corrections.append(correction)
        records.append(
            {
                "image": image.name,
                "samples": len(values),
                "inliers": int(inliers.sum()),
                "correction_m": correction,
                "residual_median_m": residual_median,
                "accepted": accepted,
            }
        )
    return translations, {
        "refined_images": len(translations),
        "candidate_images": len(records),
        "correction_median_m": (
            float(np.median(corrections)) if corrections else None
        ),
        "correction_p90_m": (
            float(np.percentile(corrections, 90.0)) if corrections else None
        ),
        "records": records,
    }


def _target_point_cloud(
    image: Any,
    camera: Any,
    image_root: Path,
    depth_root: Path,
    source_intrinsic: Any,
    args: argparse.Namespace,
    np: Any,
    cv2: Any,
    o3d: Any,
) -> Any | None:
    color = cv2.imread(str(image_root / image.name), cv2.IMREAD_COLOR)
    depth = cv2.imread(
        str(depth_root / f"{Path(image.name).stem}.png"), cv2.IMREAD_UNCHANGED
    )
    if color is None or depth is None:
        return None
    depth = warp_depth_to_camera(depth, source_intrinsic, camera, np, cv2)
    color = cv2.resize(
        color, (args.icp_width, args.icp_height), interpolation=cv2.INTER_AREA
    )
    depth = cv2.resize(
        depth, (args.icp_width, args.icp_height), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32) / args.depth_scale
    depth = filter_tof_depth(
        depth,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        edge_absolute=args.tof_edge_absolute,
        edge_relative=args.tof_edge_relative,
        np=np,
    )
    depth, _ = mask_depth_to_screen_target(
        depth,
        color_bgr=color if args.skin_seed else None,
        seed_x_fraction=args.target_seed_x_fraction,
        seed_y_fraction=args.target_seed_y_fraction,
        seed_fraction=args.target_seed_fraction,
        radius_x_fraction=args.target_screen_radius_x,
        radius_y_fraction=args.target_screen_radius_y,
        depth_band=args.target_depth_band,
        np=np,
        cv2=cv2,
    )
    if args.grabcut_mask and np.any(depth > 0):
        foreground = grabcut_foreground_mask(
            color,
            x_min_fraction=args.grabcut_x_min,
            x_max_fraction=args.grabcut_x_max,
            y_min_fraction=args.grabcut_y_min,
            y_max_fraction=args.grabcut_y_max,
            iterations=args.grabcut_iterations,
            np=np,
            cv2=cv2,
        )
        depth[~foreground] = 0.0
    rows, columns = np.nonzero(depth > 0)
    if len(rows) < args.icp_min_points:
        return None
    z = depth[rows, columns].astype(np.float64)
    intrinsic, _ = camera_matrix_and_distortion(camera, np)
    scale_x = args.icp_width / camera.width
    scale_y = args.icp_height / camera.height
    fx = float(intrinsic[0, 0] * scale_x)
    fy = float(intrinsic[1, 1] * scale_y)
    cx = float(intrinsic[0, 2] * scale_x)
    cy = float(intrinsic[1, 2] * scale_y)
    points = np.column_stack(
        ((columns - cx) * z / fx, (rows - cy) * z / fy, z)
    )
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(args.icp_voxel_size)
    if len(cloud.points) < args.icp_min_points:
        return None
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.icp_voxel_size * 4.0, max_nn=30
        )
    )
    return cloud


def refine_metric_poses_with_tof_icp(
    model_dir: Path,
    image_root: Path,
    depth_root: Path,
    source_intrinsic: Any,
    args: argparse.Namespace,
    np: Any,
    cv2: Any,
    o3d: Any,
) -> tuple[
    dict[int, tuple[tuple[float, ...], tuple[float, ...]]], dict[str, Any]
]:
    cameras = read_colmap_cameras(model_dir / "cameras.txt")
    images = sorted(
        read_colmap_images(model_dir / "images.txt").values(),
        key=lambda item: int(Path(item.name).stem.split("_")[-1]),
    )
    poses: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    records: list[dict[str, Any]] = []
    previous_cloud = None
    previous_camera_to_world = None
    previous_original_camera_to_world = None
    previous_frame = None
    anchors: list[tuple[Any, Any, Any, int]] = []
    for index, image in enumerate(images, start=1):
        frame = int(Path(image.name).stem.split("_")[-1])
        original_world_to_camera = _world_to_rectified_camera(
            image, np.identity(3), np
        )
        original_camera_to_world = np.linalg.inv(original_world_to_camera)
        cloud = _target_point_cloud(
            image,
            cameras[image.camera_id],
            image_root,
            depth_root,
            source_intrinsic,
            args,
            np,
            cv2,
            o3d,
        )
        refined_camera_to_world = original_camera_to_world
        accepted = False
        fitness = None
        rmse = None
        contiguous = (
            cloud is not None
            and previous_cloud is not None
            and previous_frame is not None
            and frame - previous_frame <= args.icp_max_frame_gap
        )
        reference_cloud = previous_cloud if contiguous else None
        reference_camera_to_world = previous_camera_to_world if contiguous else None
        reference_original_camera_to_world = (
            previous_original_camera_to_world if contiguous else None
        )
        edge_type = "sequential" if contiguous else None
        if cloud is not None and not contiguous and args.icp_loop_closure:
            current_center = original_camera_to_world[:3, 3]
            current_view = original_camera_to_world[:3, 2]
            candidates: list[tuple[float, tuple[Any, Any, Any, int]]] = []
            for anchor in anchors:
                _, _, anchor_original, _ = anchor
                center_distance = float(
                    np.linalg.norm(anchor_original[:3, 3] - current_center)
                )
                cosine = float(
                    np.clip(
                        np.dot(anchor_original[:3, 2], current_view), -1.0, 1.0
                    )
                )
                view_angle = float(np.degrees(np.arccos(cosine)))
                if (
                    center_distance <= args.icp_loop_max_center_distance
                    and view_angle <= args.icp_loop_max_view_angle
                ):
                    candidates.append(
                        (center_distance + 0.002 * view_angle, anchor)
                    )
            if candidates:
                _, selected_anchor = min(candidates, key=lambda item: item[0])
                (
                    reference_cloud,
                    reference_camera_to_world,
                    reference_original_camera_to_world,
                    _,
                ) = selected_anchor
                edge_type = "loop"
        if reference_cloud is not None:
            initial = (
                np.linalg.inv(reference_original_camera_to_world)
                @ original_camera_to_world
            )
            registration = o3d.pipelines.registration.registration_icp(
                cloud,
                reference_cloud,
                args.icp_max_correspondence,
                initial,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )
            fitness = float(registration.fitness)
            rmse = float(registration.inlier_rmse)
            candidate = reference_camera_to_world @ registration.transformation
            center_delta = float(
                np.linalg.norm(candidate[:3, 3] - original_camera_to_world[:3, 3])
            )
            rotation_delta = candidate[:3, :3] @ original_camera_to_world[:3, :3].T
            rotation_cosine = np.clip(
                (np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0
            )
            rotation_degrees = float(np.degrees(np.arccos(rotation_cosine)))
            accepted = (
                fitness >= args.icp_min_fitness
                and rmse <= args.icp_max_rmse
                and center_delta <= args.icp_max_pose_correction
                and rotation_degrees <= args.icp_max_rotation_correction
            )
            if accepted:
                prior = args.icp_realityscan_prior
                blended = np.identity(4, dtype=np.float64)
                blended[:3, 3] = (
                    (1.0 - prior) * candidate[:3, 3]
                    + prior * original_camera_to_world[:3, 3]
                )
                matrix = (
                    (1.0 - prior) * candidate[:3, :3]
                    + prior * original_camera_to_world[:3, :3]
                )
                u, _, vt = np.linalg.svd(matrix)
                blended[:3, :3] = u @ vt
                if np.linalg.det(blended[:3, :3]) < 0:
                    u[:, -1] *= -1
                    blended[:3, :3] = u @ vt
                refined_camera_to_world = blended
        refined_world_to_camera = np.linalg.inv(refined_camera_to_world)
        quaternion = _rotation_to_quaternion(
            refined_world_to_camera[:3, :3], np
        )
        translation = tuple(
            float(value) for value in refined_world_to_camera[:3, 3]
        )
        poses[image.image_id] = (quaternion, translation)
        records.append(
            {
                "image": image.name,
                "contiguous": contiguous,
                "edge_type": edge_type,
                "accepted": accepted,
                "fitness": fitness,
                "rmse": rmse,
            }
        )
        if cloud is not None:
            anchors.append(
                (
                    cloud,
                    refined_camera_to_world,
                    original_camera_to_world,
                    frame,
                )
            )
            previous_cloud = cloud
            previous_camera_to_world = refined_camera_to_world
            previous_original_camera_to_world = original_camera_to_world
            previous_frame = frame
        else:
            previous_cloud = None
            previous_camera_to_world = None
            previous_original_camera_to_world = None
            previous_frame = None
        if index % 25 == 0:
            print(f"ToF ICP {index}/{len(images)}", flush=True)
    return poses, {
        "images": len(images),
        "accepted_edges": sum(record["accepted"] for record in records),
        "accepted_loop_edges": sum(
            record["accepted"] and record["edge_type"] == "loop"
            for record in records
        ),
        "records": records,
    }


def project_rectified_depth_to_camera(
    view: DepthView,
    camera: Any,
    output_width: int,
    output_height: int,
    np: Any,
) -> Any:
    rows, columns = np.nonzero(view.valid & np.isfinite(view.depth) & (view.depth > 0))
    output = np.zeros((output_height, output_width), dtype=np.float32)
    if len(rows) == 0:
        return output
    depth = view.depth[rows, columns].astype(np.float64)
    projection = view.pair.left_projection
    rectified = np.column_stack(
        (
            (columns - projection[0, 2]) * depth / projection[0, 0],
            (rows - projection[1, 2]) * depth / projection[1, 1],
            depth,
        )
    )
    camera_points = rectified @ view.pair.left_rectification
    positive = camera_points[:, 2] > 0
    camera_points = camera_points[positive]
    if len(camera_points) == 0:
        return output
    intrinsic, _ = camera_matrix_and_distortion(camera, np)
    scale_x = output_width / camera.width
    scale_y = output_height / camera.height
    pixel_x = np.rint(
        (intrinsic[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsic[0, 2])
        * scale_x
    ).astype(np.int64)
    pixel_y = np.rint(
        (intrinsic[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsic[1, 2])
        * scale_y
    ).astype(np.int64)
    inside = (
        (pixel_x >= 0)
        & (pixel_x < output_width)
        & (pixel_y >= 0)
        & (pixel_y < output_height)
    )
    indices = pixel_y[inside] * output_width + pixel_x[inside]
    z = camera_points[inside, 2].astype(np.float32)
    flat = np.full(output_width * output_height, np.inf, dtype=np.float32)
    np.minimum.at(flat, indices, z)
    flat[~np.isfinite(flat)] = 0.0
    return flat.reshape(output_height, output_width)


def filter_tof_depth(
    depth: Any,
    *,
    depth_min: float,
    depth_max: float,
    edge_absolute: float,
    edge_relative: float,
    np: Any,
) -> Any:
    result = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(result) & (result >= depth_min) & (result <= depth_max)
    edge = np.zeros_like(valid)
    horizontal = valid[:, 1:] & valid[:, :-1]
    tolerance = edge_absolute + edge_relative * np.minimum(
        result[:, 1:], result[:, :-1]
    )
    jump = horizontal & (np.abs(result[:, 1:] - result[:, :-1]) > tolerance)
    edge[:, 1:] |= jump
    edge[:, :-1] |= jump
    vertical = valid[1:, :] & valid[:-1, :]
    tolerance = edge_absolute + edge_relative * np.minimum(
        result[1:, :], result[:-1, :]
    )
    jump = vertical & (np.abs(result[1:, :] - result[:-1, :]) > tolerance)
    edge[1:, :] |= jump
    edge[:-1, :] |= jump
    result[~valid | edge] = 0.0
    return result


def arbitrate_depth(
    tof_depth: Any,
    stereo_depths: Sequence[Any],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    anchor_radius: int,
    np: Any,
    cv2: Any,
) -> tuple[Any, dict[str, int]]:
    tof = np.asarray(tof_depth, dtype=np.float32)
    tof_valid = np.isfinite(tof) & (tof > 0)
    result = np.where(tof_valid, tof, 0).astype(np.float32)
    if not stereo_depths:
        return result, {
            "tof_pixels": int(tof_valid.sum()),
            "stereo_fill_pixels": 0,
            "stereo_conflict_pixels": 0,
        }
    stack = np.stack(
        [np.where(np.asarray(item) > 0, item, np.nan) for item in stereo_depths]
    ).astype(np.float32)
    count = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        candidate = np.nanmedian(stack, axis=0)
        minimum = np.nanmin(stack, axis=0)
        maximum = np.nanmax(stack, axis=0)
    candidate = np.where(np.isfinite(candidate), candidate, 0).astype(np.float32)
    tolerance = absolute_tolerance + relative_tolerance * candidate
    agreement = (count == 1) | ((maximum - minimum) <= tolerance)

    diameter = anchor_radius * 2 + 1
    kernel = np.ones((diameter, diameter), dtype=np.uint8)
    local_has_tof = cv2.dilate(tof_valid.astype(np.uint8), kernel) > 0
    large = np.float32(1e6)
    local_min = cv2.erode(np.where(tof_valid, tof, large), kernel)
    local_max = cv2.dilate(np.where(tof_valid, tof, 0), kernel)
    anchored = (
        local_has_tof
        & (candidate >= local_min - tolerance)
        & (candidate <= local_max + tolerance)
    )
    fill = (~tof_valid) & (count > 0) & agreement & anchored
    result[fill] = candidate[fill]
    conflict = (count > 1) & ~agreement
    return result, {
        "tof_pixels": int(tof_valid.sum()),
        "stereo_fill_pixels": int(fill.sum()),
        "stereo_conflict_pixels": int(conflict.sum()),
    }


def estimate_target_center(
    images: dict[int, Any],
    cameras: dict[int, Any],
    depth_root: Path,
    *,
    output_width: int,
    output_height: int,
    depth_scale: float,
    depth_min: float,
    depth_max: float,
    seed_fraction: float,
    np: Any,
    cv2: Any,
) -> tuple[Any, dict[str, Any]]:
    centers: list[Any] = []
    for image in sorted(images.values(), key=lambda item: item.name.casefold()):
        depth_path = depth_root / f"{Path(image.name).stem}.png"
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            continue
        depth = cv2.resize(
            depth_mm,
            (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float64) / depth_scale
        half_width = max(2, round(output_width * seed_fraction * 0.5))
        half_height = max(2, round(output_height * seed_fraction * 0.5))
        center_x = output_width // 2
        center_y = output_height // 2
        patch = depth[
            center_y - half_height : center_y + half_height + 1,
            center_x - half_width : center_x + half_width + 1,
        ]
        valid = patch[(patch >= depth_min) & (patch <= depth_max)]
        if len(valid) < 25:
            continue
        z = float(np.median(valid))
        camera = cameras[image.camera_id]
        intrinsic, _ = camera_matrix_and_distortion(camera, np)
        scale_x = output_width / camera.width
        scale_y = output_height / camera.height
        fx = float(intrinsic[0, 0] * scale_x)
        fy = float(intrinsic[1, 1] * scale_y)
        cx = float(intrinsic[0, 2] * scale_x)
        cy = float(intrinsic[1, 2] * scale_y)
        camera_point = np.asarray(
            [(center_x - cx) * z / fx, (center_y - cy) * z / fy, z],
            dtype=np.float64,
        )
        rotation = quaternion_to_rotation(image.quaternion, np)
        translation = np.asarray(image.translation, dtype=np.float64)
        centers.append((camera_point - translation) @ rotation)
    if len(centers) < 3:
        raise RuntimeError("Could not estimate a target center from central ToF samples")
    values = np.asarray(centers, dtype=np.float64)
    initial = np.median(values, axis=0)
    distances = np.linalg.norm(values - initial, axis=1)
    distance_median = float(np.median(distances))
    distance_mad = float(np.median(np.abs(distances - distance_median)))
    limit = distance_median + max(0.03, 3.0 * 1.4826 * distance_mad)
    inliers = distances <= limit
    center = np.median(values[inliers], axis=0)
    return center, {
        "center_world": [float(value) for value in center],
        "samples": len(centers),
        "inliers": int(inliers.sum()),
        "sample_distance_median": distance_median,
        "sample_distance_p90": float(np.percentile(distances[inliers], 90.0)),
    }


def mask_depth_to_world_sphere(
    depth: Any,
    image: Any,
    camera: Any,
    center_world: Any,
    radius: float,
    np: Any,
) -> Any:
    result = np.asarray(depth, dtype=np.float32).copy()
    rows, columns = np.nonzero(np.isfinite(result) & (result > 0))
    if len(rows) == 0:
        return result
    z = result[rows, columns].astype(np.float64)
    intrinsic, _ = camera_matrix_and_distortion(camera, np)
    scale_x = result.shape[1] / camera.width
    scale_y = result.shape[0] / camera.height
    fx = float(intrinsic[0, 0] * scale_x)
    fy = float(intrinsic[1, 1] * scale_y)
    cx = float(intrinsic[0, 2] * scale_x)
    cy = float(intrinsic[1, 2] * scale_y)
    camera_points = np.column_stack(
        ((columns - cx) * z / fx, (rows - cy) * z / fy, z)
    )
    rotation = quaternion_to_rotation(image.quaternion, np)
    translation = np.asarray(image.translation, dtype=np.float64)
    world = (camera_points - translation) @ rotation
    inside = np.linalg.norm(world - np.asarray(center_world), axis=1) <= radius
    rejected_rows = rows[~inside]
    rejected_columns = columns[~inside]
    result[rejected_rows, rejected_columns] = 0.0
    return result


def mask_depth_to_screen_target(
    depth: Any,
    *,
    color_bgr: Any | None = None,
    seed_x_fraction: float,
    seed_y_fraction: float,
    seed_fraction: float,
    radius_x_fraction: float,
    radius_y_fraction: float,
    depth_band: float,
    np: Any,
    cv2: Any,
) -> tuple[Any, dict[str, Any]]:
    source = np.asarray(depth, dtype=np.float32)
    height, width = source.shape
    seed_x = int(round((width - 1) * seed_x_fraction))
    seed_y = int(round((height - 1) * seed_y_fraction))
    half_width = max(2, round(width * seed_fraction * 0.5))
    half_height = max(2, round(height * seed_fraction * 0.5))
    x0 = max(0, seed_x - half_width)
    x1 = min(width, seed_x + half_width + 1)
    y0 = max(0, seed_y - half_height)
    y1 = min(height, seed_y + half_height + 1)
    patch = source[y0:y1, x0:x1]
    valid_patch = patch[patch > 0]
    skin_seed_pixels = 0
    skin_seed_depth: float | None = None
    if color_bgr is not None:
        ycrcb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2YCrCb)
        skin = (
            (ycrcb[..., 0] >= 50)
            & (ycrcb[..., 1] >= 128)
            & (ycrcb[..., 1] <= 180)
            & (ycrcb[..., 2] >= 75)
            & (ycrcb[..., 2] <= 140)
        )
        search_half_width = max(half_width, round(width * radius_x_fraction))
        search_half_height = max(half_height, round(height * radius_y_fraction))
        sx0 = max(0, seed_x - search_half_width)
        sx1 = min(width, seed_x + search_half_width + 1)
        sy0 = max(0, seed_y - search_half_height)
        sy1 = min(height, seed_y + search_half_height + 1)
        skin_depth = source[sy0:sy1, sx0:sx1]
        skin_valid = skin[sy0:sy1, sx0:sx1] & (skin_depth > 0)
        skin_values = skin_depth[skin_valid]
        if len(skin_values) >= 20:
            skin_seed_depth = float(np.percentile(skin_values, 20.0))
            skin_seed_pixels = int(len(skin_values))
    if len(valid_patch) < 10:
        return np.zeros_like(source), {
            "seed_depth": None,
            "skin_seed_pixels": skin_seed_pixels,
            "target_pixels": 0,
        }
    seed_depth = (
        skin_seed_depth
        if skin_seed_depth is not None
        else float(np.median(valid_patch))
    )
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    radius_x = max(1.0, width * radius_x_fraction)
    radius_y = max(1.0, height * radius_y_fraction)
    ellipse = (
        ((columns - seed_x) / radius_x) ** 2
        + ((rows - seed_y) / radius_y) ** 2
        <= 1.0
    )
    candidate = (
        (source > 0)
        & (np.abs(source - seed_depth) <= depth_band)
        & ellipse
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(source), {
            "seed_depth": seed_depth,
            "skin_seed_pixels": skin_seed_pixels,
            "target_pixels": 0,
        }
    seed_labels = labels[y0:y1, x0:x1]
    seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels):
        label_counts = np.bincount(seed_labels)
        selected = int(np.argmax(label_counts))
    else:
        selected = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    target = labels == selected
    result = np.where(target, source, 0).astype(np.float32)
    return result, {
        "seed_depth": seed_depth,
        "skin_seed_pixels": skin_seed_pixels,
        "target_pixels": int(target.sum()),
    }


def grabcut_foreground_mask(
    color_bgr: Any,
    *,
    x_min_fraction: float,
    x_max_fraction: float,
    y_min_fraction: float,
    y_max_fraction: float,
    iterations: int,
    np: Any,
    cv2: Any,
) -> Any:
    height, width = color_bgr.shape[:2]
    x0 = max(1, int(round(width * x_min_fraction)))
    x1 = min(width - 1, int(round(width * x_max_fraction)))
    y0 = max(1, int(round(height * y_min_fraction)))
    y1 = min(height - 1, int(round(height * y_max_fraction)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid GrabCut rectangle")
    mask = np.zeros((height, width), dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        color_bgr,
        mask,
        (x0, y0, x1 - x0, y1 - y0),
        background,
        foreground,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def select_mesh_component(
    mesh: Any,
    *,
    mode: str,
    max_extent: float,
    np: Any,
) -> tuple[Any, dict[str, Any]]:
    if mode == "none" or not len(mesh.triangles):
        return mesh, {"mode": mode, "selected": None, "components": 0}
    clusters, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(clusters)
    count_array = np.asarray(counts)
    selected = int(np.argmax(count_array))
    records: list[dict[str, Any]] = []
    if mode == "skin-compact" and mesh.has_vertex_colors():
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
        candidate_ids = np.argsort(count_array)[-min(50, len(count_array)) :]
        best_score = -1.0
        for component_id in candidate_ids:
            triangle_mask = labels == component_id
            vertex_ids = np.unique(triangles[triangle_mask].reshape(-1))
            component_vertices = vertices[vertex_ids]
            component_colors = colors[vertex_ids]
            extent = component_vertices.max(axis=0) - component_vertices.min(axis=0)
            maximum_extent = float(extent.max())
            skin = (
                (component_colors[:, 0] >= 0.45)
                & (component_colors[:, 1] >= 0.30)
                & (component_colors[:, 2] >= 0.20)
                & (component_colors[:, 0] >= component_colors[:, 1] * 1.02)
                & (component_colors[:, 1] >= component_colors[:, 2] * 0.95)
            )
            skin_vertices = int(skin.sum())
            compact = maximum_extent <= max_extent
            score = float(skin_vertices) / max(0.02, maximum_extent) if compact else -1.0
            records.append(
                {
                    "component": int(component_id),
                    "triangles": int(count_array[component_id]),
                    "vertices": int(len(vertex_ids)),
                    "max_extent_m": maximum_extent,
                    "skin_vertices": skin_vertices,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                selected = int(component_id)
    mesh.remove_triangles_by_mask(labels != selected)
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "mode": mode,
        "selected": selected,
        "components": int(len(count_array)),
        "selected_triangles": int(count_array[selected]),
        "candidates": sorted(records, key=lambda item: item["score"], reverse=True),
    }


def fuse_tof_anchored_depths(
    model_dir: Path,
    image_root: Path,
    depth_root: Path,
    foundation_root: Path | None,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    np, cv2, o3d = _lazy_imports()
    cameras = read_colmap_cameras(model_dir / "cameras.txt")
    images = read_colmap_images(model_dir / "images.txt")
    points = read_colmap_points(model_dir / "points3D.txt")
    calibration = json.loads(
        (args.output.expanduser().resolve() / "calibration.json").read_text(
            encoding="utf-8"
        )
    )
    source_intrinsic = calibration["undistorted_intrinsic"]
    views: list[DepthView] = []
    if foundation_root is not None:
        source_payload = json.loads(
            (foundation_root / "run.json").read_text(encoding="utf-8")
        )
        views, _ = load_saved_depth_views(
            foundation_root,
            source_payload,
            cameras,
            images,
            image_root,
            np,
            cv2,
        )
    stereo_by_reference: dict[str, list[Any]] = {}
    for index, view in enumerate(views, start=1):
        camera = cameras[view.pair.left.camera_id]
        projected = project_rectified_depth_to_camera(
            view, camera, args.fusion_width, args.fusion_height, np
        )
        stereo_by_reference.setdefault(view.pair.left.name.casefold(), []).append(projected)
        if index % 25 == 0:
            print(f"unrectify stereo {index}/{len(views)}", flush=True)

    target_center = None
    target_info: dict[str, Any] = {"radius": args.target_radius}
    if args.target_radius > 0:
        target_center, target_info = estimate_target_center(
            images,
            cameras,
            depth_root,
            output_width=args.fusion_width,
            output_height=args.fusion_height,
            depth_scale=args.depth_scale,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            seed_fraction=args.target_seed_fraction,
            np=np,
            cv2=cv2,
        )
        target_info["radius"] = args.target_radius
        print(
            "target center "
            + ", ".join(f"{float(value):.4f}" for value in target_center),
            flush=True,
        )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    fused_depth_root = output_root / "fused_depth"
    fused_depth_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    integrated = 0
    for image in sorted(images.values(), key=lambda item: item.name.casefold()):
        depth_path = depth_root / f"{Path(image.name).stem}.png"
        color_path = image_root / image.name
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if depth_mm is None or color_bgr is None:
            continue
        camera = cameras[image.camera_id]
        depth_mm = warp_depth_to_camera(
            depth_mm, source_intrinsic, camera, np, cv2
        )
        color_bgr = cv2.resize(
            color_bgr,
            (args.fusion_width, args.fusion_height),
            interpolation=cv2.INTER_AREA,
        )
        tof = cv2.resize(
            depth_mm,
            (args.fusion_width, args.fusion_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32) / args.depth_scale
        tof = filter_tof_depth(
            tof,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            edge_absolute=args.tof_edge_absolute,
            edge_relative=args.tof_edge_relative,
            np=np,
        )
        tof, screen_target = mask_depth_to_screen_target(
            tof,
            color_bgr=color_bgr if args.skin_seed else None,
            seed_x_fraction=args.target_seed_x_fraction,
            seed_y_fraction=args.target_seed_y_fraction,
            seed_fraction=args.target_seed_fraction,
            radius_x_fraction=args.target_screen_radius_x,
            radius_y_fraction=args.target_screen_radius_y,
            depth_band=args.target_depth_band,
            np=np,
            cv2=cv2,
        )
        if args.grabcut_mask and np.any(tof > 0):
            foreground = grabcut_foreground_mask(
                color_bgr,
                x_min_fraction=args.grabcut_x_min,
                x_max_fraction=args.grabcut_x_max,
                y_min_fraction=args.grabcut_y_min,
                y_max_fraction=args.grabcut_y_max,
                iterations=args.grabcut_iterations,
                np=np,
                cv2=cv2,
            )
            tof[~foreground] = 0.0
            screen_target["grabcut_pixels"] = int(foreground.sum())
        final_depth, stats = arbitrate_depth(
            tof,
            stereo_by_reference.get(image.name.casefold(), ()),
            absolute_tolerance=args.stereo_tof_absolute_tolerance,
            relative_tolerance=args.stereo_tof_relative_tolerance,
            anchor_radius=args.stereo_anchor_radius,
            np=np,
            cv2=cv2,
        )
        if target_center is not None:
            final_depth = mask_depth_to_world_sphere(
                final_depth,
                image,
                cameras[image.camera_id],
                target_center,
                args.target_radius,
                np,
            )
        valid_pixels = int((final_depth > 0).sum())
        if valid_pixels < args.min_fusion_pixels:
            records.append(
                {"image": image.name, **screen_target, **stats, "integrated": False}
            )
            continue
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(final_depth),
            depth_scale=1.0,
            depth_trunc=args.depth_max,
            convert_rgb_to_intensity=False,
        )
        intrinsic, _ = camera_matrix_and_distortion(camera, np)
        scale_x = args.fusion_width / camera.width
        scale_y = args.fusion_height / camera.height
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            args.fusion_width,
            args.fusion_height,
            float(intrinsic[0, 0] * scale_x),
            float(intrinsic[1, 1] * scale_y),
            float(intrinsic[0, 2] * scale_x),
            float(intrinsic[1, 2] * scale_y),
        )
        identity_rectification = np.identity(3, dtype=np.float64)
        volume.integrate(
            rgbd,
            pinhole,
            _world_to_rectified_camera(image, identity_rectification, np),
        )
        if stats["stereo_fill_pixels"]:
            output_depth = np.rint(final_depth * args.depth_scale).astype(np.uint16)
            cv2.imwrite(
                str(fused_depth_root / f"{Path(image.name).stem}.png"), output_depth
            )
        records.append(
            {
                "image": image.name,
                **screen_target,
                **stats,
                "valid_pixels": valid_pixels,
                "integrated": True,
            }
        )
        integrated += 1
        if integrated % 50 == 0:
            print(f"hybrid TSDF {integrated}/{len(images)}", flush=True)
    if integrated < 2:
        raise RuntimeError("Fewer than two ToF-anchored depth maps were integrated")

    dense_root = output_root / "dense"
    mesh_root = output_root / "mesh"
    dense_root.mkdir(parents=True, exist_ok=True)
    mesh_root.mkdir(parents=True, exist_ok=True)
    cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    raw_mesh_path = mesh_root / "tof_anchored_stereo_tsdf_raw.ply"
    if not o3d.io.write_triangle_mesh(
        str(raw_mesh_path), mesh, write_ascii=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"Could not write raw hybrid mesh: {raw_mesh_path}")
    component_mode = (
        args.component_selection if args.keep_largest_component else "none"
    )
    mesh, component_info = select_mesh_component(
        mesh,
        mode=component_mode,
        max_extent=args.component_max_extent,
        np=np,
    )
    mesh.compute_vertex_normals()
    cloud_path = dense_root / "tof_anchored_stereo_fused.ply"
    mesh_path = mesh_root / "tof_anchored_stereo_tsdf.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False):
        raise RuntimeError(f"Could not write hybrid point cloud: {cloud_path}")
    if not o3d.io.write_triangle_mesh(
        str(mesh_path), mesh, write_ascii=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"Could not write hybrid mesh: {mesh_path}")
    mesh_stats = validate_mesh(mesh_path)
    stereo_fill = sum(record.get("stereo_fill_pixels", 0) for record in records)
    tof_pixels = sum(record.get("tof_pixels", 0) for record in records)
    return {
        "integrated_frames": integrated,
        "stereo_reference_frames": len(stereo_by_reference),
        "tof_pixels": tof_pixels,
        "stereo_fill_pixels": stereo_fill,
        "stereo_fill_fraction": stereo_fill / max(1, tof_pixels + stereo_fill),
        "target": target_info,
        "cloud": str(cloud_path),
        "cloud_points": len(cloud.points),
        "mesh": str(mesh_path),
        "mesh_stats": mesh_stats,
        "raw_mesh": str(raw_mesh_path),
        "component_selection": component_info,
        "records": records,
        "sparse_points": len(points),
    }


def _run_foundation_depth_only(
    model_dir: Path,
    image_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    argv = [
        str(model_dir),
        str(image_root),
        str(output_root),
        "--model",
        str(args.model),
        "--model-sha256",
        args.model_sha256,
        "--provider",
        args.provider,
        "--device-id",
        str(args.device_id),
        "--inference-width",
        str(args.inference_width),
        "--inference-height",
        str(args.inference_height),
        "--rectification-alpha",
        "1",
        "--spatial-bins",
        str(args.spatial_bins),
        "--references-per-bin",
        str(args.references_per_bin),
        "--sources-per-reference",
        str(args.sources_per_reference),
        "--max-pairs",
        str(args.max_pairs),
        "--min-baseline-ratio",
        str(args.min_baseline_ratio),
        "--target-baseline-ratio",
        str(args.target_baseline_ratio),
        "--max-baseline-ratio",
        str(args.max_baseline_ratio),
        "--max-view-angle",
        str(args.max_view_angle),
        "--max-expected-disparity",
        str(args.max_expected_disparity),
        "--min-valid-pixels",
        str(args.min_stereo_pixels),
        "--voxel-size",
        str(args.voxel_size),
        "--no-lr-check",
        "--depth-only",
        "--save-pairs",
    ]
    foundation_args = build_foundation_parser().parse_args(argv)
    validate_foundation_arguments(foundation_args)
    return run_experiment(foundation_args)


def run_hybrid(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    mkv_path = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not mkv_path.is_file():
        raise FileNotFoundError(f"MKV not found: {mkv_path}")
    if not args.k4a_bin.expanduser().resolve().is_dir():
        raise FileNotFoundError(f"K4A runtime not found: {args.k4a_bin}")
    if args.resume:
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        ensure_new_output(output_root)

    frames_manifest = output_root / "frames" / "frames.json"
    if frames_manifest.is_file():
        frames = json.loads(frames_manifest.read_text(encoding="utf-8"))
    else:
        frames = extract_common_rgbd_frames(
            mkv_path,
            output_root,
            args.k4a_bin.expanduser().resolve(),
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            source_fps=args.source_fps,
            jpeg_quality=args.jpeg_quality,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            undistort_alpha=args.undistort_alpha,
        )
    image_root = Path(frames["rgb_root"])
    depth_root = Path(frames["depth_root"])
    if args.stop_after == "extract":
        result = {
            "status": "complete",
            "stage": "extract",
            "input": str(mkv_path),
            "output": str(output_root),
            "extracted_frames": len(frames["frames"]),
            "frames": str(frames_manifest),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output_root / "run.json", result)
        return result

    realityscan_root = output_root / "realityscan"
    realityscan_manifest = realityscan_root / "run.json"
    if not realityscan_manifest.is_file():
        execute_pipeline(
            PipelineConfig(
                image_dir=image_root,
                output_dir=realityscan_root,
                backend=BackendName.REALITYSCAN,
                target=Target.SFM,
                timeout_seconds=args.timeout,
                realityscan_exe=args.realityscan_exe,
                realityscan_no_distortion=True,
                realityscan_shared_intrinsics=True,
                realityscan_sensitive_alignment=True,
            )
        )
    if args.stop_after == "sfm":
        result = {
            "status": "complete",
            "stage": "sfm",
            "input": str(mkv_path),
            "output": str(output_root),
            "extracted_frames": len(frames["frames"]),
            "realityscan": str(realityscan_manifest),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output_root / "run.json", result)
        return result
    source_model = realityscan_root / "colmap" / "sparse" / "0"
    exported_image_root = realityscan_root / "colmap" / "images"
    np, cv2, _ = _lazy_imports()
    calibration = json.loads(
        (output_root / "calibration.json").read_text(encoding="utf-8")
    )
    scale_manifest = output_root / f"{args.metric_name}_scale.json"
    metric_root = output_root / args.metric_name
    metric_model = metric_root / "sparse" / "0"
    if scale_manifest.is_file() and metric_model.is_dir():
        scale_info = json.loads(scale_manifest.read_text(encoding="utf-8"))
    else:
        scale_info = estimate_metric_scale(
            source_model,
            depth_root,
            depth_scale=args.depth_scale,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            max_samples_per_image=args.max_scale_samples_per_image,
            source_intrinsic=calibration["undistorted_intrinsic"],
            np=np,
            cv2=cv2,
        )
        if args.refine_translations:
            translations, translation_info = refine_metric_translations(
                source_model,
                depth_root,
                scale=float(scale_info["scale"]),
                source_intrinsic=calibration["undistorted_intrinsic"],
                depth_scale=args.depth_scale,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                max_samples_per_image=args.max_scale_samples_per_image,
                max_correction=args.max_translation_correction,
                np=np,
                cv2=cv2,
            )
        else:
            translations = {}
            translation_info = {"refined_images": 0, "disabled": True}
        scale_info["translation_refinement"] = translation_info
        write_json(scale_manifest, scale_info)
        write_metric_colmap_model(
            source_model,
            exported_image_root,
            metric_root,
            float(scale_info["scale"]),
            translations=translations,
        )
        if args.icp_refine:
            _, _, o3d = _lazy_imports()
            poses, icp_info = refine_metric_poses_with_tof_icp(
                metric_model,
                metric_root / "images",
                depth_root,
                calibration["undistorted_intrinsic"],
                args,
                np,
                cv2,
                o3d,
            )
            scale_info["icp_refinement"] = icp_info
            write_json(scale_manifest, scale_info)
            write_metric_colmap_model(
                source_model,
                exported_image_root,
                metric_root,
                float(scale_info["scale"]),
                poses=poses,
            )
    if args.stop_after == "metric":
        result = {
            "status": "complete",
            "stage": "metric",
            "input": str(mkv_path),
            "output": str(output_root),
            "extracted_frames": len(frames["frames"]),
            "realityscan": str(realityscan_manifest),
            "metric_scale": scale_info,
            "metric_colmap_model": str(metric_model),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output_root / "run.json", result)
        return result

    foundation_root = output_root / "foundationstereo"
    if not args.tof_only and not (foundation_root / "run.json").is_file():
        _run_foundation_depth_only(metric_model, metric_root / "images", foundation_root, args)
    if args.stop_after == "stereo" and not args.tof_only:
        result = {
            "status": "complete",
            "stage": "stereo",
            "input": str(mkv_path),
            "output": str(output_root),
            "extracted_frames": len(frames["frames"]),
            "realityscan": str(realityscan_manifest),
            "metric_scale": scale_info,
            "metric_colmap_model": str(metric_model),
            "foundationstereo": str(foundation_root / "run.json"),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output_root / "run.json", result)
        return result

    hybrid_root = output_root / args.hybrid_name
    hybrid_root.mkdir(parents=True, exist_ok=True)
    hybrid = fuse_tof_anchored_depths(
        metric_model,
        metric_root / "images",
        depth_root,
        None if args.tof_only else foundation_root,
        hybrid_root,
        args,
    )
    textured_mesh = None
    texture_files: list[str] = []
    if args.texture:
        _, _, o3d = _lazy_imports()
        texturing_mesh, _ = _prepare_texturing_mesh(
            Path(hybrid["mesh"]), hybrid_root, args.texture_max_triangles, o3d
        )
        runner = CommandRunner(
            hybrid_root / "logs" / "texture.log", timeout_seconds=args.timeout
        )
        textured_mesh_path, textures = run_openmvs_texturing(
            hybrid_root,
            texturing_mesh,
            metric_root,
            args.openmvs_dir.expanduser().resolve(),
            runner,
        )
        textured_mesh = str(textured_mesh_path)
        texture_files = [str(path) for path in textures]

    result = {
        "status": "complete",
        "mode": "realityscan-tof-foundationstereo-hybrid",
        "input": str(mkv_path),
        "output": str(output_root),
        "extracted_frames": len(frames["frames"]),
        "realityscan": str(realityscan_manifest),
        "metric_scale": scale_info,
        "metric_colmap_model": str(metric_model),
        "foundationstereo": (
            None if args.tof_only else str(foundation_root / "run.json")
        ),
        "hybrid": hybrid,
        "textured_mesh": textured_mesh,
        "texture_files": texture_files,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output_root / "run.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rgbd-hybrid",
        description="RealityScan poses + ToF-anchored FoundationStereo depth fusion",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--k4a-bin", type=Path, default=DEFAULT_K4A_BIN)
    parser.add_argument("--realityscan-exe", type=Path)
    parser.add_argument("--openmvs-dir", type=Path, default=DEFAULT_OPENMVS_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--undistort-alpha", type=float, default=0.0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-min", type=float, default=0.25)
    parser.add_argument("--depth-max", type=float, default=0.8)
    parser.add_argument("--max-scale-samples-per-image", type=int, default=1000)
    parser.add_argument("--max-translation-correction", type=float, default=0.10)
    parser.add_argument(
        "--refine-translations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--icp-refine", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--icp-width", type=int, default=480)
    parser.add_argument("--icp-height", type=int, default=270)
    parser.add_argument("--icp-voxel-size", type=float, default=0.004)
    parser.add_argument("--icp-max-correspondence", type=float, default=0.02)
    parser.add_argument("--icp-min-points", type=int, default=500)
    parser.add_argument("--icp-max-frame-gap", type=int, default=15)
    parser.add_argument("--icp-min-fitness", type=float, default=0.25)
    parser.add_argument("--icp-max-rmse", type=float, default=0.012)
    parser.add_argument("--icp-max-pose-correction", type=float, default=0.05)
    parser.add_argument("--icp-max-rotation-correction", type=float, default=8.0)
    parser.add_argument("--icp-realityscan-prior", type=float, default=0.05)
    parser.add_argument(
        "--icp-loop-closure", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--icp-loop-max-center-distance", type=float, default=0.25)
    parser.add_argument("--icp-loop-max-view-angle", type=float, default=25.0)
    parser.add_argument("--metric-name", default="metric_colmap")
    parser.add_argument("--inference-width", type=int, default=960)
    parser.add_argument("--inference-height", type=int, default=544)
    parser.add_argument("--spatial-bins", type=int, default=48)
    parser.add_argument("--references-per-bin", type=int, default=4)
    parser.add_argument("--sources-per-reference", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--min-baseline-ratio", type=float, default=0.003)
    parser.add_argument("--target-baseline-ratio", type=float, default=0.03)
    parser.add_argument("--max-baseline-ratio", type=float, default=0.08)
    parser.add_argument("--max-view-angle", type=float, default=10.0)
    parser.add_argument("--max-expected-disparity", type=float, default=200.0)
    parser.add_argument("--min-stereo-pixels", type=int, default=5000)
    parser.add_argument("--fusion-width", type=int, default=960)
    parser.add_argument("--fusion-height", type=int, default=540)
    parser.add_argument("--voxel-size", type=float, default=0.001)
    parser.add_argument("--sdf-trunc", type=float, default=0.008)
    parser.add_argument("--tof-edge-absolute", type=float, default=0.015)
    parser.add_argument("--tof-edge-relative", type=float, default=0.01)
    parser.add_argument("--stereo-tof-absolute-tolerance", type=float, default=0.015)
    parser.add_argument("--stereo-tof-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--stereo-anchor-radius", type=int, default=5)
    parser.add_argument("--min-fusion-pixels", type=int, default=5000)
    parser.add_argument("--target-seed-fraction", type=float, default=0.10)
    parser.add_argument("--target-seed-x-fraction", type=float, default=0.50)
    parser.add_argument("--target-seed-y-fraction", type=float, default=0.40)
    parser.add_argument("--target-screen-radius-x", type=float, default=0.30)
    parser.add_argument("--target-screen-radius-y", type=float, default=0.55)
    parser.add_argument("--target-depth-band", type=float, default=0.20)
    parser.add_argument(
        "--skin-seed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--grabcut-mask", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--grabcut-x-min", type=float, default=0.22)
    parser.add_argument("--grabcut-x-max", type=float, default=0.72)
    parser.add_argument("--grabcut-y-min", type=float, default=0.01)
    parser.add_argument("--grabcut-y-max", type=float, default=0.72)
    parser.add_argument("--grabcut-iterations", type=int, default=2)
    parser.add_argument("--target-radius", type=float, default=0.0)
    parser.add_argument(
        "--keep-largest-component", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--component-selection",
        choices=("largest", "skin-compact", "none"),
        default="skin-compact",
    )
    parser.add_argument("--component-max-extent", type=float, default=0.45)
    parser.add_argument("--texture", action="store_true")
    parser.add_argument(
        "--tof-only",
        action="store_true",
        help="skip FoundationStereo and build the RealityScan-pose ToF baseline",
    )
    parser.add_argument("--texture-max-triangles", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=14400.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hybrid-name", default="hybrid")
    parser.add_argument(
        "--stop-after",
        choices=("extract", "sfm", "metric", "stereo", "fusion"),
        default="fusion",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    for name in (
        "frame_step",
        "max_scale_samples_per_image",
        "inference_width",
        "inference_height",
        "spatial_bins",
        "references_per_bin",
        "sources_per_reference",
        "max_pairs",
        "min_stereo_pixels",
        "fusion_width",
        "fusion_height",
        "min_fusion_pixels",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("max-frames must be at least 2")
    if args.inference_width % 32 or args.inference_height % 32:
        raise ValueError("inference dimensions must be multiples of 32")
    if not 0 < args.depth_min < args.depth_max:
        raise ValueError("depth range must satisfy 0 < min < max")
    if args.depth_scale <= 0 or args.voxel_size <= 0 or args.sdf_trunc <= 0:
        raise ValueError("depth-scale, voxel-size, and sdf-trunc must be positive")
    if args.stereo_anchor_radius < 1:
        raise ValueError("stereo-anchor-radius must be positive")
    if not 0 < args.target_seed_fraction <= 0.5 or args.target_radius < 0:
        raise ValueError("target seed fraction must be positive and radius non-negative")
    for name in (
        "target_seed_x_fraction",
        "target_seed_y_fraction",
        "target_screen_radius_x",
        "target_screen_radius_y",
    ):
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f"{name.replace('_', '-')} must be in (0, 1]")
    if args.target_depth_band <= 0:
        raise ValueError("target-depth-band must be positive")
    if not (
        0 <= args.grabcut_x_min < args.grabcut_x_max <= 1
        and 0 <= args.grabcut_y_min < args.grabcut_y_max <= 1
    ):
        raise ValueError("GrabCut rectangle fractions must be ordered within [0, 1]")
    if args.grabcut_iterations < 1:
        raise ValueError("grabcut-iterations must be positive")
    if not args.hybrid_name or Path(args.hybrid_name).name != args.hybrid_name:
        raise ValueError("hybrid-name must be one directory name")
    if not args.metric_name or Path(args.metric_name).name != args.metric_name:
        raise ValueError("metric-name must be one directory name")
    if args.max_translation_correction <= 0:
        raise ValueError("max-translation-correction must be positive")
    if args.icp_width <= 0 or args.icp_height <= 0 or args.icp_min_points <= 0:
        raise ValueError("ICP dimensions and min points must be positive")
    if (
        args.icp_voxel_size <= 0
        or args.icp_max_correspondence <= 0
        or args.icp_max_frame_gap <= 0
        or args.icp_max_rmse <= 0
        or args.icp_max_pose_correction <= 0
        or args.icp_max_rotation_correction <= 0
        or args.icp_loop_max_center_distance <= 0
        or args.icp_loop_max_view_angle <= 0
    ):
        raise ValueError("ICP thresholds must be positive")
    if not 0 <= args.icp_min_fitness <= 1 or not 0 <= args.icp_realityscan_prior <= 1:
        raise ValueError("ICP fitness and RealityScan prior must be in [0, 1]")
    if args.component_max_extent <= 0:
        raise ValueError("component-max-extent must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be between 1 and 100")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_arguments(args)
        result = run_hybrid(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
