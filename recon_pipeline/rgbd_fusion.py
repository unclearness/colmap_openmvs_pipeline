from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from recon_pipeline.artifacts import ensure_new_output, validate_mesh, write_json
from recon_pipeline.process import CommandRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K4A_BIN = (
    PROJECT_ROOT
    / "tools"
    / "Azure-Kinect-Sensor-SDK-1.4.2"
    / "lib"
    / "native"
    / "amd64"
    / "release"
)
DEFAULT_OPENMVS_DIR = (
    PROJECT_ROOT / "tools" / "OpenMVS-2.4.0" / "vc17" / "x64" / "Release"
)


@dataclass(slots=True)
class ColorCalibration:
    width: int
    height: int
    intrinsic: list[list[float]]
    distortion: list[float]
    undistorted_intrinsic: list[list[float]]
    roi: list[int]


@dataclass(slots=True)
class PoseRecord:
    source_frame: int
    timestamp_seconds: float
    camera_to_world: list[list[float]]
    fitness: float
    inlier_rmse: float
    texture_image: str | None = None


def _lazy_imports() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "RGB-D fusion requires Python 3.12 with open3d, numpy, and "
            "opencv-python-headless. See README.md."
        ) from exc
    return np, cv2, o3d


def extract_k4a_calibration(mkv_path: Path, destination: Path) -> Path:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(mkv_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr.strip()}")
    metadata = json.loads(probe.stdout)
    attachment_index = None
    for stream in metadata.get("streams", []):
        tags = stream.get("tags", {})
        if stream.get("codec_type") == "attachment" and (
            tags.get("K4A_CALIBRATION_FILE") or tags.get("filename") == "calibration.json"
        ):
            attachment_index = int(stream["index"])
            break
    if attachment_index is None:
        raise RuntimeError("MKV has no K4A calibration attachment")

    destination.parent.mkdir(parents=True, exist_ok=True)
    null_output = "NUL" if os.name == "nt" else "/dev/null"
    extraction = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            f"-dump_attachment:{attachment_index}",
            str(destination),
            "-i",
            str(mkv_path),
            "-t",
            "0.001",
            "-f",
            "null",
            null_output,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if extraction.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"Could not extract K4A calibration: {extraction.stderr.strip()}")
    return destination


def derive_color_calibration(
    calibration_json: Path,
    width: int,
    height: int,
    *,
    alpha: float = 0.0,
) -> ColorCalibration:
    np, cv2, _ = _lazy_imports()
    payload = json.loads(calibration_json.read_text(encoding="utf-8"))
    cameras = payload["CalibrationInformation"]["Cameras"]
    color = next(
        camera
        for camera in cameras
        if camera.get("Purpose") == "CALIBRATION_CameraPurposePhotoVideo"
    )
    params = [float(value) for value in color["Intrinsics"]["ModelParameters"]]
    if color["Intrinsics"].get("ModelType") != "CALIBRATION_LensDistortionModelBrownConrady":
        raise ValueError("Only K4A Brown-Conrady color calibration is supported")

    mode = {
        (1280, 720): ((1280, 960), (0, 120)),
        (1920, 1080): ((1920, 1440), (0, 180)),
        (2560, 1440): ((2560, 1920), (0, 240)),
        (2048, 1536): ((2048, 1536), (0, 0)),
        (3840, 2160): ((3840, 2880), (0, 360)),
        (4096, 3072): ((4096, 3072), (0, 0)),
    }.get((width, height))
    if mode is None:
        raise ValueError(f"Unsupported K4A color resolution: {width}x{height}")
    (calibration_width, calibration_height), (crop_x, crop_y) = mode

    cx = params[0] * calibration_width - crop_x - 0.5
    cy = params[1] * calibration_height - crop_y - 0.5
    fx = params[2] * calibration_width
    fy = params[3] * calibration_height
    intrinsic = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    # K4A order after fx/fy is k1..k6, codx/cody, p2, p1. OpenCV's
    # rational model expects k1, k2, p1, p2, k3, k4, k5, k6.
    distortion = np.array(
        [params[4], params[5], params[13], params[12], params[6], params[7], params[8], params[9]],
        dtype=np.float64,
    )
    undistorted, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic,
        distortion,
        (width, height),
        alpha,
        (width, height),
        centerPrincipalPoint=False,
    )
    return ColorCalibration(
        width=width,
        height=height,
        intrinsic=intrinsic.tolist(),
        distortion=distortion.tolist(),
        undistorted_intrinsic=undistorted.tolist(),
        roi=[int(value) for value in roi],
    )


def _scaled_intrinsic(intrinsic: Any, scale_x: float, scale_y: float, np: Any) -> Any:
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def _rotation_angle_degrees(transform: Any, np: Any) -> float:
    trace = float(np.trace(transform[:3, :3]))
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cosine))


def _rotation_to_quaternion(rotation: Any, np: Any) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return tuple(float(value) for value in quaternion)


def _write_colmap_model(
    root: Path,
    calibration: ColorCalibration,
    poses: list[PoseRecord],
) -> list[PoseRecord]:
    np, _, _ = _lazy_imports()
    model = root / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)
    intrinsic = np.asarray(calibration.undistorted_intrinsic, dtype=np.float64)
    (model / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {calibration.width} {calibration.height} "
        f"{intrinsic[0, 0]:.12g} {intrinsic[1, 1]:.12g} "
        f"{intrinsic[0, 2]:.12g} {intrinsic[1, 2]:.12g}\n",
        encoding="utf-8",
        newline="\n",
    )
    image_lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
    ]
    texture_poses = [pose for pose in poses if pose.texture_image is not None]
    for image_id, pose in enumerate(texture_poses, start=1):
        camera_to_world = np.asarray(pose.camera_to_world, dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        qw, qx, qy, qz = _rotation_to_quaternion(world_to_camera[:3, :3], np)
        tx, ty, tz = (float(value) for value in world_to_camera[:3, 3])
        image_lines.append(
            f"{image_id} {qw:.17g} {qx:.17g} {qy:.17g} {qz:.17g} "
            f"{tx:.17g} {ty:.17g} {tz:.17g} 1 {pose.texture_image}\n\n"
        )
    (model / "images.txt").write_text(
        "".join(image_lines), encoding="utf-8", newline="\n"
    )
    (model / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "# Number of points: 0\n",
        encoding="utf-8",
        newline="\n",
    )
    return texture_poses


def _prepare_openmvs_input(colmap_root: Path, destination: Path) -> Path:
    sparse_source = colmap_root / "sparse" / "0"
    sparse_destination = destination / "sparse"
    sparse_destination.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(sparse_source / name, sparse_destination / name)
    return destination


def select_texture_poses(
    poses: Sequence[PoseRecord],
    image_root: Path,
    target_center: Sequence[float],
    max_views: int,
    min_view_cosine: float,
    np: Any,
    cv2: Any,
) -> list[PoseRecord]:
    texture_poses = [pose for pose in poses if pose.texture_image is not None]
    if max_views <= 0 or len(texture_poses) <= max_views:
        return texture_poses

    center = np.asarray(target_center, dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for pose in texture_poses:
        camera_to_world = np.asarray(pose.camera_to_world, dtype=np.float64)
        camera = camera_to_world[:3, 3]
        to_center = center - camera
        distance = float(np.linalg.norm(to_center))
        if distance <= 1e-9:
            continue
        view_cosine = float(
            np.dot(camera_to_world[:3, 2], to_center / distance)
        )
        if view_cosine < min_view_cosine:
            continue
        image = cv2.imread(
            str(image_root / str(pose.texture_image)), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            continue
        preview = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(preview, cv2.CV_64F).var())
        candidates.append(
            {
                "pose": pose,
                "direction": (camera - center) / distance,
                "sharpness": sharpness,
            }
        )

    if not candidates:
        raise RuntimeError(
            "No texture view faces the reconstructed mesh center; check poses or "
            "lower --texture-min-view-cosine"
        )
    if len(candidates) <= max_views:
        return sorted(
            (candidate["pose"] for candidate in candidates),
            key=lambda pose: pose.source_frame,
        )

    sharpness_values = np.asarray(
        [candidate["sharpness"] for candidate in candidates], dtype=np.float64
    )
    sharpness_min = float(sharpness_values.min())
    sharpness_span = float(sharpness_values.max() - sharpness_min) or 1.0
    first = max(candidates, key=lambda candidate: candidate["sharpness"])
    selected = [first]
    remaining = [candidate for candidate in candidates if candidate is not first]
    while remaining and len(selected) < max_views:
        def score(candidate: dict[str, Any]) -> float:
            angular_separation = min(
                float(
                    np.arccos(
                        np.clip(
                            np.dot(candidate["direction"], chosen["direction"]),
                            -1.0,
                            1.0,
                        )
                    )
                )
                for chosen in selected
            )
            normalized_sharpness = (
                candidate["sharpness"] - sharpness_min
            ) / sharpness_span
            return angular_separation + 0.20 * normalized_sharpness

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(
        (candidate["pose"] for candidate in selected),
        key=lambda pose: pose.source_frame,
    )


def run_openmvs_texturing(
    output_root: Path,
    mesh_path: Path,
    colmap_root: Path,
    openmvs_dir: Path,
    runner: CommandRunner,
    selected_poses: Sequence[PoseRecord] | None = None,
    cost_smoothness_ratio: float = 1.0,
) -> tuple[Path, list[Path]]:
    interface = openmvs_dir / "InterfaceCOLMAP.exe"
    texturer = openmvs_dir / "TextureMesh.exe"
    for executable in (interface, texturer):
        if not executable.is_file():
            raise FileNotFoundError(f"OpenMVS executable not found: {executable}")

    native = output_root / "native" / "openmvs"
    input_workspace = _prepare_openmvs_input(
        colmap_root, output_root / "native" / "openmvs_input"
    )
    native.mkdir(parents=True, exist_ok=True)
    scene = native / "scene.mvs"
    textured_dir = output_root / "mesh" / "textured"
    textured_dir.mkdir(parents=True, exist_ok=True)
    textured_mesh = textured_dir / "mesh.obj"
    views_file = None
    if selected_poses:
        views_file = native / "texture_views.txt"
        views_file.write_text(
            "".join(f"{pose.texture_image}\n" for pose in selected_poses),
            encoding="utf-8",
            newline="\n",
        )

    runner.run(
        "openmvs.interface_colmap",
        [
            interface,
            "-i",
            input_workspace,
            "--image-folder",
            colmap_root / "images",
            "-o",
            scene,
            "-w",
            native,
        ],
        cwd=native,
    )
    texture_command: list[object] = [
        texturer,
        "-i",
        scene,
        "-m",
        mesh_path,
        "-o",
        textured_mesh,
        "-w",
        native,
        "--export-type",
        "obj",
        "--cost-smoothness-ratio",
        cost_smoothness_ratio,
        "--resolution-level",
        "0",
        "--global-seam-leveling",
        "0",
        "--local-seam-leveling",
        "0",
    ]
    if views_file is not None:
        texture_command.extend(["--views-file", views_file])
    runner.run(
        "openmvs.texture_mesh",
        texture_command,
        cwd=native,
    )
    validate_mesh(textured_mesh)
    textures = sorted(
        [
            path
            for path in textured_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    if not textures:
        raise RuntimeError(f"OpenMVS produced no texture image in {textured_dir}")
    return textured_mesh, textures


def _prepare_texturing_mesh(
    source: Path,
    output_root: Path,
    max_triangles: int | None,
    o3d: Any,
) -> tuple[Path, dict[str, int | str]]:
    mesh = o3d.io.read_triangle_mesh(str(source))
    if max_triangles is None or len(mesh.triangles) <= max_triangles:
        return source, validate_mesh(source)
    mesh = mesh.simplify_quadric_decimation(max_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    destination = output_root / "mesh" / "open3d_mesh_texturing.ply"
    if not o3d.io.write_triangle_mesh(
        str(destination), mesh, write_ascii=False, compressed=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"Could not write texturing mesh: {destination}")
    return destination, validate_mesh(destination)


def resume_texturing(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output.expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"Existing RGB-D output not found: {output_root}")
    raw_mesh = output_root / "mesh" / "open3d_mesh.ply"
    colmap_root = output_root / "colmap"
    trajectory_path = output_root / "trajectory.json"
    for required in (raw_mesh, trajectory_path, colmap_root / "sparse" / "0" / "images.txt"):
        if not required.is_file():
            raise FileNotFoundError(f"Resume input not found: {required}")
    np, cv2, o3d = _lazy_imports()
    texturing_mesh, texturing_stats = _prepare_texturing_mesh(
        raw_mesh, output_root, args.texture_max_triangles, o3d
    )
    runner = CommandRunner(
        output_root / "logs" / "texture_resume.log", timeout_seconds=args.timeout
    )
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))["poses"]
    poses = [PoseRecord(**pose) for pose in trajectory]
    mesh_center = (
        o3d.io.read_triangle_mesh(str(texturing_mesh))
        .get_axis_aligned_bounding_box()
        .get_center()
    )
    selected_poses = select_texture_poses(
        poses,
        colmap_root / "images",
        mesh_center,
        args.texture_view_count,
        args.texture_min_view_cosine,
        np,
        cv2,
    )
    textured_mesh, textures = run_openmvs_texturing(
        output_root,
        texturing_mesh,
        colmap_root,
        args.openmvs_dir.expanduser().resolve(),
        runner,
        selected_poses,
        args.texture_cost_smoothness_ratio,
    )
    payload = {
        "status": "complete",
        "mode": "resume-texture",
        "input": str(args.input.expanduser().resolve()),
        "output": str(output_root),
        "integrated_frames": len(trajectory),
        "texture_keyframes": sum(pose.get("texture_image") is not None for pose in trajectory),
        "texture_views_selected": len(selected_poses),
        "raw_mesh": str(raw_mesh),
        "raw_mesh_stats": validate_mesh(raw_mesh),
        "texturing_mesh": str(texturing_mesh),
        "texturing_mesh_stats": texturing_stats,
        "textured_mesh": str(textured_mesh),
        "textured_mesh_stats": validate_mesh(textured_mesh),
        "texture_files": [str(path) for path in textures],
        "commands": [record.to_dict() for record in runner.records],
    }
    write_json(output_root / "run.json", payload)
    return payload


def run_roi_reintegration(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.roi_from.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    mkv_path = args.input.expanduser().resolve()
    k4a_bin = args.k4a_bin.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source trajectory output not found: {source_root}")
    for required in (
        source_root / "trajectory.json",
        source_root / "calibration.json",
        source_root / "colmap" / "images",
    ):
        if not required.exists():
            raise FileNotFoundError(f"ROI reintegration input not found: {required}")
    ensure_new_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    runner = CommandRunner(
        output_root / "logs" / "pipeline.log", timeout_seconds=args.timeout
    )
    started = time.monotonic()

    os.environ["K4A_LIB_DIR"] = str(k4a_bin)
    os.environ["PATH"] = str(k4a_bin) + os.pathsep + os.environ.get("PATH", "")
    dll_directory_handle = (
        os.add_dll_directory(str(k4a_bin)) if os.name == "nt" else None
    )
    np, cv2, o3d = _lazy_imports()

    calibration_data = json.loads(
        (source_root / "calibration.json").read_text(encoding="utf-8")
    )
    calibration = ColorCalibration(**calibration_data)
    write_json(output_root / "calibration.json", calibration_data)
    source_poses = [
        PoseRecord(**pose)
        for pose in json.loads(
            (source_root / "trajectory.json").read_text(encoding="utf-8")
        )["poses"]
    ]
    if args.roi_source_frame_min is not None:
        source_poses = [
            pose for pose in source_poses if pose.source_frame >= args.roi_source_frame_min
        ]
    if args.roi_source_frame_max is not None:
        source_poses = [
            pose for pose in source_poses if pose.source_frame <= args.roi_source_frame_max
        ]
    pose_by_frame = {pose.source_frame: pose for pose in source_poses}

    full_intrinsic = np.asarray(calibration.intrinsic, dtype=np.float64)
    full_undistorted = np.asarray(calibration.undistorted_intrinsic, dtype=np.float64)
    full_distortion = np.asarray(calibration.distortion, dtype=np.float64)
    tracking_width = args.tracking_width
    tracking_height = round(calibration.height * tracking_width / calibration.width)
    scale_x = tracking_width / calibration.width
    scale_y = tracking_height / calibration.height
    track_intrinsic = _scaled_intrinsic(full_intrinsic, scale_x, scale_y, np)
    track_undistorted = _scaled_intrinsic(full_undistorted, scale_x, scale_y, np)
    track_map_x, track_map_y = cv2.initUndistortRectifyMap(
        track_intrinsic,
        full_distortion,
        None,
        track_undistorted,
        (tracking_width, tracking_height),
        cv2.CV_32FC1,
    )
    pixel_u, pixel_v = np.meshgrid(
        np.arange(tracking_width, dtype=np.float32),
        np.arange(tracking_height, dtype=np.float32),
    )
    ray_x = (pixel_u - track_undistorted[0, 2]) / track_undistorted[0, 0]
    ray_y = (pixel_v - track_undistorted[1, 2]) / track_undistorted[1, 1]
    roi_min = np.asarray(args.roi_min, dtype=np.float64)
    roi_max = np.asarray(args.roi_max, dtype=np.float64)
    if np.any(roi_max <= roi_min):
        raise ValueError("Each ROI max coordinate must be greater than ROI min")

    device = o3d.core.Device(args.device)
    intrinsic_tensor = o3d.core.Tensor(track_undistorted, o3d.core.Dtype.Float64)
    model = o3d.t.pipelines.slam.Model(
        args.roi_voxel_size,
        16,
        args.roi_block_count,
        o3d.core.Tensor(np.identity(4), o3d.core.Dtype.Float64, device),
        device,
    )
    input_frame = o3d.t.pipelines.slam.Frame(
        tracking_height, tracking_width, intrinsic_tensor, device
    )

    colmap_root = output_root / "colmap"
    image_root = colmap_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    accepted_poses: list[PoseRecord] = []
    source_frame = -1
    integrated_frames = 0
    skipped_visibility = 0
    roi_pixel_counts: list[int] = []

    reader = o3d.io.AzureKinectMKVReader()
    if not reader.open(str(mkv_path)):
        raise RuntimeError(f"Open3D could not open MKV: {mkv_path}")
    try:
        while not reader.is_eof() and pose_by_frame:
            rgbd = reader.next_frame()
            source_frame += 1
            pose = pose_by_frame.pop(source_frame, None)
            if pose is None or rgbd is None:
                continue
            depth_small = cv2.resize(
                np.asarray(rgbd.depth),
                (tracking_width, tracking_height),
                interpolation=cv2.INTER_NEAREST,
            )
            color_small = cv2.resize(
                np.asarray(rgbd.color),
                (tracking_width, tracking_height),
                interpolation=cv2.INTER_AREA,
            )
            depth_track = cv2.remap(
                depth_small,
                track_map_x,
                track_map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            ).astype(np.uint16, copy=False)
            color_track = cv2.remap(
                color_small,
                track_map_x,
                track_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            ).astype(np.uint8, copy=False)

            depth_m = depth_track.astype(np.float32) / args.depth_scale
            x_camera = ray_x * depth_m
            y_camera = ray_y * depth_m
            camera_to_world = np.asarray(pose.camera_to_world, dtype=np.float64)
            rotation = camera_to_world[:3, :3]
            translation = camera_to_world[:3, 3]
            world_x = (
                rotation[0, 0] * x_camera
                + rotation[0, 1] * y_camera
                + rotation[0, 2] * depth_m
                + translation[0]
            )
            world_y = (
                rotation[1, 0] * x_camera
                + rotation[1, 1] * y_camera
                + rotation[1, 2] * depth_m
                + translation[1]
            )
            world_z = (
                rotation[2, 0] * x_camera
                + rotation[2, 1] * y_camera
                + rotation[2, 2] * depth_m
                + translation[2]
            )
            inside = (
                (depth_track > 0)
                & (world_x >= roi_min[0])
                & (world_x <= roi_max[0])
                & (world_y >= roi_min[1])
                & (world_y <= roi_max[1])
                & (world_z >= roi_min[2])
                & (world_z <= roi_max[2])
            )
            roi_pixels = int(inside.sum())
            roi_pixel_counts.append(roi_pixels)
            if roi_pixels < args.roi_min_pixels:
                skipped_visibility += 1
                continue
            masked_depth = np.where(inside, depth_track, 0).astype(np.uint16)
            input_frame.set_data_from_image(
                "depth",
                o3d.t.geometry.Image(
                    o3d.core.Tensor(np.ascontiguousarray(masked_depth), device=device)
                ),
            )
            input_frame.set_data_from_image(
                "color",
                o3d.t.geometry.Image(
                    o3d.core.Tensor(np.ascontiguousarray(color_track), device=device)
                ),
            )
            transform = o3d.core.Tensor(
                camera_to_world, o3d.core.Dtype.Float64, device
            )
            model.update_frame_pose(integrated_frames, transform)
            model.integrate(
                input_frame,
                args.depth_scale,
                args.depth_max,
                args.roi_trunc_voxel_multiplier,
            )
            integrated_frames += 1

            texture_image = None
            if pose.texture_image is not None:
                source_image = source_root / "colmap" / "images" / pose.texture_image
                if source_image.is_file():
                    texture_image = pose.texture_image
                    shutil.copy2(source_image, image_root / texture_image)
            accepted_poses.append(
                PoseRecord(
                    source_frame=pose.source_frame,
                    timestamp_seconds=pose.timestamp_seconds,
                    camera_to_world=pose.camera_to_world,
                    fitness=pose.fitness,
                    inlier_rmse=pose.inlier_rmse,
                    texture_image=texture_image,
                )
            )
            if integrated_frames % args.progress_interval == 0:
                print(
                    f"roi integrated={integrated_frames} source={source_frame} "
                    f"pixels={roi_pixels} skipped={skipped_visibility}",
                    flush=True,
                )
    finally:
        reader.close()

    if integrated_frames < 2:
        raise RuntimeError("ROI reintegration produced fewer than two frames")
    texture_poses = _write_colmap_model(colmap_root, calibration, accepted_poses)
    if not texture_poses and not args.skip_openmvs:
        raise RuntimeError("ROI reintegration produced no usable texture keyframe")
    write_json(
        output_root / "trajectory.json",
        {
            "coordinate_system": "Open3D camera-to-world",
            "roi_min": roi_min.tolist(),
            "roi_max": roi_max.tolist(),
            "poses": [asdict(pose) for pose in accepted_poses],
        },
    )

    print("Extracting 1mm ROI mesh...", flush=True)
    mesh = model.extract_trianglemesh(
        args.surface_weight_threshold, args.estimated_vertex_count
    ).to_legacy()
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(roi_min, roi_max))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if args.roi_keep_largest_component and len(mesh.triangles) > 0:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters_array = np.asarray(clusters)
        counts_array = np.asarray(counts)
        largest = int(np.argmax(counts_array))
        mesh.remove_triangles_by_mask(clusters_array != largest)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh_dir = output_root / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    raw_mesh = mesh_dir / "open3d_mesh_1mm_roi.ply"
    if not o3d.io.write_triangle_mesh(
        str(raw_mesh), mesh, write_ascii=False, compressed=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"Could not write ROI mesh: {raw_mesh}")
    raw_stats = validate_mesh(raw_mesh)
    texturing_mesh, texturing_stats = _prepare_texturing_mesh(
        raw_mesh, output_root, args.texture_max_triangles, o3d
    )
    selected_poses: list[PoseRecord] = []
    textured_mesh = None
    textures: list[Path] = []
    if not args.skip_openmvs:
        selected_poses = select_texture_poses(
            texture_poses,
            image_root,
            mesh.get_axis_aligned_bounding_box().get_center(),
            args.texture_view_count,
            args.texture_min_view_cosine,
            np,
            cv2,
        )
        textured_mesh, textures = run_openmvs_texturing(
            output_root,
            texturing_mesh,
            colmap_root,
            args.openmvs_dir.expanduser().resolve(),
            runner,
            selected_poses,
            args.texture_cost_smoothness_ratio,
        )
    payload = {
        "status": "complete",
        "mode": "roi-reintegration",
        "input": str(mkv_path),
        "source_trajectory": str(source_root),
        "output": str(output_root),
        "roi_min": roi_min.tolist(),
        "roi_max": roi_max.tolist(),
        "roi_source_frame_min": args.roi_source_frame_min,
        "roi_source_frame_max": args.roi_source_frame_max,
        "voxel_size": args.roi_voxel_size,
        "integrated_frames": integrated_frames,
        "skipped_visibility_frames": skipped_visibility,
        "roi_pixels_min_median_max": [
            int(min(roi_pixel_counts)),
            float(np.median(roi_pixel_counts)),
            int(max(roi_pixel_counts)),
        ],
        "texture_keyframes": len(texture_poses),
        "texture_views_selected": len(selected_poses),
        "raw_mesh": str(raw_mesh),
        "raw_mesh_stats": raw_stats,
        "texturing_mesh": str(texturing_mesh),
        "texturing_mesh_stats": texturing_stats,
        "textured_mesh": str(textured_mesh) if textured_mesh else None,
        "textured_mesh_stats": validate_mesh(textured_mesh) if textured_mesh else None,
        "texture_files": [str(path) for path in textures],
        "commands": [record.to_dict() for record in runner.records],
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output_root / "run.json", payload)
    return payload


def run_fusion(args: argparse.Namespace) -> dict[str, Any]:
    mkv_path = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    k4a_bin = args.k4a_bin.expanduser().resolve()
    openmvs_dir = args.openmvs_dir.expanduser().resolve()
    if not mkv_path.is_file():
        raise FileNotFoundError(f"MKV not found: {mkv_path}")
    if not k4a_bin.is_dir():
        raise FileNotFoundError(f"K4A runtime directory not found: {k4a_bin}")
    if args.texture_frame_step % args.frame_step != 0:
        raise ValueError("texture-frame-step must be a multiple of frame-step")
    ensure_new_output(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    runner = CommandRunner(logs / "pipeline.log", timeout_seconds=args.timeout)
    started = time.monotonic()

    # Open3D imports its K4A bridge lazily, but Windows must also retain an
    # AddDllDirectory handle so dependencies adjacent to k4a.dll are found.
    os.environ["K4A_LIB_DIR"] = str(k4a_bin)
    os.environ["PATH"] = str(k4a_bin) + os.pathsep + os.environ.get("PATH", "")
    dll_directory_handle = (
        os.add_dll_directory(str(k4a_bin)) if os.name == "nt" else None
    )

    calibration_path = extract_k4a_calibration(
        mkv_path, output_root / "native" / "calibration.json"
    )
    calibration = derive_color_calibration(
        calibration_path, args.color_width, args.color_height, alpha=args.undistort_alpha
    )
    write_json(output_root / "calibration.json", asdict(calibration))

    np, cv2, o3d = _lazy_imports()

    full_intrinsic = np.asarray(calibration.intrinsic, dtype=np.float64)
    full_undistorted = np.asarray(calibration.undistorted_intrinsic, dtype=np.float64)
    full_distortion = np.asarray(calibration.distortion, dtype=np.float64)
    full_map_x, full_map_y = cv2.initUndistortRectifyMap(
        full_intrinsic,
        full_distortion,
        None,
        full_undistorted,
        (calibration.width, calibration.height),
        cv2.CV_32FC1,
    )

    tracking_width = args.tracking_width
    tracking_height = round(calibration.height * tracking_width / calibration.width)
    scale_x = tracking_width / calibration.width
    scale_y = tracking_height / calibration.height
    track_intrinsic = _scaled_intrinsic(full_intrinsic, scale_x, scale_y, np)
    track_undistorted = _scaled_intrinsic(full_undistorted, scale_x, scale_y, np)
    track_map_x, track_map_y = cv2.initUndistortRectifyMap(
        track_intrinsic,
        full_distortion,
        None,
        track_undistorted,
        (tracking_width, tracking_height),
        cv2.CV_32FC1,
    )

    device = o3d.core.Device(args.device)
    intrinsic_tensor = o3d.core.Tensor(track_undistorted, o3d.core.Dtype.Float64)
    transform = o3d.core.Tensor(np.identity(4), o3d.core.Dtype.Float64, device)
    model = o3d.t.pipelines.slam.Model(
        args.voxel_size, 16, args.block_count, transform, device
    )
    input_frame = o3d.t.pipelines.slam.Frame(
        tracking_height, tracking_width, intrinsic_tensor, device
    )
    raycast_frame = o3d.t.pipelines.slam.Frame(
        tracking_height, tracking_width, intrinsic_tensor, device
    )

    reader = o3d.io.AzureKinectMKVReader()
    if not reader.open(str(mkv_path)):
        raise RuntimeError(f"Open3D could not open MKV: {mkv_path}")

    colmap_root = output_root / "colmap"
    image_root = colmap_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    poses: list[PoseRecord] = []
    source_frame = -1
    sampled_frames = 0
    integrated_frames = 0
    rejected_frames = 0
    tracking_errors: list[dict[str, object]] = []

    try:
        while not reader.is_eof():
            rgbd = reader.next_frame()
            source_frame += 1
            if rgbd is None or source_frame % args.frame_step != 0:
                continue
            if args.max_frames is not None and sampled_frames >= args.max_frames:
                break
            sampled_frames += 1

            color_full = np.asarray(rgbd.color)
            depth_full = np.asarray(rgbd.depth)
            if color_full.size == 0 or depth_full.size == 0:
                rejected_frames += 1
                continue
            color_small = cv2.resize(
                color_full, (tracking_width, tracking_height), interpolation=cv2.INTER_AREA
            )
            depth_small = cv2.resize(
                depth_full, (tracking_width, tracking_height), interpolation=cv2.INTER_NEAREST
            )
            color_track = cv2.remap(
                color_small,
                track_map_x,
                track_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            depth_track = cv2.remap(
                depth_small,
                track_map_x,
                track_map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            depth_track = np.ascontiguousarray(depth_track.astype(np.uint16, copy=False))
            color_track = np.ascontiguousarray(color_track.astype(np.uint8, copy=False))
            input_frame.set_data_from_image(
                "depth", o3d.t.geometry.Image(o3d.core.Tensor(depth_track, device=device))
            )
            input_frame.set_data_from_image(
                "color", o3d.t.geometry.Image(o3d.core.Tensor(color_track, device=device))
            )

            fitness = 1.0
            inlier_rmse = 0.0
            candidate = transform
            accepted = integrated_frames == 0
            if integrated_frames > 0:
                try:
                    result = model.track_frame_to_model(
                        input_frame,
                        raycast_frame,
                        args.depth_scale,
                        args.depth_max,
                        args.odometry_distance_threshold,
                    )
                except RuntimeError as exc:
                    rejected_frames += 1
                    tracking_errors.append(
                        {"source_frame": source_frame, "message": str(exc)}
                    )
                    print(
                        f"sample={sampled_frames} source={source_frame} tracking_error={exc}",
                        flush=True,
                    )
                    continue
                fitness = float(result.fitness)
                inlier_rmse = float(result.inlier_rmse)
                delta = result.transformation.cpu().numpy()
                translation = float(np.linalg.norm(delta[:3, 3]))
                rotation = _rotation_angle_degrees(delta, np)
                accepted = (
                    np.isfinite(delta).all()
                    and fitness >= args.min_fitness
                    and translation <= args.max_step_translation
                    and rotation <= args.max_step_rotation
                )
                if accepted:
                    candidate = transform @ result.transformation

            if not accepted:
                rejected_frames += 1
                if sampled_frames % args.progress_interval == 0:
                    print(
                        f"sample={sampled_frames} source={source_frame} rejected "
                        f"fitness={fitness:.4f} rmse={inlier_rmse:.5f}",
                        flush=True,
                    )
                continue

            transform = candidate
            pose_array = transform.cpu().numpy()
            texture_image = None
            if source_frame % args.texture_frame_step == 0:
                color_undistorted = cv2.remap(
                    color_full,
                    full_map_x,
                    full_map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                texture_image = f"frame_{source_frame:06d}.jpg"
                cv2.imwrite(
                    str(image_root / texture_image),
                    cv2.cvtColor(color_undistorted, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
            poses.append(
                PoseRecord(
                    source_frame=source_frame,
                    timestamp_seconds=source_frame / args.source_fps,
                    camera_to_world=pose_array.tolist(),
                    fitness=fitness,
                    inlier_rmse=inlier_rmse,
                    texture_image=texture_image,
                )
            )
            model.update_frame_pose(integrated_frames, transform)
            model.integrate(
                input_frame, args.depth_scale, args.depth_max, args.trunc_voxel_multiplier
            )
            model.synthesize_model_frame(
                raycast_frame,
                args.depth_scale,
                args.depth_min,
                args.depth_max,
                args.trunc_voxel_multiplier,
                False,
            )
            integrated_frames += 1
            if sampled_frames % args.progress_interval == 0:
                print(
                    f"sample={sampled_frames} source={source_frame} integrated={integrated_frames} "
                    f"rejected={rejected_frames} fitness={fitness:.4f} rmse={inlier_rmse:.5f}",
                    flush=True,
                )
    finally:
        reader.close()

    if integrated_frames < 2:
        raise RuntimeError("Open3D tracking produced fewer than two integrated frames")

    texture_poses = _write_colmap_model(colmap_root, calibration, poses)
    if not texture_poses:
        raise RuntimeError("No accepted pose coincided with a texture keyframe")
    write_json(
        output_root / "trajectory.json",
        {"coordinate_system": "Open3D camera-to-world", "poses": [asdict(pose) for pose in poses]},
    )

    print("Extracting Open3D TSDF mesh...", flush=True)
    mesh = model.extract_trianglemesh(
        args.surface_weight_threshold, args.estimated_vertex_count
    ).to_legacy()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if args.max_triangles is not None and len(mesh.triangles) > args.max_triangles:
        mesh = mesh.simplify_quadric_decimation(args.max_triangles)
    if args.min_cluster_triangles > 0 and len(mesh.triangles) > 0:
        clusters, cluster_counts, _ = mesh.cluster_connected_triangles()
        clusters_array = np.asarray(clusters)
        counts_array = np.asarray(cluster_counts)
        remove_mask = counts_array[clusters_array] < args.min_cluster_triangles
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    mesh_dir = output_root / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    raw_mesh = mesh_dir / "open3d_mesh.ply"
    if not o3d.io.write_triangle_mesh(
        str(raw_mesh), mesh, write_ascii=False, compressed=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"Could not write Open3D mesh: {raw_mesh}")
    raw_mesh_stats = validate_mesh(raw_mesh)
    texturing_mesh, texturing_mesh_stats = _prepare_texturing_mesh(
        raw_mesh, output_root, args.texture_max_triangles, o3d
    )
    selected_poses: list[PoseRecord] = []

    textured_mesh = None
    texture_files: list[Path] = []
    if not args.skip_openmvs:
        selected_poses = select_texture_poses(
            texture_poses,
            image_root,
            mesh.get_axis_aligned_bounding_box().get_center(),
            args.texture_view_count,
            args.texture_min_view_cosine,
            np,
            cv2,
        )
        textured_mesh, texture_files = run_openmvs_texturing(
            output_root,
            texturing_mesh,
            colmap_root,
            openmvs_dir,
            runner,
            selected_poses,
            args.texture_cost_smoothness_ratio,
        )

    result_payload = {
        "status": "complete",
        "input": str(mkv_path),
        "output": str(output_root),
        "sampled_frames": sampled_frames,
        "integrated_frames": integrated_frames,
        "rejected_frames": rejected_frames,
        "tracking_errors": tracking_errors,
        "texture_keyframes": len(texture_poses),
        "texture_views_selected": len(selected_poses),
        "tracking_resolution": [tracking_width, tracking_height],
        "calibration": asdict(calibration),
        "raw_mesh": str(raw_mesh),
        "raw_mesh_stats": raw_mesh_stats,
        "texturing_mesh": str(texturing_mesh),
        "texturing_mesh_stats": texturing_mesh_stats,
        "textured_mesh": str(textured_mesh) if textured_mesh else None,
        "texture_files": [str(path) for path in texture_files],
        "commands": [record.to_dict() for record in runner.records],
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output_root / "run.json", result_payload)
    return result_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rgbd-fusion",
        description="Open3D KinectFusion-style RGB-D SLAM and OpenMVS texturing",
    )
    parser.add_argument("input", type=Path, help="K4A-compatible RGB-D MKV")
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--k4a-bin", type=Path, default=DEFAULT_K4A_BIN)
    parser.add_argument("--openmvs-dir", type=Path, default=DEFAULT_OPENMVS_DIR)
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--texture-frame-step", type=int, default=15)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--tracking-width", type=int, default=640)
    parser.add_argument("--color-width", type=int, default=1920)
    parser.add_argument("--color-height", type=int, default=1080)
    parser.add_argument("--undistort-alpha", type=float, default=0.0)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--block-count", type=int, default=20000)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-min", type=float, default=0.25)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--trunc-voxel-multiplier", type=float, default=8.0)
    parser.add_argument("--odometry-distance-threshold", type=float, default=0.07)
    parser.add_argument("--min-fitness", type=float, default=0.05)
    parser.add_argument("--max-step-translation", type=float, default=0.25)
    parser.add_argument("--max-step-rotation", type=float, default=45.0)
    parser.add_argument("--surface-weight-threshold", type=float, default=3.0)
    parser.add_argument("--estimated-vertex-count", type=int, default=1000000)
    parser.add_argument("--min-cluster-triangles", type=int, default=100)
    parser.add_argument("--max-triangles", type=int, default=500000)
    parser.add_argument("--texture-max-triangles", type=int, default=100000)
    parser.add_argument(
        "--texture-view-count",
        type=int,
        default=36,
        help="maximum sharp, directionally distributed texture views; 0 uses all",
    )
    parser.add_argument("--texture-min-view-cosine", type=float, default=0.55)
    parser.add_argument(
        "--texture-cost-smoothness-ratio", type=float, default=1.0
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="CPU:0")
    parser.add_argument("--skip-openmvs", action="store_true")
    parser.add_argument("--resume-texture", action="store_true")
    parser.add_argument("--roi-from", type=Path)
    parser.add_argument(
        "--roi-min", type=float, nargs=3, default=(-0.22, -0.24, 0.28)
    )
    parser.add_argument(
        "--roi-max", type=float, nargs=3, default=(0.14, 0.0, 0.68)
    )
    parser.add_argument("--roi-voxel-size", type=float, default=0.001)
    parser.add_argument("--roi-block-count", type=int, default=12000)
    parser.add_argument("--roi-trunc-voxel-multiplier", type=float, default=5.0)
    parser.add_argument("--roi-min-pixels", type=int, default=500)
    parser.add_argument("--roi-source-frame-min", type=int)
    parser.add_argument("--roi-source-frame-max", type=int)
    parser.add_argument(
        "--roi-keep-largest-component",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.resume_texture:
            result = resume_texturing(args)
        elif args.roi_from is not None:
            result = run_roi_reintegration(args)
        else:
            result = run_fusion(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
