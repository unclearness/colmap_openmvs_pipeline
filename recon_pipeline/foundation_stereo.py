from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from recon_pipeline.artifacts import ensure_new_output, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "tools"
    / "FoundationStereo-NGC-2.0"
    / "deployable_foundation_stereo_s_dynamic_v2.0.onnx"
)
DEFAULT_MODEL_SHA256 = (
    "a001a7bc0512a0bc3b3218194e924784e58b20656c6f1ea2c151024e555cfd64"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ColmapImage:
    image_id: int
    quaternion: tuple[float, float, float, float]
    translation: tuple[float, float, float]
    camera_id: int
    name: str
    point3d_ids: frozenset[int]
    observations: tuple[tuple[float, float, int], ...]


@dataclass(frozen=True, slots=True)
class StereoPair:
    reference: ColmapImage
    source: ColmapImage
    shared_points: int
    baseline: float
    median_depth: float
    baseline_ratio: float
    view_angle_degrees: float


@dataclass(slots=True)
class RectifiedPair:
    left: ColmapImage
    right: ColmapImage
    left_bgr: Any
    right_bgr: Any
    left_valid: Any
    right_valid: Any
    model_left_bgr: Any
    model_right_bgr: Any
    model_left_valid: Any
    model_right_valid: Any
    left_rectification: Any
    left_projection: Any
    right_projection: Any
    reprojection: Any
    vertical: bool


@dataclass(slots=True)
class DepthView:
    pair: RectifiedPair
    depth: Any
    valid: Any
    record_index: int
    artifact_dir: Path | None
    normal_world: Any | None = None


def _lazy_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "FoundationStereo requires Python 3.12 with numpy, "
            "opencv-python-headless, onnxruntime-gpu, and open3d. See README.md."
        ) from exc
    return np, cv2, ort, o3d


def _data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_colmap_cameras(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in _data_lines(path):
        fields = line.split()
        camera = ColmapCamera(
            camera_id=int(fields[0]),
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            params=tuple(float(value) for value in fields[4:]),
        )
        cameras[camera.camera_id] = camera
    if not cameras:
        raise RuntimeError(f"COLMAP model contains no cameras: {path}")
    return cameras


def read_colmap_images(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        while True:
            pose_line = stream.readline()
            if not pose_line:
                break
            if not pose_line.strip() or pose_line.lstrip().startswith("#"):
                continue
            fields = pose_line.split()
            if len(fields) < 10:
                raise ValueError(f"Invalid COLMAP image record in {path}: {pose_line}")
            observations = stream.readline()
            if observations == "":
                observations = "\n"
            observation_fields = observations.split()
            observations = tuple(
                (
                    float(observation_fields[index]),
                    float(observation_fields[index + 1]),
                    int(observation_fields[index + 2]),
                )
                for index in range(0, len(observation_fields), 3)
                if int(observation_fields[index + 2]) >= 0
            )
            point3d_ids = frozenset(observation[2] for observation in observations)
            image = ColmapImage(
                image_id=int(fields[0]),
                quaternion=tuple(float(value) for value in fields[1:5]),
                translation=tuple(float(value) for value in fields[5:8]),
                camera_id=int(fields[8]),
                name=" ".join(fields[9:]),
                point3d_ids=point3d_ids,
                observations=observations,
            )
            images[image.image_id] = image
    if not images:
        raise RuntimeError(f"COLMAP model contains no registered images: {path}")
    return images


def read_colmap_points(path: Path) -> dict[int, tuple[float, float, float]]:
    points: dict[int, tuple[float, float, float]] = {}
    for line in _data_lines(path):
        fields = line.split()
        points[int(fields[0])] = tuple(float(value) for value in fields[1:4])
    if not points:
        raise RuntimeError(f"COLMAP model contains no sparse points: {path}")
    return points


def quaternion_to_rotation(quaternion: Sequence[float], np: Any) -> Any:
    qw, qx, qy, qz = quaternion
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0:
        raise ValueError("Quaternion has zero norm")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.asarray(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def camera_matrix_and_distortion(camera: ColmapCamera, np: Any) -> tuple[Any, Any]:
    model = camera.model.upper()
    params = camera.params
    if model == "SIMPLE_PINHOLE":
        focal, cx, cy = params
        fx = fy = focal
        distortion: tuple[float, ...] = ()
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        distortion = ()
    elif model == "SIMPLE_RADIAL":
        focal, cx, cy, k1 = params
        fx = fy = focal
        distortion = (k1, 0.0, 0.0, 0.0)
    elif model == "RADIAL":
        focal, cx, cy, k1, k2 = params
        fx = fy = focal
        distortion = (k1, k2, 0.0, 0.0)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        distortion = (k1, k2, p1, p2)
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy, *distortion = params
    else:
        raise ValueError(
            f"Unsupported camera model for OpenCV stereo rectification: {model}"
        )
    intrinsic = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return intrinsic, np.asarray(distortion, dtype=np.float64)


def camera_center(image: ColmapImage, np: Any) -> Any:
    rotation = quaternion_to_rotation(image.quaternion, np)
    translation = np.asarray(image.translation, dtype=np.float64)
    return -rotation.T @ translation


def _visible_depths(
    image: ColmapImage,
    points: dict[int, tuple[float, float, float]],
    np: Any,
) -> Any:
    coordinates = [points[point_id] for point_id in image.point3d_ids if point_id in points]
    if not coordinates:
        return np.empty((0,), dtype=np.float64)
    world = np.asarray(coordinates, dtype=np.float64)
    rotation = quaternion_to_rotation(image.quaternion, np)
    translation = np.asarray(image.translation, dtype=np.float64)
    camera_points = (rotation @ world.T).T + translation
    return camera_points[camera_points[:, 2] > 0, 2]


def camera_orbit_angles(
    images: Sequence[ColmapImage], np: Any
) -> tuple[Any, Any, dict[int, float]]:
    centers = np.asarray([camera_center(image, np) for image in images])
    forwards = np.asarray(
        [
            quaternion_to_rotation(image.quaternion, np).T
            @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            for image in images
        ]
    )
    system = np.zeros((3, 3), dtype=np.float64)
    right_hand_side = np.zeros(3, dtype=np.float64)
    for center, forward in zip(centers, forwards):
        projector = np.identity(3) - np.outer(forward, forward)
        system += projector
        right_hand_side += projector @ center
    target = np.linalg.pinv(system) @ right_hand_side
    centered = centers - target
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] <= 1e-9:
        raise RuntimeError("Camera centers do not span a usable orbit plane")
    first_axis, second_axis = axes[0], axes[1]
    angles = {
        image.image_id: float(
            math.atan2(
                np.dot(center - target, second_axis),
                np.dot(center - target, first_axis),
            )
            % (2.0 * math.pi)
        )
        for image, center in zip(images, centers)
    }
    return target, axes[2], angles


def spatially_uniform_reference_order(
    images: Sequence[ColmapImage],
    angles: dict[int, float],
    *,
    bin_count: int,
    references_per_bin: int,
) -> list[ColmapImage]:
    if bin_count < 1 or references_per_bin < 1:
        raise ValueError("spatial bins and references-per-bin must be positive")
    bins: list[list[tuple[float, ColmapImage]]] = [[] for _ in range(bin_count)]
    bin_width = 2.0 * math.pi / bin_count
    for image in images:
        angle = angles[image.image_id]
        bin_index = min(int(angle / bin_width), bin_count - 1)
        center = (bin_index + 0.5) * bin_width
        distance = abs((angle - center + math.pi) % (2.0 * math.pi) - math.pi)
        bins[bin_index].append((distance, image))
    for members in bins:
        members.sort(key=lambda item: (item[0], -len(item[1].point3d_ids), item[1].name))

    selected: list[ColmapImage] = []
    selected_ids: set[int] = set()
    for rank in range(references_per_bin):
        for members in bins:
            if rank < len(members):
                image = members[rank][1]
                selected.append(image)
                selected_ids.add(image.image_id)
    # Keep remaining cameras as a fallback after every occupied direction has
    # received the requested number of representatives.
    for members in bins:
        for _, image in members:
            if image.image_id not in selected_ids:
                selected.append(image)
    return selected


def _select_diverse_source_candidates(
    candidates: Sequence[tuple[float, float, StereoPair]], count: int
) -> list[tuple[float, float, StereoPair]]:
    if count < 1:
        return []
    ordered = sorted(candidates, key=lambda item: item[0], reverse=True)
    chosen: list[tuple[float, float, StereoPair]] = []
    if count >= 2:
        negative = [candidate for candidate in ordered if candidate[1] < 0]
        positive = [candidate for candidate in ordered if candidate[1] >= 0]
        if negative and positive:
            chosen.extend((negative[0], positive[0]))
    for candidate in ordered:
        if candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) >= count:
            break
    return chosen[:count]


def select_stereo_pairs(
    images: dict[int, ColmapImage],
    points: dict[int, tuple[float, float, float]],
    *,
    reference_step: int,
    sources_per_reference: int,
    min_shared_points: int,
    min_baseline_ratio: float,
    max_baseline_ratio: float,
    target_baseline_ratio: float,
    max_view_angle_degrees: float,
    max_pairs: int | None,
    np: Any,
    spatial_bins: int = 0,
    references_per_bin: int = 1,
) -> list[StereoPair]:
    if reference_step < 1 or sources_per_reference < 1:
        raise ValueError("reference-step and sources-per-reference must be positive")
    ordered = sorted(images.values(), key=lambda image: image.name.lower())
    rotations = {
        image.image_id: quaternion_to_rotation(image.quaternion, np) for image in ordered
    }
    centers = {image.image_id: camera_center(image, np) for image in ordered}
    forwards = {
        image.image_id: rotations[image.image_id].T
        @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        for image in ordered
    }
    _, _, orbit_angles = camera_orbit_angles(ordered, np)
    references = (
        spatially_uniform_reference_order(
            ordered,
            orbit_angles,
            bin_count=spatial_bins,
            references_per_bin=references_per_bin,
        )
        if spatial_bins > 0
        else ordered[::reference_step]
    )
    selected: list[StereoPair] = []
    used_unordered: set[tuple[int, int]] = set()
    for reference in references:
        depths = _visible_depths(reference, points, np)
        if len(depths) < min_shared_points:
            continue
        median_depth = float(np.median(depths))
        candidates: list[tuple[float, float, StereoPair]] = []
        for source in ordered:
            if source.image_id == reference.image_id:
                continue
            pair_key = tuple(sorted((reference.image_id, source.image_id)))
            if pair_key in used_unordered:
                continue
            shared = len(reference.point3d_ids & source.point3d_ids)
            if shared < min_shared_points:
                continue
            baseline = float(
                np.linalg.norm(centers[reference.image_id] - centers[source.image_id])
            )
            ratio = baseline / median_depth
            if not min_baseline_ratio <= ratio <= max_baseline_ratio:
                continue
            view_cosine = float(
                np.clip(
                    np.dot(forwards[reference.image_id], forwards[source.image_id]),
                    -1.0,
                    1.0,
                )
            )
            angle = math.degrees(math.acos(view_cosine))
            if angle > max_view_angle_degrees:
                continue
            ratio_score = math.exp(
                -abs(math.log(max(ratio, 1e-9) / target_baseline_ratio))
            )
            angle_score = math.exp(-((angle / max_view_angle_degrees) ** 2))
            pair = StereoPair(
                reference=reference,
                source=source,
                shared_points=shared,
                baseline=baseline,
                median_depth=median_depth,
                baseline_ratio=ratio,
                view_angle_degrees=angle,
            )
            source_offset = (
                orbit_angles[source.image_id]
                - orbit_angles[reference.image_id]
                + math.pi
            ) % (2.0 * math.pi) - math.pi
            candidates.append(
                (math.log1p(shared) * ratio_score * angle_score, source_offset, pair)
            )
        chosen = _select_diverse_source_candidates(
            candidates, sources_per_reference
        )
        for _, _, pair in chosen:
            selected.append(pair)
            used_unordered.add(
                tuple(sorted((pair.reference.image_id, pair.source.image_id)))
            )
            if max_pairs is not None and len(selected) >= max_pairs:
                return selected
    if not selected:
        raise RuntimeError(
            "No suitable stereo pairs. Relax shared-point, baseline-ratio, or "
            "view-angle thresholds."
        )
    return selected


def rectify_pair(
    pair: StereoPair,
    cameras: dict[int, ColmapCamera],
    image_root: Path,
    output_size: tuple[int, int],
    np: Any,
    cv2: Any,
    *,
    allow_swap: bool = True,
    rectification_alpha: float = 1.0,
) -> RectifiedPair:
    left_camera = cameras[pair.reference.camera_id]
    right_camera = cameras[pair.source.camera_id]
    if (left_camera.width, left_camera.height) != (
        right_camera.width,
        right_camera.height,
    ):
        raise ValueError("Stereo pair cameras must have equal image dimensions")
    left_path = image_root / pair.reference.name
    right_path = image_root / pair.source.name
    left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left_image is None or right_image is None:
        raise FileNotFoundError(f"Could not read stereo images: {left_path}, {right_path}")
    expected_shape = (left_camera.height, left_camera.width)
    if left_image.shape[:2] != expected_shape or right_image.shape[:2] != expected_shape:
        raise ValueError(
            f"Image dimensions do not match COLMAP camera {expected_shape}: "
            f"{left_path}={left_image.shape[:2]}, {right_path}={right_image.shape[:2]}"
        )

    left_intrinsic, left_distortion = camera_matrix_and_distortion(left_camera, np)
    right_intrinsic, right_distortion = camera_matrix_and_distortion(right_camera, np)
    left_rotation = quaternion_to_rotation(pair.reference.quaternion, np)
    right_rotation = quaternion_to_rotation(pair.source.quaternion, np)
    left_translation = np.asarray(pair.reference.translation, dtype=np.float64)
    right_translation = np.asarray(pair.source.translation, dtype=np.float64)
    relative_rotation = right_rotation @ left_rotation.T
    relative_translation = (
        right_translation - relative_rotation @ left_translation
    ).reshape(3, 1)
    rectification = cv2.stereoRectify(
        left_intrinsic,
        left_distortion.reshape(-1, 1),
        right_intrinsic,
        right_distortion.reshape(-1, 1),
        (left_camera.width, left_camera.height),
        relative_rotation,
        relative_translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=rectification_alpha,
        newImageSize=output_size,
    )
    left_rectification, right_rectification, left_projection, right_projection, reprojection = (
        rectification[:5]
    )
    horizontal = abs(float(right_projection[0, 3])) >= abs(
        float(right_projection[1, 3])
    )
    axis = 0 if horizontal else 1
    if float(right_projection[axis, 3]) > 0:
        if not allow_swap:
            raise RuntimeError("Could not establish positive stereo disparity ordering")
        swapped = StereoPair(
            reference=pair.source,
            source=pair.reference,
            shared_points=pair.shared_points,
            baseline=pair.baseline,
            median_depth=pair.median_depth,
            baseline_ratio=pair.baseline_ratio,
            view_angle_degrees=pair.view_angle_degrees,
        )
        return rectify_pair(
            swapped,
            cameras,
            image_root,
            output_size,
            np,
            cv2,
            allow_swap=False,
            rectification_alpha=rectification_alpha,
        )

    map_left = cv2.initUndistortRectifyMap(
        left_intrinsic,
        left_distortion,
        left_rectification,
        left_projection[:, :3],
        output_size,
        cv2.CV_32FC1,
    )
    map_right = cv2.initUndistortRectifyMap(
        right_intrinsic,
        right_distortion,
        right_rectification,
        right_projection[:, :3],
        output_size,
        cv2.CV_32FC1,
    )
    left_bgr = cv2.remap(
        left_image, *map_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    right_bgr = cv2.remap(
        right_image,
        *map_right,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    source_mask = np.full(expected_shape, 255, dtype=np.uint8)
    left_valid = cv2.remap(
        source_mask,
        *map_left,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    right_valid = cv2.remap(
        source_mask,
        *map_right,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    vertical = not horizontal
    if vertical:
        model_left_bgr = np.ascontiguousarray(np.rot90(left_bgr, 1))
        model_right_bgr = np.ascontiguousarray(np.rot90(right_bgr, 1))
        model_left_valid = np.ascontiguousarray(np.rot90(left_valid, 1))
        model_right_valid = np.ascontiguousarray(np.rot90(right_valid, 1))
    else:
        model_left_bgr = left_bgr
        model_right_bgr = right_bgr
        model_left_valid = left_valid
        model_right_valid = right_valid
    return RectifiedPair(
        left=pair.reference,
        right=pair.source,
        left_bgr=left_bgr,
        right_bgr=right_bgr,
        left_valid=left_valid,
        right_valid=right_valid,
        model_left_bgr=model_left_bgr,
        model_right_bgr=model_right_bgr,
        model_left_valid=model_left_valid,
        model_right_valid=model_right_valid,
        left_rectification=left_rectification,
        left_projection=left_projection,
        right_projection=right_projection,
        reprojection=reprojection,
        vertical=vertical,
    )


class FoundationStereoRunner:
    def __init__(self, model_path: Path, provider: str, device_id: int, np: Any, ort: Any):
        self.np = np
        ort.set_default_logger_severity(3)
        if provider == "cuda":
            ort.preload_dlls(directory="")
            providers: list[Any] = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
        if provider == "cuda" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"CUDA provider was requested but is not active: {self.session.get_providers()}"
            )
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def infer(self, bgr_left: Any, bgr_right: Any) -> Any:
        def prepare(image: Any) -> Any:
            rgb = image[..., ::-1].astype(self.np.float32) / 255.0
            normalized = (rgb - self.mean) / self.std
            return self.np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])

        left = prepare(bgr_left)
        right = prepare(bgr_right)
        disparity = self.session.run(
            ["disparity"], {"left_image": left, "right_image": right}
        )[0]
        return self.np.asarray(disparity[0, 0], dtype=self.np.float32)


def disparity_with_consistency(
    runner: FoundationStereoRunner,
    pair: RectifiedPair,
    *,
    lr_check: bool,
    lr_threshold: float,
    min_disparity: float,
    max_disparity: float,
    np: Any,
    cv2: Any,
) -> tuple[Any, Any]:
    disparity = runner.infer(pair.model_left_bgr, pair.model_right_bgr)
    valid = (
        np.isfinite(disparity)
        & (disparity >= min_disparity)
        & (disparity <= max_disparity)
        & pair.model_left_valid
    )
    height, width = disparity.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    right_x = grid_x - disparity
    right_in_bounds = (right_x >= 0) & (right_x <= width - 1)
    sampled_right_valid = cv2.remap(
        pair.model_right_valid.astype(np.uint8),
        right_x.astype(np.float32),
        grid_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    valid &= right_in_bounds & sampled_right_valid
    if lr_check:
        reverse = runner.infer(
            np.ascontiguousarray(pair.model_right_bgr[:, ::-1]),
            np.ascontiguousarray(pair.model_left_bgr[:, ::-1]),
        )[:, ::-1]
        sampled_reverse = cv2.remap(
            reverse,
            right_x.astype(np.float32),
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid &= np.isfinite(sampled_reverse) & (
            np.abs(disparity - sampled_reverse) <= lr_threshold
        )
    if pair.vertical:
        disparity = np.ascontiguousarray(np.rot90(disparity, 3))
        valid = np.ascontiguousarray(np.rot90(valid, 3))
    return disparity, valid


def calibrate_disparity_from_sparse_points(
    disparity: Any,
    valid: Any,
    pair: RectifiedPair,
    points: dict[int, tuple[float, float, float]],
    np: Any,
    cv2: Any,
) -> tuple[Any, dict[str, float | int | None]]:
    shared_ids = pair.left.point3d_ids & pair.right.point3d_ids
    coordinates = [points[point_id] for point_id in shared_ids if point_id in points]
    if not coordinates:
        return disparity, {
            "samples": 0,
            "scale": 1.0,
            "median_relative_error": None,
            "p90_relative_error": None,
        }
    world = np.asarray(coordinates, dtype=np.float64)
    rotation = quaternion_to_rotation(pair.left.quaternion, np)
    translation = np.asarray(pair.left.translation, dtype=np.float64)
    camera = (rotation @ world.T).T + translation
    rectified = (pair.left_rectification @ camera.T).T
    positive = rectified[:, 2] > 0
    rectified = rectified[positive]
    if len(rectified) == 0:
        return disparity, {
            "samples": 0,
            "scale": 1.0,
            "median_relative_error": None,
            "p90_relative_error": None,
        }
    projection = pair.left_projection[:, :3]
    pixels = (projection @ rectified.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    axis = 1 if pair.vertical else 0
    expected = np.abs(float(pair.right_projection[axis, 3])) / rectified[:, 2]
    sample_x = pixels[:, 0]
    sample_y = pixels[:, 1]
    height, width = disparity.shape
    in_bounds = (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
        & np.isfinite(expected)
        & (expected > 0)
    )
    if not np.any(in_bounds):
        return disparity, {
            "samples": 0,
            "scale": 1.0,
            "median_relative_error": None,
            "p90_relative_error": None,
        }
    predicted = cv2.remap(
        disparity,
        sample_x[in_bounds].astype(np.float32).reshape(-1, 1),
        sample_y[in_bounds].astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).reshape(-1)
    sampled_valid = cv2.remap(
        valid.astype(np.uint8),
        sample_x[in_bounds].astype(np.float32).reshape(-1, 1),
        sample_y[in_bounds].astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).reshape(-1) > 0
    expected = expected[in_bounds]
    usable = sampled_valid & np.isfinite(predicted) & (predicted > 0.5)
    predicted = predicted[usable]
    expected = expected[usable]
    if len(predicted) == 0:
        return disparity, {
            "samples": 0,
            "scale": 1.0,
            "median_relative_error": None,
            "p90_relative_error": None,
        }
    ratios = expected / predicted
    scale = float(np.exp(np.median(np.log(np.clip(ratios, 1e-6, 1e6)))))
    relative_error = np.abs(predicted * scale - expected) / expected
    corrected = np.asarray(disparity * scale, dtype=np.float32)
    return corrected, {
        "samples": int(len(predicted)),
        "scale": scale,
        "median_relative_error": float(np.median(relative_error)),
        "p90_relative_error": float(np.percentile(relative_error, 90.0)),
    }


def _rectified_sparse_depth(
    pair: RectifiedPair,
    points: dict[int, tuple[float, float, float]],
    np: Any,
) -> float:
    coordinates = [
        points[point_id] for point_id in pair.left.point3d_ids if point_id in points
    ]
    world = np.asarray(coordinates, dtype=np.float64)
    rotation = quaternion_to_rotation(pair.left.quaternion, np)
    translation = np.asarray(pair.left.translation, dtype=np.float64)
    camera = (rotation @ world.T).T + translation
    rectified = (pair.left_rectification @ camera.T).T
    positive = rectified[rectified[:, 2] > 0, 2]
    if len(positive) == 0:
        raise RuntimeError(f"No sparse depth in rectified view {pair.left.name}")
    return float(np.median(positive))


def disparity_to_depth(
    disparity: Any,
    valid: Any,
    pair: RectifiedPair,
    sparse_median_depth: float,
    near_factor: float,
    far_factor: float,
    np: Any,
    cv2: Any,
) -> tuple[Any, Any]:
    points = cv2.reprojectImageTo3D(disparity, pair.reprojection)
    depth = np.asarray(points[..., 2], dtype=np.float32)
    valid &= (
        np.isfinite(depth)
        & (depth >= sparse_median_depth * near_factor)
        & (depth <= sparse_median_depth * far_factor)
    )
    depth[~valid] = 0.0
    return depth, valid


def _world_to_rectified_camera(image: ColmapImage, rectification: Any, np: Any) -> Any:
    rotation = quaternion_to_rotation(image.quaternion, np)
    translation = np.asarray(image.translation, dtype=np.float64)
    extrinsic = np.identity(4, dtype=np.float64)
    extrinsic[:3, :3] = rectification @ rotation
    extrinsic[:3, 3] = rectification @ translation
    return extrinsic


def estimate_world_normals_from_depth(
    view: DepthView,
    *,
    relative_discontinuity: float,
    absolute_discontinuity: float,
    np: Any,
) -> Any:
    depth = np.asarray(view.depth, dtype=np.float64)
    height, width = depth.shape
    projection = view.pair.left_projection
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    points = np.stack(
        (
            (pixel_x - projection[0, 2]) * depth / projection[0, 0],
            (pixel_y - projection[1, 2]) * depth / projection[1, 1],
            depth,
        ),
        axis=-1,
    )
    delta_x = np.zeros_like(points)
    delta_y = np.zeros_like(points)
    delta_x[:, 1:-1] = points[:, 2:] - points[:, :-2]
    delta_y[1:-1, :] = points[2:, :] - points[:-2, :]
    normals_camera = np.cross(delta_x, delta_y)
    lengths = np.linalg.norm(normals_camera, axis=2)

    neighbor_valid = np.zeros_like(view.valid)
    neighbor_valid[1:-1, 1:-1] = (
        view.valid[1:-1, 1:-1]
        & view.valid[1:-1, :-2]
        & view.valid[1:-1, 2:]
        & view.valid[:-2, 1:-1]
        & view.valid[2:, 1:-1]
    )
    center = depth[1:-1, 1:-1]
    tolerance = absolute_discontinuity + relative_discontinuity * center
    continuous = np.zeros_like(view.valid)
    continuous[1:-1, 1:-1] = (
        np.abs(depth[1:-1, :-2] - center) <= tolerance
    ) & (
        np.abs(depth[1:-1, 2:] - center) <= tolerance
    ) & (
        np.abs(depth[:-2, 1:-1] - center) <= tolerance
    ) & (
        np.abs(depth[2:, 1:-1] - center) <= tolerance
    )
    usable = neighbor_valid & continuous & np.isfinite(lengths) & (lengths > 1e-12)
    normals_camera[usable] /= lengths[usable, None]
    normals_camera[~usable] = 0.0

    extrinsic = _world_to_rectified_camera(
        view.pair.left, view.pair.left_rectification, np
    )
    normals_world = normals_camera @ extrinsic[:3, :3]
    world = (
        extrinsic[:3, :3].T
        @ (points.reshape(-1, 3) - extrinsic[:3, 3]).T
    ).T.reshape(height, width, 3)
    camera = camera_center(view.pair.left, np)
    toward_camera = np.sum(normals_world * (camera - world), axis=2)
    normals_world[toward_camera < 0] *= -1.0
    normals_world[~usable] = 0.0
    return normals_world.astype(np.float32)


def multiview_consistency_mask(
    view: DepthView,
    other_views: Sequence[DepthView],
    *,
    min_support_views: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    normal_cosine_threshold: float | None = None,
    np: Any,
    cv2: Any,
) -> Any:
    if min_support_views <= 0:
        return view.valid.copy()
    rows, columns = np.nonzero(view.valid & (view.depth > 0))
    if len(rows) == 0:
        return np.zeros_like(view.valid)
    depth = view.depth[rows, columns].astype(np.float64)
    projection = view.pair.left_projection
    rectified_points = np.column_stack(
        (
            (columns - projection[0, 2]) * depth / projection[0, 0],
            (rows - projection[1, 2]) * depth / projection[1, 1],
            depth,
        )
    )
    reference_extrinsic = _world_to_rectified_camera(
        view.pair.left, view.pair.left_rectification, np
    )
    world = (
        reference_extrinsic[:3, :3].T
        @ (rectified_points - reference_extrinsic[:3, 3]).T
    ).T
    support = np.zeros(len(world), dtype=np.uint16)
    reference_normals = (
        view.normal_world[rows, columns]
        if normal_cosine_threshold is not None and view.normal_world is not None
        else None
    )
    reference_center = camera_center(view.pair.left, np)
    ordered_others = sorted(
        (other for other in other_views if other is not view),
        key=lambda other: float(
            np.linalg.norm(camera_center(other.pair.left, np) - reference_center)
        ),
    )
    for other in ordered_others:
        active = np.flatnonzero(support < min_support_views)
        if len(active) == 0:
            break
        active_world = world[active]
        extrinsic = _world_to_rectified_camera(
            other.pair.left, other.pair.left_rectification, np
        )
        camera = (extrinsic[:3, :3] @ active_world.T).T + extrinsic[:3, 3]
        positive = camera[:, 2] > 0
        other_projection = other.pair.left_projection
        sample_x = (
            other_projection[0, 0] * camera[:, 0] / camera[:, 2]
            + other_projection[0, 2]
        )
        sample_y = (
            other_projection[1, 1] * camera[:, 1] / camera[:, 2]
            + other_projection[1, 2]
        )
        height, width = other.depth.shape
        in_bounds = (
            positive
            & (sample_x >= 0)
            & (sample_x <= width - 1)
            & (sample_y >= 0)
            & (sample_y <= height - 1)
        )
        sampled_depth = np.zeros(len(active_world), dtype=np.float32)
        bounded_x = sample_x[in_bounds]
        bounded_y = sample_y[in_bounds]
        x0 = np.floor(bounded_x).astype(np.int64)
        y0 = np.floor(bounded_y).astype(np.int64)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        wx = bounded_x - x0
        wy = bounded_y - y0
        neighbor_depths = np.stack(
            (
                other.depth[y0, x0],
                other.depth[y0, x1],
                other.depth[y1, x0],
                other.depth[y1, x1],
            ),
            axis=1,
        )
        weights = np.stack(
            (
                (1.0 - wx) * (1.0 - wy),
                wx * (1.0 - wy),
                (1.0 - wx) * wy,
                wx * wy,
            ),
            axis=1,
        )
        weights *= neighbor_depths > 0
        weight_sum = weights.sum(axis=1)
        interpolated = np.divide(
            (neighbor_depths * weights).sum(axis=1),
            weight_sum,
            out=np.zeros_like(weight_sum, dtype=np.float64),
            where=weight_sum > 1e-6,
        )
        sampled_depth[in_bounds] = interpolated.astype(np.float32)
        tolerance = absolute_tolerance + relative_tolerance * camera[:, 2]
        consistent = (
            in_bounds
            & (sampled_depth > 0)
            & np.isfinite(sampled_depth)
            & (np.abs(sampled_depth - camera[:, 2]) <= tolerance)
        )
        if (
            normal_cosine_threshold is not None
            and reference_normals is not None
            and other.normal_world is not None
        ):
            sampled_normal = np.zeros((len(active_world), 3), dtype=np.float32)
            nearest_columns = np.clip(
                np.rint(sample_x[in_bounds]).astype(np.int64), 0, width - 1
            )
            nearest_rows = np.clip(
                np.rint(sample_y[in_bounds]).astype(np.int64), 0, height - 1
            )
            sampled_normal[in_bounds] = other.normal_world[
                nearest_rows, nearest_columns
            ]
            sampled_length = np.linalg.norm(sampled_normal, axis=1)
            normal_dot = np.sum(
                reference_normals[active] * sampled_normal, axis=1
            )
            consistent &= (
                sampled_length > 0.5
            ) & (normal_dot >= normal_cosine_threshold)
        support[active[consistent]] += 1
    mask = np.zeros_like(view.valid)
    mask[rows, columns] = support >= min_support_views
    return mask


def _write_pair_artifacts(
    root: Path,
    index: int,
    pair: RectifiedPair,
    disparity: Any,
    depth: Any,
    valid: Any,
    np: Any,
    cv2: Any,
) -> Path:
    name = f"{index:04d}_{Path(pair.left.name).stem}_{Path(pair.right.name).stem}"
    destination = root / "pairs" / name
    destination.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination / "left.png"), pair.left_bgr)
    cv2.imwrite(str(destination / "right.png"), pair.right_bgr)
    np.save(destination / "disparity.npy", disparity)
    np.save(destination / "depth.npy", depth)
    cv2.imwrite(str(destination / "valid.png"), valid.astype(np.uint8) * 255)
    finite = disparity[valid]
    if len(finite):
        low, high = np.percentile(finite, (2.0, 98.0))
        scale = max(float(high - low), 1e-6)
        preview = np.clip((disparity - low) / scale * 255.0, 0, 255).astype(np.uint8)
        preview[~valid] = 0
        preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(destination / "disparity.png"), preview)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.colmap_model.expanduser().resolve()
    image_root = args.images.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    colmap_fused = (
        args.colmap_fused.expanduser().resolve()
        if args.colmap_fused is not None
        else None
    )
    required_inputs = [
        model_dir / "cameras.txt",
        model_dir / "images.txt",
        model_dir / "points3D.txt",
        model_path,
    ]
    if colmap_fused is not None:
        required_inputs.append(colmap_fused)
    for required in required_inputs:
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    ensure_new_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    np, cv2, ort, o3d = _lazy_imports()

    actual_hash = _sha256(model_path)
    expected_hash = args.model_sha256.lower() if args.model_sha256 else None
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            f"FoundationStereo SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    cameras = read_colmap_cameras(model_dir / "cameras.txt")
    images = read_colmap_images(model_dir / "images.txt")
    points = read_colmap_points(model_dir / "points3D.txt")
    _, _, orbit_angles = camera_orbit_angles(
        sorted(images.values(), key=lambda image: image.name.lower()), np
    )
    candidate_limit = args.max_pairs * 4 if args.max_pairs is not None else None
    pairs = select_stereo_pairs(
        images,
        points,
        reference_step=args.reference_step,
        sources_per_reference=args.sources_per_reference,
        min_shared_points=args.min_shared_points,
        min_baseline_ratio=args.min_baseline_ratio,
        max_baseline_ratio=args.max_baseline_ratio,
        target_baseline_ratio=args.target_baseline_ratio,
        max_view_angle_degrees=args.max_view_angle,
        max_pairs=candidate_limit,
        np=np,
        spatial_bins=args.spatial_bins,
        references_per_bin=args.references_per_bin,
    )
    pair_depths = [pair.median_depth for pair in pairs]
    scene_depth = float(np.median(np.asarray(pair_depths, dtype=np.float64)))
    voxel_size = args.voxel_size or scene_depth * args.voxel_relative
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * args.sdf_trunc_multiplier,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    runner = FoundationStereoRunner(
        model_path, args.provider, args.device_id, np, ort
    )
    pair_records: list[dict[str, Any]] = []
    depth_views: list[DepthView] = []
    for index, selected_pair in enumerate(pairs):
        if args.max_pairs is not None and len(depth_views) >= args.max_pairs:
            break
        pair_started = time.monotonic()
        rectified = rectify_pair(
            selected_pair,
            cameras,
            image_root,
            (args.inference_width, args.inference_height),
            np,
            cv2,
            rectification_alpha=args.rectification_alpha,
        )
        sparse_depth = _rectified_sparse_depth(rectified, points, np)
        disparity_axis = 1 if rectified.vertical else 0
        camera = cameras[rectified.left.camera_id]
        intrinsic, _ = camera_matrix_and_distortion(camera, np)
        scale = (
            args.inference_height / camera.height
            if rectified.vertical
            else args.inference_width / camera.width
        )
        nominal_focal = float(intrinsic[disparity_axis, disparity_axis]) * scale
        rectified_focal = float(
            rectified.left_projection[disparity_axis, disparity_axis]
        )
        focal_ratio = rectified_focal / nominal_focal
        expected_disparity = rectified_focal * selected_pair.baseline / sparse_depth
        if not (
            args.min_rectified_focal_ratio
            <= focal_ratio
            <= args.max_rectified_focal_ratio
        ) or expected_disparity > args.max_expected_disparity:
            pair_records.append(
                {
                    **asdict(selected_pair),
                    "reference": rectified.left.name,
                    "source": rectified.right.name,
                    "reference_azimuth_degrees": math.degrees(
                        orbit_angles[rectified.left.image_id]
                    ),
                    "source_azimuth_degrees": math.degrees(
                        orbit_angles[rectified.right.image_id]
                    ),
                    "vertical_rectification": rectified.vertical,
                    "sparse_median_rectified_depth": sparse_depth,
                    "rectified_focal_ratio": focal_ratio,
                    "expected_disparity": expected_disparity,
                    "valid_pixels": 0,
                    "valid_fraction": 0.0,
                    "integrated": False,
                    "skip_reason": "rectification_geometry",
                    "elapsed_seconds": time.monotonic() - pair_started,
                }
            )
            print(
                f"candidate {index + 1}/{len(pairs)} {rectified.left.name} + "
                f"{rectified.right.name}: skipped focal_ratio={focal_ratio:.2f} "
                f"expected_disp={expected_disparity:.1f}",
                flush=True,
            )
            continue
        disparity, valid = disparity_with_consistency(
            runner,
            rectified,
            lr_check=args.lr_check,
            lr_threshold=args.lr_threshold,
            min_disparity=args.min_disparity,
            max_disparity=args.max_disparity,
            np=np,
            cv2=cv2,
        )
        disparity, sparse_calibration = calibrate_disparity_from_sparse_points(
            disparity, valid, rectified, points, np, cv2
        )
        valid &= (
            np.isfinite(disparity)
            & (disparity >= args.min_disparity)
            & (disparity <= args.max_disparity)
        )
        depth, valid = disparity_to_depth(
            disparity,
            valid,
            rectified,
            sparse_depth,
            args.near_depth_factor,
            args.far_depth_factor,
            np,
            cv2,
        )
        valid_pixels = int(valid.sum())
        artifact_dir = None
        if args.save_pairs:
            artifact_dir = _write_pair_artifacts(
                output_root, index, rectified, disparity, depth, valid, np, cv2
            )
        record = {
            **asdict(selected_pair),
            "reference": rectified.left.name,
            "source": rectified.right.name,
            "reference_azimuth_degrees": math.degrees(
                orbit_angles[rectified.left.image_id]
            ),
            "source_azimuth_degrees": math.degrees(
                orbit_angles[rectified.right.image_id]
            ),
            "vertical_rectification": rectified.vertical,
            "sparse_median_rectified_depth": sparse_depth,
            "rectified_focal_ratio": focal_ratio,
            "expected_disparity": expected_disparity,
            "sparse_disparity_calibration": sparse_calibration,
            "valid_pixels": valid_pixels,
            "valid_fraction": valid_pixels / valid.size,
            "accepted": valid_pixels >= args.min_valid_pixels,
            "integrated": False,
            "elapsed_seconds": time.monotonic() - pair_started,
        }
        pair_records.append(record)
        if valid_pixels >= args.min_valid_pixels:
            depth_views.append(
                DepthView(
                    pair=rectified,
                    depth=depth,
                    valid=valid,
                    record_index=len(pair_records) - 1,
                    artifact_dir=artifact_dir,
                )
            )
        print(
            f"candidate {index + 1}/{len(pairs)} {rectified.left.name} + "
            f"{rectified.right.name}: valid={valid_pixels / valid.size:.1%} "
            f"vertical={rectified.vertical}",
            flush=True,
        )

    if not depth_views:
        raise RuntimeError("No FoundationStereo depth map passed the pair filter")
    if args.depth_only:
        result = {
            "status": "complete",
            "mode": "colmap-sfm-foundationstereo-depth-only",
            "colmap_model": str(model_dir),
            "images": str(image_root),
            "output": str(output_root),
            "model": str(model_path),
            "model_sha256": actual_hash,
            "provider": runner.session.get_providers()[0],
            "registered_images": len(images),
            "sparse_points": len(points),
            "candidate_pairs": len(pairs),
            "selected_pairs": len(pair_records),
            "accepted_pairs": len(depth_views),
            "integrated_pairs": 0,
            "spatial_bins": args.spatial_bins,
            "references_per_bin": args.references_per_bin,
            "sources_per_reference": args.sources_per_reference,
            "inference_size": [args.inference_width, args.inference_height],
            "rectification_alpha": args.rectification_alpha,
            "lr_check": args.lr_check,
            "multiview_consistency": False,
            "voxel_size": voxel_size,
            "pairs": pair_records,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output_root / "run.json", result)
        return result
    normal_cosine_threshold = (
        math.cos(math.radians(args.normal_consistency_angle))
        if args.normal_consistency_angle > 0
        else None
    )
    if normal_cosine_threshold is not None:
        for view in depth_views:
            view.normal_world = estimate_world_normals_from_depth(
                view,
                relative_discontinuity=args.normal_discontinuity_relative,
                absolute_discontinuity=voxel_size
                * args.normal_discontinuity_voxel_multiplier,
                np=np,
            )
    integrated = 0
    for view_index, view in enumerate(depth_views):
        if args.multiview_consistency and len(depth_views) > 1:
            consistent = multiview_consistency_mask(
                view,
                depth_views,
                min_support_views=args.min_consistent_views,
                relative_tolerance=args.consistency_relative_tolerance,
                absolute_tolerance=voxel_size
                * args.consistency_absolute_voxel_multiplier,
                normal_cosine_threshold=normal_cosine_threshold,
                np=np,
                cv2=cv2,
            )
        else:
            consistent = view.valid
        consistent_pixels = int(consistent.sum())
        record = pair_records[view.record_index]
        record["consistent_pixels"] = consistent_pixels
        record["consistent_fraction"] = consistent_pixels / consistent.size
        if view.artifact_dir is not None:
            cv2.imwrite(
                str(view.artifact_dir / "consistent.png"),
                consistent.astype(np.uint8) * 255,
            )
        if consistent_pixels < args.min_valid_pixels:
            record["skip_reason"] = "multiview_consistency"
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
            f"fusion {view_index + 1}/{len(depth_views)} {view.pair.left.name}: "
            f"consistent={consistent_pixels / consistent.size:.1%}",
            flush=True,
        )
    if integrated == 0:
        raise RuntimeError("No depth map passed multi-view consistency")
    mesh_dir = output_root / "mesh"
    dense_dir = output_root / "dense"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    raw_mesh_path = mesh_dir / "foundationstereo_tsdf_raw.ply"
    if not o3d.io.write_triangle_mesh(
        str(raw_mesh_path), mesh, write_ascii=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"Could not write raw TSDF mesh: {raw_mesh_path}")
    if args.keep_largest_component and len(mesh.triangles):
        clusters, counts, _ = mesh.cluster_connected_triangles()
        cluster_array = np.asarray(clusters)
        count_array = np.asarray(counts)
        mesh.remove_triangles_by_mask(cluster_array != int(np.argmax(count_array)))
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    cloud_path = dense_dir / "foundationstereo_fused.ply"
    mesh_path = mesh_dir / "foundationstereo_tsdf.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False):
        raise RuntimeError(f"Could not write fused point cloud: {cloud_path}")
    if not o3d.io.write_triangle_mesh(
        str(mesh_path), mesh, write_ascii=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"Could not write TSDF mesh: {mesh_path}")
    hybrid_path = None
    hybrid_points = None
    if colmap_fused is not None:
        colmap_cloud = o3d.io.read_point_cloud(str(colmap_fused))
        if len(colmap_cloud.points) == 0:
            raise RuntimeError(f"COLMAP fused cloud is empty: {colmap_fused}")
        hybrid = (colmap_cloud + cloud).voxel_down_sample(voxel_size)
        hybrid_path = dense_dir / "hybrid_colmap_foundationstereo_fused.ply"
        if not o3d.io.write_point_cloud(
            str(hybrid_path), hybrid, write_ascii=False
        ):
            raise RuntimeError(f"Could not write hybrid point cloud: {hybrid_path}")
        hybrid_points = len(hybrid.points)
    result = {
        "status": "complete",
        "mode": "colmap-sfm-foundationstereo-tsdf",
        "colmap_model": str(model_dir),
        "images": str(image_root),
        "output": str(output_root),
        "model": str(model_path),
        "model_sha256": actual_hash,
        "provider": runner.session.get_providers()[0],
        "registered_images": len(images),
        "sparse_points": len(points),
        "candidate_pairs": len(pairs),
        "selected_pairs": len(pair_records),
        "integrated_pairs": integrated,
        "spatial_bins": args.spatial_bins,
        "references_per_bin": args.references_per_bin,
        "sources_per_reference": args.sources_per_reference,
        "integrated_reference_azimuth_degrees": [
            record["reference_azimuth_degrees"]
            for record in pair_records
            if record.get("integrated")
        ],
        "inference_size": [args.inference_width, args.inference_height],
        "rectification_alpha": args.rectification_alpha,
        "lr_check": args.lr_check,
        "multiview_consistency": args.multiview_consistency,
        "min_consistent_views": args.min_consistent_views,
        "consistency_relative_tolerance": args.consistency_relative_tolerance,
        "normal_consistency_angle": args.normal_consistency_angle,
        "voxel_size": voxel_size,
        "cloud": str(cloud_path),
        "cloud_points": len(cloud.points),
        "hybrid_cloud": str(hybrid_path) if hybrid_path else None,
        "hybrid_cloud_points": hybrid_points,
        "raw_mesh": str(raw_mesh_path),
        "mesh": str(mesh_path),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "pairs": pair_records,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output_root / "run.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundation-stereo-dense",
        description=(
            "Experimental COLMAP SfM + two-view FoundationStereo + TSDF fusion"
        ),
    )
    parser.add_argument("colmap_model", type=Path, help="COLMAP text model directory")
    parser.add_argument("images", type=Path, help="images referenced by images.txt")
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument(
        "--colmap-fused",
        type=Path,
        help="optional existing COLMAP fused.ply to union with the neural cloud",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--inference-width", type=int, default=544)
    parser.add_argument("--inference-height", type=int, default=960)
    parser.add_argument(
        "--rectification-alpha",
        type=float,
        default=1.0,
        help="OpenCV stereoRectify alpha; 1 preserves the full source field of view",
    )
    parser.add_argument(
        "--reference-step",
        type=int,
        default=4,
        help="filename-order fallback stride when --spatial-bins=0",
    )
    parser.add_argument("--spatial-bins", type=int, default=48)
    parser.add_argument("--references-per-bin", type=int, default=4)
    parser.add_argument("--sources-per-reference", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=96)
    parser.add_argument("--min-shared-points", type=int, default=30)
    parser.add_argument("--min-baseline-ratio", type=float, default=0.003)
    parser.add_argument("--max-baseline-ratio", type=float, default=0.08)
    parser.add_argument("--target-baseline-ratio", type=float, default=0.03)
    parser.add_argument("--max-view-angle", type=float, default=10.0)
    parser.add_argument("--min-rectified-focal-ratio", type=float, default=0.25)
    parser.add_argument("--max-rectified-focal-ratio", type=float, default=4.0)
    parser.add_argument("--max-expected-disparity", type=float, default=200.0)
    parser.add_argument(
        "--lr-check", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lr-threshold", type=float, default=1.5)
    parser.add_argument(
        "--multiview-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-consistent-views", type=int, default=3)
    parser.add_argument(
        "--consistency-relative-tolerance", type=float, default=0.01
    )
    parser.add_argument(
        "--consistency-absolute-voxel-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--normal-consistency-angle",
        type=float,
        default=25.0,
        help="maximum world-normal disagreement in degrees; 0 disables",
    )
    parser.add_argument(
        "--normal-discontinuity-relative", type=float, default=0.03
    )
    parser.add_argument(
        "--normal-discontinuity-voxel-multiplier", type=float, default=2.0
    )
    parser.add_argument("--min-disparity", type=float, default=0.5)
    parser.add_argument("--max-disparity", type=float, default=416.0)
    parser.add_argument("--near-depth-factor", type=float, default=0.4)
    parser.add_argument("--far-depth-factor", type=float, default=1.8)
    parser.add_argument("--min-valid-pixels", type=int, default=10000)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--voxel-relative", type=float, default=0.0015)
    parser.add_argument("--sdf-trunc-multiplier", type=float, default=5.0)
    parser.add_argument(
        "--keep-largest-component",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-pairs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="save accepted pair depth maps without running TSDF fusion",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.inference_width <= 0 or args.inference_height <= 0:
        raise ValueError("inference width and height must be positive")
    if args.inference_width % 32 or args.inference_height % 32:
        raise ValueError("inference width and height must be multiples of 32")
    if not -1.0 <= args.rectification_alpha <= 1.0:
        raise ValueError("rectification-alpha must be between -1 and 1")
    for name in (
        "reference_step",
        "sources_per_reference",
        "min_shared_points",
        "min_valid_pixels",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("max-pairs must be positive, or 0 to process all candidates")
    if args.spatial_bins < 0:
        raise ValueError("spatial-bins must not be negative")
    if args.references_per_bin < 1:
        raise ValueError("references-per-bin must be positive")
    if not (
        0 < args.min_baseline_ratio
        <= args.target_baseline_ratio
        <= args.max_baseline_ratio
    ):
        raise ValueError(
            "baseline ratios must satisfy 0 < min <= target <= max"
        )
    if not 0 < args.max_view_angle < 180:
        raise ValueError("max-view-angle must be between 0 and 180 degrees")
    if not (
        0 < args.min_rectified_focal_ratio <= args.max_rectified_focal_ratio
    ):
        raise ValueError("rectified focal ratios must satisfy 0 < min <= max")
    if not 0 < args.max_expected_disparity <= args.max_disparity:
        raise ValueError(
            "max-expected-disparity must be positive and no greater than max-disparity"
        )
    if not 0 < args.min_disparity < args.max_disparity:
        raise ValueError("disparity limits must satisfy 0 < min < max")
    if args.lr_threshold <= 0:
        raise ValueError("lr-threshold must be positive")
    if not 0 < args.near_depth_factor < args.far_depth_factor:
        raise ValueError("depth factors must satisfy 0 < near < far")
    if args.voxel_size is not None and args.voxel_size <= 0:
        raise ValueError("voxel-size must be positive")
    if args.voxel_relative <= 0 or args.sdf_trunc_multiplier <= 0:
        raise ValueError("voxel-relative and sdf-trunc-multiplier must be positive")
    if args.min_consistent_views < 0:
        raise ValueError("min-consistent-views must not be negative")
    if not 0 <= args.normal_consistency_angle <= 90:
        raise ValueError("normal-consistency-angle must be between 0 and 90")
    if (
        args.normal_discontinuity_relative < 0
        or args.normal_discontinuity_voxel_multiplier < 0
    ):
        raise ValueError("normal discontinuity tolerances must not be negative")
    if (
        args.consistency_relative_tolerance < 0
        or args.consistency_absolute_voxel_multiplier < 0
    ):
        raise ValueError("multi-view consistency tolerances must not be negative")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.max_pairs == 0:
            args.max_pairs = None
        validate_arguments(args)
        result = run_experiment(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
