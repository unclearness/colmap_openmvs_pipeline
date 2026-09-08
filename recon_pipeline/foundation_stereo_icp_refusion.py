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
    DepthView,
    _lazy_imports,
    _world_to_rectified_camera,
    camera_center,
    estimate_world_normals_from_depth,
    multiview_consistency_mask,
    read_colmap_cameras,
    read_colmap_images,
    read_colmap_points,
)
from recon_pipeline.foundation_stereo_refusion import load_saved_depth_views


def _camera_to_world(view: DepthView, np: Any) -> Any:
    return np.linalg.inv(
        _world_to_rectified_camera(
            view.pair.left, view.pair.left_rectification, np
        )
    )


def select_icp_edges(
    camera_to_world: Sequence[Any],
    *,
    neighbors: int,
    max_view_angle_degrees: float,
    np: Any,
) -> list[tuple[int, int, float]]:
    centers = np.asarray([pose[:3, 3] for pose in camera_to_world])
    forwards = np.asarray([pose[:3, 2] for pose in camera_to_world])
    forwards /= np.linalg.norm(forwards, axis=1, keepdims=True)
    selected: dict[tuple[int, int], float] = {}
    for source in range(len(camera_to_world)):
        delta = centers - centers[source]
        distances = np.linalg.norm(delta, axis=1)
        cosines = np.clip(forwards @ forwards[source], -1.0, 1.0)
        angles = np.degrees(np.arccos(cosines))
        order = np.argsort(distances)
        accepted = 0
        for target in order:
            if target == source or angles[target] > max_view_angle_degrees:
                continue
            edge = (min(source, int(target)), max(source, int(target)))
            selected.setdefault(edge, float(distances[target]))
            accepted += 1
            if accepted >= neighbors:
                break
    return [
        (source, target, distance)
        for (source, target), distance in sorted(
            selected.items(), key=lambda item: (item[1], item[0])
        )
    ]


def clamp_pose_correction(
    initial: Any,
    optimized: Any,
    *,
    max_translation: float,
    max_rotation_degrees: float,
    np: Any,
    cv2: Any,
) -> tuple[Any, float, float, bool]:
    result = np.asarray(optimized, dtype=np.float64).copy()
    translation_delta = result[:3, 3] - initial[:3, 3]
    translation = float(np.linalg.norm(translation_delta))
    clamped = False
    if translation > max_translation:
        result[:3, 3] = (
            initial[:3, 3]
            + translation_delta * (max_translation / translation)
        )
        clamped = True
    rotation_delta = result[:3, :3] @ initial[:3, :3].T
    rotation_vector, _ = cv2.Rodrigues(rotation_delta)
    angle = float(np.linalg.norm(rotation_vector))
    rotation_degrees = math.degrees(angle)
    if rotation_degrees > max_rotation_degrees and angle > 1e-12:
        limited_vector = rotation_vector * (
            math.radians(max_rotation_degrees) / angle
        )
        limited_rotation, _ = cv2.Rodrigues(limited_vector)
        result[:3, :3] = limited_rotation @ initial[:3, :3]
        clamped = True
    return result, translation, rotation_degrees, clamped


def _depth_point_cloud(
    view: DepthView,
    mask: Any,
    *,
    voxel_size: float,
    np: Any,
    o3d: Any,
) -> Any:
    rows, columns = np.nonzero(mask & np.isfinite(view.depth) & (view.depth > 0))
    depth = view.depth[rows, columns].astype(np.float64)
    projection = view.pair.left_projection
    points = np.column_stack(
        (
            (columns - projection[0, 2]) * depth / projection[0, 0],
            (rows - projection[1, 2]) * depth / projection[1, 1],
            depth,
        )
    )
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel_size)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4.0, max_nn=40
        )
    )
    return cloud


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True


def optimize_view_poses(
    views: Sequence[DepthView],
    clouds: Sequence[Any],
    args: argparse.Namespace,
    np: Any,
    cv2: Any,
    o3d: Any,
) -> tuple[list[Any], dict[str, Any]]:
    initial_poses = [_camera_to_world(view, np) for view in views]
    candidates = select_icp_edges(
        initial_poses,
        neighbors=args.icp_neighbors,
        max_view_angle_degrees=args.icp_max_view_angle,
        np=np,
    )
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for pose in initial_poses:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
    accepted_edges: list[dict[str, Any]] = []
    disjoint = _DisjointSet(len(views))
    max_correspondence = args.icp_max_correspondence or (
        args.icp_voxel_size * 4.0
    )
    for edge_index, (source, target, camera_distance) in enumerate(candidates, start=1):
        initial = np.linalg.inv(initial_poses[target]) @ initial_poses[source]
        if disjoint.union(source, target):
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source,
                    target,
                    initial,
                    np.identity(6, dtype=np.float64)
                    * args.pose_prior_information,
                    uncertain=False,
                )
            )
        registration = o3d.pipelines.registration.registration_icp(
            clouds[source],
            clouds[target],
            max_correspondence,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=args.icp_iterations,
            ),
        )
        fitness = float(registration.fitness)
        rmse = float(registration.inlier_rmse)
        measurement_delta = registration.transformation @ np.linalg.inv(initial)
        measurement_translation = float(
            np.linalg.norm(measurement_delta[:3, 3])
        )
        measurement_cosine = np.clip(
            (np.trace(measurement_delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0
        )
        measurement_rotation = float(
            np.degrees(np.arccos(measurement_cosine))
        )
        accepted = (
            fitness >= args.icp_min_fitness
            and rmse <= args.icp_max_rmse
            and measurement_translation <= args.max_icp_translation
            and measurement_rotation <= args.max_icp_rotation
        )
        record = {
            "source": source,
            "target": target,
            "source_image": views[source].pair.left.name,
            "target_image": views[target].pair.left.name,
            "camera_distance": camera_distance,
            "fitness": fitness,
            "rmse": rmse,
            "accepted": accepted,
            "certain": False,
            "measurement_translation": measurement_translation,
            "measurement_rotation_degrees": measurement_rotation,
        }
        if accepted:
            information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                clouds[source],
                clouds[target],
                max_correspondence,
                registration.transformation,
            )
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source,
                    target,
                    registration.transformation,
                    information,
                    uncertain=True,
                )
            )
        accepted_edges.append(record)
        if edge_index % 25 == 0:
            print(
                f"ICP edge {edge_index}/{len(candidates)} accepted="
                f"{sum(item['accepted'] for item in accepted_edges)}",
                flush=True,
            )
    certain_edges = sum(not edge.uncertain for edge in pose_graph.edges)
    if certain_edges < len(views) - 1:
        raise RuntimeError(
            f"Pose-prior graph is too sparse: {certain_edges} edges for {len(views)} views"
        )
    roots = {disjoint.find(index) for index in range(len(views))}
    if len(roots) != 1:
        raise RuntimeError(f"ICP graph has {len(roots)} disconnected components")

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence,
            edge_prune_threshold=args.edge_prune_threshold,
            preference_loop_closure=args.loop_closure_preference,
            reference_node=0,
        ),
    )
    optimized_poses: list[Any] = []
    corrections: list[dict[str, Any]] = []
    for index, node in enumerate(pose_graph.nodes):
        pose, translation, rotation, clamped = clamp_pose_correction(
            initial_poses[index],
            np.asarray(node.pose),
            max_translation=args.max_pose_translation,
            max_rotation_degrees=args.max_pose_rotation,
            np=np,
            cv2=cv2,
        )
        optimized_poses.append(pose)
        corrections.append(
            {
                "index": index,
                "image": views[index].pair.left.name,
                "translation": translation,
                "rotation_degrees": rotation,
                "clamped": clamped,
                "camera_to_world": pose.tolist(),
            }
        )
    translation_values = np.asarray(
        [record["translation"] for record in corrections], dtype=np.float64
    )
    rotation_values = np.asarray(
        [record["rotation_degrees"] for record in corrections], dtype=np.float64
    )
    return optimized_poses, {
        "candidate_edges": len(candidates),
        "accepted_edges": sum(record["accepted"] for record in accepted_edges),
        "certain_edges": certain_edges,
        "edges": accepted_edges,
        "corrections": corrections,
        "translation_p50": float(np.percentile(translation_values, 50.0)),
        "translation_p90": float(np.percentile(translation_values, 90.0)),
        "rotation_p50_degrees": float(np.percentile(rotation_values, 50.0)),
        "rotation_p90_degrees": float(np.percentile(rotation_values, 90.0)),
        "clamped_poses": sum(record["clamped"] for record in corrections),
    }


def optimize_view_poses_leave_one_out(
    views: Sequence[DepthView],
    clouds: Sequence[Any],
    args: argparse.Namespace,
    np: Any,
    cv2: Any,
    o3d: Any,
) -> tuple[list[Any], dict[str, Any]]:
    if args.loo_reference_index >= len(views):
        raise ValueError("loo-reference-index is outside the filtered view list")
    initial_poses = [_camera_to_world(view, np) for view in views]
    poses = [pose.copy() for pose in initial_poses]

    def transformed_cloud(index: int) -> Any:
        result = o3d.geometry.PointCloud(clouds[index])
        result.transform(poses[index])
        return result

    world_clouds = [transformed_cloud(index) for index in range(len(clouds))]
    target_voxel = args.loo_target_voxel_size or args.icp_voxel_size * 2.0
    max_correspondence = args.icp_max_correspondence or args.icp_voxel_size * 4.0
    if args.loo_tukey_k is None:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=args.loo_tukey_k)
        )
    sweeps: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    for sweep in range(args.loo_iterations):
        accepted_updates = 0
        sweep_translation: list[float] = []
        sweep_rotation: list[float] = []
        for index in range(len(views)):
            if index == args.loo_reference_index:
                continue
            target_points = np.concatenate(
                [
                    np.asarray(cloud.points)
                    for other_index, cloud in enumerate(world_clouds)
                    if other_index != index and len(cloud.points)
                ],
                axis=0,
            )
            target_normals = np.concatenate(
                [
                    np.asarray(cloud.normals)
                    for other_index, cloud in enumerate(world_clouds)
                    if other_index != index and len(cloud.points)
                ],
                axis=0,
            )
            target = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(target_points)
            )
            target.normals = o3d.utility.Vector3dVector(target_normals)
            target = target.voxel_down_sample(target_voxel)
            if not target.has_normals():
                target.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=target_voxel * 4.0, max_nn=40
                    )
                )
            registration = o3d.pipelines.registration.registration_icp(
                clouds[index],
                target,
                max_correspondence,
                poses[index],
                estimation,
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-7,
                    relative_rmse=1e-7,
                    max_iteration=args.icp_iterations,
                ),
            )
            candidate, step_translation, step_rotation, step_clamped = clamp_pose_correction(
                poses[index],
                np.asarray(registration.transformation),
                max_translation=args.loo_max_step_translation,
                max_rotation_degrees=args.loo_max_step_rotation,
                np=np,
                cv2=cv2,
            )
            candidate, total_translation, total_rotation, total_clamped = clamp_pose_correction(
                initial_poses[index],
                candidate,
                max_translation=args.max_pose_translation,
                max_rotation_degrees=args.max_pose_rotation,
                np=np,
                cv2=cv2,
            )
            _, applied_translation, applied_rotation, _ = clamp_pose_correction(
                poses[index],
                candidate,
                max_translation=float("inf"),
                max_rotation_degrees=180.0,
                np=np,
                cv2=cv2,
            )
            fitness = float(registration.fitness)
            rmse = float(registration.inlier_rmse)
            accepted = fitness >= args.loo_min_fitness and rmse <= args.loo_max_rmse
            if accepted:
                poses[index] = candidate
                world_clouds[index] = transformed_cloud(index)
                accepted_updates += 1
                sweep_translation.append(applied_translation)
                sweep_rotation.append(applied_rotation)
            all_updates.append(
                {
                    "sweep": sweep + 1,
                    "index": index,
                    "image": views[index].pair.left.name,
                    "fitness": fitness,
                    "rmse": rmse,
                    "accepted": accepted,
                    "proposed_step_translation": step_translation,
                    "proposed_step_rotation_degrees": step_rotation,
                    "applied_step_translation": applied_translation,
                    "applied_step_rotation_degrees": applied_rotation,
                    "total_translation": total_translation,
                    "total_rotation_degrees": total_rotation,
                    "clamped": step_clamped or total_clamped,
                }
            )
            if (index + 1) % 10 == 0:
                print(
                    f"LOO sweep {sweep + 1}/{args.loo_iterations} "
                    f"view {index + 1}/{len(views)} accepted={accepted_updates}",
                    flush=True,
                )
        maximum_translation = max(sweep_translation, default=0.0)
        maximum_rotation = max(sweep_rotation, default=0.0)
        sweeps.append(
            {
                "sweep": sweep + 1,
                "accepted_updates": accepted_updates,
                "translation_p50": float(np.median(sweep_translation)) if sweep_translation else None,
                "rotation_p50_degrees": float(np.median(sweep_rotation)) if sweep_rotation else None,
                "maximum_translation": maximum_translation,
                "maximum_rotation_degrees": maximum_rotation,
            }
        )
        if (
            maximum_translation <= args.loo_translation_convergence
            and maximum_rotation <= args.loo_rotation_convergence
        ):
            break
    corrections: list[dict[str, Any]] = []
    for index, pose in enumerate(poses):
        _, translation, rotation, _ = clamp_pose_correction(
            initial_poses[index],
            pose,
            max_translation=float("inf"),
            max_rotation_degrees=180.0,
            np=np,
            cv2=cv2,
        )
        corrections.append(
            {
                "index": index,
                "image": views[index].pair.left.name,
                "translation": translation,
                "rotation_degrees": rotation,
                "camera_to_world": pose.tolist(),
            }
        )
    translation_values = np.asarray([record["translation"] for record in corrections])
    rotation_values = np.asarray([record["rotation_degrees"] for record in corrections])
    return poses, {
        "optimizer": "leave-one-out",
        "reference_index": args.loo_reference_index,
        "target_voxel_size": target_voxel,
        "tukey_k": args.loo_tukey_k,
        "sweeps": sweeps,
        "updates": all_updates,
        "corrections": corrections,
        "translation_p50": float(np.percentile(translation_values, 50.0)),
        "translation_p90": float(np.percentile(translation_values, 90.0)),
        "rotation_p50_degrees": float(np.percentile(rotation_values, 50.0)),
        "rotation_p90_degrees": float(np.percentile(rotation_values, 90.0)),
        "accepted_updates": sum(record["accepted"] for record in all_updates),
    }


def run_icp_refusion(args: argparse.Namespace) -> dict[str, Any]:
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
    np, cv2, _, o3d = _lazy_imports()
    cameras = read_colmap_cameras(model_dir / "cameras.txt")
    images = read_colmap_images(model_dir / "images.txt")
    read_colmap_points(model_dir / "points3D.txt")
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
    normal_cosine = math.cos(math.radians(args.normal_consistency_angle))
    filtered_views: list[DepthView] = []
    filtered_masks: list[Any] = []
    filtered_records: list[dict[str, Any]] = []
    mask_root = output_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(views, start=1):
        view.normal_world = estimate_world_normals_from_depth(
            view,
            relative_discontinuity=args.normal_discontinuity_relative,
            absolute_discontinuity=voxel_size
            * args.normal_discontinuity_voxel_multiplier,
            np=np,
        )
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
        if consistent_pixels >= args.min_valid_pixels:
            filtered_views.append(view)
            filtered_masks.append(consistent)
            filtered_records.append(record)
        else:
            record["skip_reason"] = "relaxed_depth_normal_consistency"
        print(
            f"mask {index}/{len(views)} {view.pair.left.name}: "
            f"{consistent_pixels / consistent.size:.1%}",
            flush=True,
        )
    if len(filtered_views) < 2:
        raise RuntimeError("Fewer than two views passed relaxed consistency")

    icp_voxel_size = args.icp_voxel_size or voxel_size * 2.0
    args.icp_voxel_size = icp_voxel_size
    clouds = [
        _depth_point_cloud(
            view,
            mask,
            voxel_size=icp_voxel_size,
            np=np,
            o3d=o3d,
        )
        for view, mask in zip(filtered_views, filtered_masks, strict=True)
    ]
    if args.optimizer == "leave-one-out":
        optimized_poses, optimization = optimize_view_poses_leave_one_out(
            filtered_views, clouds, args, np, cv2, o3d
        )
    else:
        optimized_poses, optimization = optimize_view_poses(
            filtered_views, clouds, args, np, cv2, o3d
        )
    write_json(output_root / "pose_graph.json", optimization)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * args.sdf_trunc_multiplier,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    integrated = 0
    for index, (view, mask, camera_to_world, record) in enumerate(
        zip(
            filtered_views,
            filtered_masks,
            optimized_poses,
            filtered_records,
            strict=True,
        ),
        start=1,
    ):
        filtered_depth = np.where(mask, view.depth, 0).astype(np.float32)
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
        volume.integrate(rgbd, intrinsic, np.linalg.inv(camera_to_world))
        record["integrated"] = True
        integrated += 1
        if index % 20 == 0:
            print(f"optimized TSDF {index}/{len(filtered_views)}", flush=True)

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
    mesh.compute_vertex_normals()
    cloud_path = dense_root / "foundationstereo_relaxed_icp_global.ply"
    mesh_path = mesh_root / "foundationstereo_relaxed_icp_global.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False):
        raise RuntimeError(f"Could not write optimized cloud: {cloud_path}")
    if not o3d.io.write_triangle_mesh(
        str(mesh_path), mesh, write_ascii=False, write_vertex_normals=True
    ):
        raise RuntimeError(f"Could not write optimized mesh: {mesh_path}")
    result = {
        "status": "complete",
        "mode": "foundationstereo-relaxed-depth-multiway-icp-refusion",
        "source_run": str(source_manifest),
        "output": str(output_root),
        "input_depth_views": len(views),
        "integrated_views": integrated,
        "voxel_size": voxel_size,
        "min_consistent_views": args.min_consistent_views,
        "consistency_relative_tolerance": args.consistency_relative_tolerance,
        "consistency_absolute_voxel_multiplier": args.consistency_absolute_voxel_multiplier,
        "normal_consistency_angle": args.normal_consistency_angle,
        "icp_voxel_size": icp_voxel_size,
        "optimization": {
            key: value
            for key, value in optimization.items()
            if key not in {"edges", "corrections", "updates"}
        },
        "pose_graph": str(output_root / "pose_graph.json"),
        "cloud": str(cloud_path),
        "cloud_points": len(cloud.points),
        "mesh": str(mesh_path),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "views": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output_root / "run.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundation-stereo-icp-refusion",
        description="Relaxed depth consistency + pose-graph/leave-one-out ICP refusion",
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-consistent-views", type=int, default=4)
    parser.add_argument("--consistency-relative-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--consistency-absolute-voxel-multiplier", type=float, default=0.5
    )
    parser.add_argument("--normal-consistency-angle", type=float, default=25.0)
    parser.add_argument("--normal-discontinuity-relative", type=float, default=0.03)
    parser.add_argument(
        "--normal-discontinuity-voxel-multiplier", type=float, default=2.0
    )
    parser.add_argument("--min-valid-pixels", type=int, default=5000)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--sdf-trunc-multiplier", type=float, default=5.0)
    parser.add_argument("--icp-voxel-size", type=float)
    parser.add_argument(
        "--optimizer", choices=("pose-graph", "leave-one-out"), default="pose-graph"
    )
    parser.add_argument("--icp-neighbors", type=int, default=5)
    parser.add_argument("--icp-max-view-angle", type=float, default=45.0)
    parser.add_argument("--icp-max-correspondence", type=float)
    parser.add_argument("--icp-iterations", type=int, default=50)
    parser.add_argument("--icp-min-fitness", type=float, default=0.10)
    parser.add_argument("--icp-max-rmse", type=float, default=0.30)
    parser.add_argument("--max-icp-translation", type=float, default=0.25)
    parser.add_argument("--max-icp-rotation", type=float, default=3.0)
    parser.add_argument("--pose-prior-information", type=float, default=1000000.0)
    parser.add_argument("--edge-prune-threshold", type=float, default=0.25)
    parser.add_argument("--loop-closure-preference", type=float, default=0.10)
    parser.add_argument("--max-pose-translation", type=float, default=0.25)
    parser.add_argument("--max-pose-rotation", type=float, default=3.0)
    parser.add_argument("--loo-iterations", type=int, default=3)
    parser.add_argument("--loo-reference-index", type=int, default=0)
    parser.add_argument("--loo-target-voxel-size", type=float)
    parser.add_argument("--loo-min-fitness", type=float, default=0.10)
    parser.add_argument("--loo-max-rmse", type=float, default=0.40)
    parser.add_argument("--loo-tukey-k", type=float)
    parser.add_argument("--loo-max-step-translation", type=float, default=0.5)
    parser.add_argument("--loo-max-step-rotation", type=float, default=3.0)
    parser.add_argument("--loo-translation-convergence", type=float, default=0.01)
    parser.add_argument("--loo-rotation-convergence", type=float, default=0.05)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.min_consistent_views < 1 or args.min_valid_pixels < 1:
        raise ValueError("view and pixel minimums must be positive")
    for name in (
        "consistency_relative_tolerance",
        "consistency_absolute_voxel_multiplier",
        "normal_discontinuity_relative",
        "normal_discontinuity_voxel_multiplier",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must not be negative")
    if not 0 < args.normal_consistency_angle <= 90:
        raise ValueError("normal-consistency-angle must be in (0, 90]")
    if args.voxel_size is not None and args.voxel_size <= 0:
        raise ValueError("voxel-size must be positive")
    if args.icp_voxel_size is not None and args.icp_voxel_size <= 0:
        raise ValueError("icp-voxel-size must be positive")
    if args.icp_neighbors < 1 or args.icp_iterations < 1:
        raise ValueError("ICP neighbors and iterations must be positive")
    if not 0 < args.icp_max_view_angle <= 180:
        raise ValueError("icp-max-view-angle must be in (0, 180]")
    if args.icp_max_correspondence is not None and args.icp_max_correspondence <= 0:
        raise ValueError("icp-max-correspondence must be positive")
    if not 0 <= args.icp_min_fitness <= 1 or args.icp_max_rmse <= 0:
        raise ValueError("ICP fitness must be in [0,1] and RMSE positive")
    if args.max_pose_translation <= 0 or args.max_pose_rotation <= 0:
        raise ValueError("pose correction limits must be positive")
    if (
        args.max_icp_translation <= 0
        or args.max_icp_rotation <= 0
        or args.pose_prior_information <= 0
    ):
        raise ValueError("ICP measurement and pose prior limits must be positive")
    if args.loo_iterations < 1 or args.loo_reference_index < 0:
        raise ValueError("leave-one-out iterations must be positive and reference non-negative")
    if args.loo_target_voxel_size is not None and args.loo_target_voxel_size <= 0:
        raise ValueError("loo-target-voxel-size must be positive")
    if not 0 <= args.loo_min_fitness <= 1 or args.loo_max_rmse <= 0:
        raise ValueError("leave-one-out fitness must be in [0,1] and RMSE positive")
    if args.loo_tukey_k is not None and args.loo_tukey_k <= 0:
        raise ValueError("loo-tukey-k must be positive")
    if (
        args.loo_max_step_translation <= 0
        or args.loo_max_step_rotation <= 0
        or args.loo_translation_convergence < 0
        or args.loo_rotation_convergence < 0
    ):
        raise ValueError("leave-one-out step limits must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_arguments(args)
        result = run_icp_refusion(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
