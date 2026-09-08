"""Isolated optional-dependency workers for the mannequin pipeline."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from recon_pipeline.artifacts import ensure_new_output, write_json


def packages() -> dict[str, str]:
    return {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}


def sam_snapshot(config: dict) -> Path:
    from huggingface_hub import snapshot_download

    model = Path(config["sam_model"])
    if model.is_dir():
        return model.resolve()
    return Path(snapshot_download(config["sam_model"], revision=config["sam_revision"], local_files_only=True))


def preflight(config: dict, kind: str) -> dict:
    if kind == "sam":
        import torch
        from transformers import Sam3Model, Sam3Processor  # noqa: F401

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 requires CUDA PyTorch in --sam-python")
        return {"snapshot": str(sam_snapshot(config)), "gpu": torch.cuda.get_device_name(config["device_id"]),
                "packages": packages()}
    import open3d
    import onnxruntime as ort
    import cv2  # noqa: F401
    if sys.platform == "win32" and sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 for Windows Open3D (--stereo-python)")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("FoundationStereo requires ONNX Runtime CUDA in --stereo-python")
    return {"open3d": open3d.__version__, "providers": ort.get_available_providers(),
            "python": sys.version, "packages": packages()}


def normalize(source: Path, destination: Path, config: dict) -> None:
    from PIL import Image, ImageOps
    from recon_pipeline.mannequin import collect_images, sha256

    paths = collect_images(source)
    ensure_new_output(destination)
    records = []
    for i, path in enumerate(paths, 1):
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            image.thumbnail((config["max_image_size"], config["max_image_size"]))
            name = f"frame_{i:06d}.png"
            image.save(destination / name)
        records.append({"source": str(path), "sha256": sha256(path), "image": name, "size": image.size})
    if config["shared_intrinsics"] and len({tuple(r["size"]) for r in records}) != 1:
        raise ValueError("Shared intrinsics require identical image dimensions; use --no-shared-intrinsics for mixed cameras")
    write_json(destination.parent / "input_images.json", {"images": records})


def select_mask(result: dict, height: int, width: int, np: Any) -> tuple[Any, float] | None:
    masks = result["masks"].detach().cpu().numpy().reshape(-1, height, width).astype(bool)
    scores = result["scores"].detach().float().cpu().numpy()
    candidates = []
    for i, mask in enumerate(masks):
        area = int(mask.sum())
        if area:
            center = mask[height//4:3*height//4, width//4:3*width//4].sum()
            candidates.append((float(scores[i]) * area**0.5 * (1 + center / area), i))
    if not candidates:
        return None
    _, index = max(candidates)
    return masks[index], float(scores[index])


def mask_images(source: Path, destination: Path, config: dict) -> None:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    from recon_pipeline.mannequin import collect_images

    paths = collect_images(source)
    ensure_new_output(destination)
    image_dir, mask_dir = destination / "images", destination / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    device = torch.device(f"cuda:{config['device_id']}")
    torch.manual_seed(config["seed"])
    snapshot = sam_snapshot(config)
    processor = Sam3Processor.from_pretrained(snapshot, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = Sam3Model.from_pretrained(snapshot, local_files_only=True, torch_dtype=dtype).to(device).eval()
    records = []
    for index, path in enumerate(paths, 1):
        with Image.open(path) as im:
            image = im.convert("RGB")
        width, height = image.size
        selected = None
        for prompt in config["prompts"]:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_instance_segmentation(
                outputs, threshold=config["sam_threshold"], mask_threshold=config["mask_threshold"],
                target_sizes=inputs["original_sizes"].detach().cpu().tolist(),
            )[0]
            selected = select_mask(result, height, width, np)
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f"No mannequin mask for {path}; adjust --prompts / --sam-threshold")
        mask, score = selected
        mask = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["mask_kernel"], config["mask_kernel"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        fraction = float((mask > 0).mean())
        if not config["min_mask_fraction"] <= fraction <= config["max_mask_fraction"]:
            raise RuntimeError(f"Suspicious mask area {fraction:.3f} for {path}; inspect masks or adjust bounds")
        rgb = np.asarray(image).copy()
        rgb[mask == 0] = 0
        # Keep registered image names, dimensions and orientation unchanged here.
        relative = path.relative_to(source)
        target = image_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        mask_target = mask_dir / relative.parent / f"{relative.name}.mask.png"
        mask_target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(target)
        Image.fromarray(mask).save(mask_target)
        records.append({"image": relative.as_posix(), "prompt": prompt, "score": score, "area_fraction": fraction})
        write_json(destination / "masks.json", {"snapshot": str(snapshot), "dtype": str(dtype), "images": records})
        print(f"SAM3 {index}/{len(paths)} {path.name}: {fraction:.1%}", flush=True)


def mesh(source: Path, destination: Path, config: dict) -> None:
    import numpy as np
    import open3d as o3d

    ensure_new_output(destination)
    payload = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if config["mesher"] == "poisson":
        cloud = o3d.io.read_point_cloud(payload["cloud"])
        if len(cloud.points) < 100 or not cloud.has_normals():
            raise RuntimeError("Poisson requires a nonempty oriented FoundationStereo point cloud")
        surface, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=config["poisson_depth"], n_threads=config["threads"],
        )
        density = np.asarray(density)
        surface.remove_vertices_by_mask(density < np.quantile(density, config["poisson_trim"]))
        surface = surface.crop(cloud.get_axis_aligned_bounding_box())
    else:
        surface = o3d.io.read_triangle_mesh(payload["mesh"])
    surface.remove_degenerate_triangles()
    surface.remove_duplicated_triangles()
    surface.remove_duplicated_vertices()
    surface.remove_unreferenced_vertices()
    if not len(surface.triangles):
        raise RuntimeError("No mesh survived reconstruction")
    if config["largest_component"]:
        labels, counts, _ = surface.cluster_connected_triangles()
        surface.remove_triangles_by_mask(np.asarray(labels) != int(np.argmax(counts)))
        surface.remove_unreferenced_vertices()
    surface.compute_vertex_normals()
    raw = destination / "mesh_full.ply"
    if not o3d.io.write_triangle_mesh(str(raw), surface):
        raise RuntimeError(f"Could not save {raw}")
    raw_faces = len(surface.triangles)
    if raw_faces > config["mesh_faces"]:
        surface = surface.simplify_quadric_decimation(config["mesh_faces"])
    # Poisson/cropping/decimation can leave >2 incident faces per edge.
    # OpenMVS seam construction assumes manifold neighborhoods.
    surface.remove_degenerate_triangles()
    surface.remove_duplicated_triangles()
    nonmanifold_edges = len(surface.get_non_manifold_edges(allow_boundary_edges=True))
    surface.remove_non_manifold_edges()
    removed_vertices = 0
    for _ in range(10):
        invalid = surface.get_non_manifold_vertices()
        if not len(invalid):
            break
        removed_vertices += len(invalid)
        surface.remove_vertices_by_index(invalid)
    surface.remove_unreferenced_vertices()
    if not len(surface.triangles) or not surface.is_vertex_manifold() or not surface.is_edge_manifold(allow_boundary_edges=True):
        raise RuntimeError("Could not produce a manifold mesh for seam leveling; inspect mesh_full.ply")
    surface.compute_vertex_normals()
    output = destination / "mesh.ply"
    if not o3d.io.write_triangle_mesh(str(output), surface):
        raise RuntimeError(f"Could not save {output}")
    write_json(destination / "mesh.json", {"method": config["mesher"], "raw_faces": raw_faces,
        "faces": len(surface.triangles), "vertices": len(surface.vertices), "mesh": str(output),
        "open3d": o3d.__version__, "historical_fused2_equivalence": "not established",
        "nonmanifold_edges_before_cleanup": nonmanifold_edges,
        "removed_nonmanifold_vertices": removed_vertices})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["preflight-sam", "preflight-stereo", "normalize", "mask", "mesh"])
    parser.add_argument("config", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.operation.startswith("preflight-"):
        write_json(args.destination, preflight(config, args.operation.split("-")[1]))
    else:
        {"normalize": normalize, "mask": mask_images, "mesh": mesh}[args.operation](args.source, args.destination, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
