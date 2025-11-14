from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

import pycolmap

from util import setup_logger

COLMAP_VERSION = "3.13.0"

JST = datetime.timezone(datetime.timedelta(hours=9))
now = datetime.datetime.now(JST)
timestamp = now.strftime("%Y%m%d_%H%M%S")

LOG_PATH = f"./{timestamp}.log"

logger = setup_logger(LOG_PATH)
if pycolmap.__version__ != COLMAP_VERSION:
    logger.warning(
        "pycolmap version %s differs from expected %s",
        pycolmap.__version__,
        COLMAP_VERSION,
    )


def _resolve_camera_mode(camera_mode: str | int) -> pycolmap.CameraMode:
    if isinstance(camera_mode, int):
        mapping = {
            0: pycolmap.CameraMode.AUTO,
            1: pycolmap.CameraMode.SINGLE,
            2: pycolmap.CameraMode.PER_FOLDER,
            3: pycolmap.CameraMode.PER_IMAGE,
        }
        if camera_mode not in mapping:
            raise ValueError(f"Invalid camera_mode index: {camera_mode}")
        return mapping[camera_mode]

    camera_mode_str = str(camera_mode).strip().upper()
    if camera_mode_str.isdigit():
        return _resolve_camera_mode(int(camera_mode_str))

    try:
        return pycolmap.CameraMode[camera_mode_str]
    except KeyError as exc:
        raise ValueError(
            "camera_mode must be one of {0,1,2,3,'AUTO','SINGLE','PER_FOLDER','PER_IMAGE'}"
        ) from exc


def _select_device(gpu_index: int) -> tuple[pycolmap.Device, str]:
    if gpu_index >= 0:
        if pycolmap.has_cuda:
            return pycolmap.Device.cuda, str(gpu_index)
        logger.warning(
            "CUDA requested (gpu_index=%d) but pycolmap is built without CUDA",
            gpu_index,
        )
    return pycolmap.Device.cpu, "-1"


def _ensure_empty_database(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    pycolmap.Database.open(str(database_path)).close()


def run_pycolmap_sfm(
    database_path: Path,
    image_dir: Path,
    output_root_dir: Path,
    *,
    gpu_index: int = -1,
    camera_mode: str | int = "1",
    camera_model: str = "SIMPLE_RADIAL",
    intrinsic_prior: Sequence[float] | None = None,
    sequential_matching: bool = True,
    forward_motion: bool = False,
    ba_fixed_camera_intrinsics: bool = False,
) -> list[Path]:
    """Run the SfM portion of the COLMAP pipeline using the pycolmap API."""

    image_dir = Path(image_dir)
    output_root_dir = Path(output_root_dir)
    database_path = Path(database_path)

    logger.info("Starting pycolmap SfM pipeline")
    output_root_dir.mkdir(parents=True, exist_ok=True)
    (output_root_dir / "sparse").mkdir(parents=True, exist_ok=True)
    (output_root_dir / "dense").mkdir(parents=True, exist_ok=True)

    _ensure_empty_database(database_path)

    device, gpu_string = _select_device(gpu_index)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera_model
    if intrinsic_prior is not None:
        reader_options.camera_params = list(intrinsic_prior)

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = gpu_index >= 0 and pycolmap.has_cuda
    extraction_options.gpu_index = gpu_string

    pycolmap.extract_features(
        str(database_path),
        str(image_dir),
        camera_mode=_resolve_camera_mode(camera_mode),
        camera_model=camera_model,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=device,
    )
    logger.info("Feature extraction finished")

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = gpu_index >= 0 and pycolmap.has_cuda
    matching_options.gpu_index = gpu_string

    if sequential_matching:
        pairing_options = pycolmap.SequentialPairingOptions()
        match_fn = pycolmap.match_sequential
        logger.info("Running sequential matching")
    else:
        pairing_options = pycolmap.ExhaustivePairingOptions()
        match_fn = pycolmap.match_exhaustive
        logger.info("Running exhaustive matching")

    match_fn(
        str(database_path),
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=device,
    )
    logger.info("Feature matching finished")

    pipeline_options = pycolmap.IncrementalPipelineOptions()
    mapper_opts = pipeline_options.get_mapper()
    if forward_motion:
        mapper_opts.init_max_forward_motion = 1.0
        mapper_opts.init_min_tri_angle = 0.5
    if ba_fixed_camera_intrinsics:
        pipeline_options.ba_refine_focal_length = False
        pipeline_options.ba_refine_principal_point = False
        pipeline_options.ba_refine_extra_params = False
        mapper_opts.abs_pose_refine_focal_length = False
        mapper_opts.abs_pose_refine_extra_params = False
    pipeline_options.mapper = mapper_opts

    tri_opts = pipeline_options.get_triangulation()
    if forward_motion:
        tri_opts.create_max_angle_error = 0.5
        tri_opts.continue_max_angle_error = 0.5
        tri_opts.min_angle = 0.5
    pipeline_options.triangulation = tri_opts

    reconstructions = pycolmap.incremental_mapping(
        str(database_path),
        str(image_dir),
        str(output_root_dir / "sparse"),
        options=pipeline_options,
    )

    if not reconstructions:
        raise RuntimeError("pycolmap incremental mapping produced no models")

    undistort_dirs: list[Path] = []
    for model_id, reconstruction in sorted(reconstructions.items()):
        logger.info(
            "Processing reconstruction %s: %d images, %d points",
            model_id,
            reconstruction.num_reg_images(),
            reconstruction.num_points3D(),
        )
        sparse_dir = output_root_dir / "sparse" / str(model_id)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        reconstruction.write(str(sparse_dir))

        converted_dir = output_root_dir / "sparse" / "converted" / str(model_id)
        converted_dir.mkdir(parents=True, exist_ok=True)
        reconstruction.write_text(str(converted_dir))
        reconstruction.export_PLY(str(converted_dir / "model.ply"))

        undistort_dir = output_root_dir / "dense" / str(model_id)
        undistort_dir.mkdir(parents=True, exist_ok=True)
        pycolmap.undistort_images(
            str(undistort_dir),
            str(converted_dir),
            str(image_dir),
        )

        undistorted_sparse = undistort_dir / "sparse"
        if undistorted_sparse.exists():
            undistorted_rec = pycolmap.Reconstruction(str(undistorted_sparse))
            undistorted_rec.write_text(str(undistorted_sparse))

        undistort_dirs.append(undistort_dir)

    return undistort_dirs


def run_pycolmap_mvs(
    undistort_dir: Path,
    *,
    gpu_index: int = -1,
    geom_consistency: bool = True,
    patchmatch_max_image_size: int = -1,
    patchmatch_window_step: int = 1,
    patchmatch_num_iterations: int = 5,
    patchmatch_window_radius: int = 5,
    patchmatch_num_samples: int = 15,
    stereofusion_min_num_pixels: int = 5,
    stereofusion_max_reproj_error: float = 2.0,
    stereofusion_max_depth_error: float = 0.01,
    use_poisson_mesher: bool = False,
    poisson_depth: int = 10,
    delaunay_quality_regularization: int = 1,
    delaunay_max_proj_dist: float = 20.0,
) -> None:
    undistort_dir = Path(undistort_dir)
    if not undistort_dir.exists():
        raise FileNotFoundError(f"undistort_dir not found: {undistort_dir}")

    _, gpu_string = _select_device(gpu_index)

    pm_opts = pycolmap.PatchMatchOptions()
    pm_opts.max_image_size = patchmatch_max_image_size
    pm_opts.window_step = patchmatch_window_step
    pm_opts.num_iterations = patchmatch_num_iterations
    pm_opts.window_radius = patchmatch_window_radius
    pm_opts.num_samples = patchmatch_num_samples
    pm_opts.geom_consistency = geom_consistency
    pm_opts.gpu_index = gpu_string

    logger.info("Running PatchMatchStereo in %s", undistort_dir)
    pycolmap.patch_match_stereo(str(undistort_dir), options=pm_opts)

    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.min_num_pixels = stereofusion_min_num_pixels
    fusion_opts.max_reproj_error = stereofusion_max_reproj_error
    fusion_opts.max_depth_error = stereofusion_max_depth_error

    fused_ply = undistort_dir / "fused.ply"
    logger.info("Running StereoFusion -> %s", fused_ply)
    pycolmap.stereo_fusion(
        str(fused_ply),
        str(undistort_dir),
        options=fusion_opts,
    )

    if use_poisson_mesher:
        mesh_out = undistort_dir / "mesh_poisson.ply"
        mesh_opts = pycolmap.PoissonMeshingOptions()
        mesh_opts.depth = poisson_depth
        logger.info("Running Poisson meshing -> %s", mesh_out)
        pycolmap.poisson_meshing(
            str(mesh_out),
            str(fused_ply),
            options=mesh_opts,
        )
    else:
        logger.warning(
            "Delaunay meshing is not exposed in pycolmap 3.13.0; skipping mesh generation",
        )


def main(
    image_dir: Path,
    output_root_dir: Path,
    *,
    gpu_index: int = 0,
    camera_mode: str | int = "1",
    camera_model: str = "SIMPLE_RADIAL",
    intrinsic_prior: Sequence[float] | None = None,
    sequential_matching: bool = True,
    forward_motion: bool = False,
    ba_fixed_camera_intrinsics: bool = False,
    use_openmvs: bool = False,
    fast_colmap_mvs: bool = False,
) -> None:
    if use_openmvs:
        raise ValueError("main_pycolmap does not integrate with OpenMVS")

    image_dir = Path(image_dir)
    output_root_dir = Path(output_root_dir)

    database_path = output_root_dir / "colmap" / "database.db"
    colmap_output_root = output_root_dir / "colmap"

    undistort_dirs = run_pycolmap_sfm(
        database_path,
        image_dir,
        colmap_output_root,
        gpu_index=gpu_index,
        camera_mode=camera_mode,
        camera_model=camera_model,
        intrinsic_prior=intrinsic_prior,
        sequential_matching=sequential_matching,
        forward_motion=forward_motion,
        ba_fixed_camera_intrinsics=ba_fixed_camera_intrinsics,
    )

    for undistort_dir in undistort_dirs:
        if fast_colmap_mvs:
            logger.info("Running fast dense reconstruction for %s", undistort_dir)
            run_pycolmap_mvs(
                undistort_dir,
                gpu_index=gpu_index,
                patchmatch_max_image_size=1024,
                patchmatch_window_step=2,
                patchmatch_num_iterations=3,
                patchmatch_window_radius=3,
                patchmatch_num_samples=8,
            )
        else:
            run_pycolmap_mvs(undistort_dir, gpu_index=gpu_index)


if __name__ == "__main__":
    for image_dir in Path("./data/").iterdir():
        if not image_dir.is_dir():
            continue
        output_root_dir = Path(__file__).parent / "output_pycolmap" / image_dir.stem
        main(image_dir, output_root_dir)
