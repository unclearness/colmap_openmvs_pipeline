from pathlib import Path
import datetime
from util import run_and_log, is_colmap_version_at_least, setup_logger

COLMAP_PATH = "./colmap-x64-windows-cuda/bin/colmap.exe"
OPENMVS_DIR = "./OpenMVS_Windows_x64/"

COLMAP_VERSION = "3.12.4"

JST = datetime.timezone(datetime.timedelta(hours=9))
now = datetime.datetime.now(JST)
timestamp = now.strftime("%Y%m%d_%H%M%S")

LOG_PATH = f"./{timestamp}.log"

logger = setup_logger(LOG_PATH)
colmap_ok, colmap_ver = is_colmap_version_at_least(COLMAP_VERSION, COLMAP_PATH)
if not colmap_ok:
    logger.warning(f"{colmap_ver} is lower than expected {COLMAP_VERSION}")


def run_cmd(cmd):
    # COLMAP log will be redirected to LOG_PATH
    # OpenMVS log will not be redirected to LOG_PATH but it will be written in its own log
    run_and_log(cmd, LOG_PATH)


def run_colmap_sfm(
    database_path,
    image_dir,
    output_root_dir,
    gpu_index=-1,
    camera_mode="1",  # [AUTO = 0, SINGLE = 1, PER_FOLDER = 2, PER_IMAGE = 3]
    camera_model="SIMPLE_RADIAL",  # https://colmap.github.io/cameras.html
    intrinsic_prior=None,
    sequential_matching=True,
    forward_motion=False,
):
    # Create output directory
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_root_dir) / "sparse").mkdir(parents=True, exist_ok=True)
    (Path(output_root_dir) / "dense").mkdir(parents=True, exist_ok=True)

    # Create database
    run_cmd(
        [
            COLMAP_PATH,
            "database_creator",
            "--database_path",
            str(database_path),
        ]
    )

    # Feature extraction
    feature_extraction_cmd = [
        COLMAP_PATH,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
    ]
    if gpu_index >= 0:
        feature_extraction_cmd.append("--SiftExtraction.use_gpu")
        feature_extraction_cmd.append("1")
        feature_extraction_cmd.append("--SiftExtraction.gpu_index")
        feature_extraction_cmd.append(str(gpu_index))
    else:
        feature_extraction_cmd.append("--SiftExtraction.use_gpu")
        feature_extraction_cmd.append("0")

    # https://github.com/colmap/colmap/blob/d478af0ce448251b44e6a4525f8bf640f6cdc9e3/src/colmap/exe/feature.h#L69
    # https://github.com/colmap/colmap/blob/107814a1b8ce60c49d0815938934be148d05b5b9/src/colmap/exe/feature.cc#L73
    feature_extraction_cmd.append("--camera_mode")
    feature_extraction_cmd.append(camera_mode)

    feature_extraction_cmd.append("--ImageReader.camera_model")
    feature_extraction_cmd.append(camera_model)
    if intrinsic_prior is not None:
        feature_extraction_cmd.append("--ImageReader.camera_params")
        intrinsic_prior_str = ""
        for intrin in intrinsic_prior:
            intrinsic_prior_str += str(intrin) + ","
        intrinsic_prior_str = intrinsic_prior_str[0:-1]
        feature_extraction_cmd.append(intrinsic_prior_str)

    run_cmd(feature_extraction_cmd)

    # Image matching
    matcher_name = "sequential_matcher" if sequential_matching else "exhaustive_matcher"
    matching_cmd = [COLMAP_PATH, matcher_name, "--database_path", str(database_path)]
    if gpu_index >= 0:
        matching_cmd.append("--SiftMatching.use_gpu")
        matching_cmd.append("1")
        matching_cmd.append("--SiftMatching.gpu_index")
        matching_cmd.append(str(gpu_index))
    else:
        matching_cmd.append("--SiftMatching.use_gpu")
        matching_cmd.append("0")
    run_cmd(matching_cmd)

    # Sparse reconstruction
    mapper_cmd = [
        COLMAP_PATH,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(output_root_dir) + "/sparse",
    ]
    if forward_motion:
        # https://github.com/colmap/colmap/issues/1490
        mapper_cmd.append("--Mapper.init_max_forward_motion")
        mapper_cmd.append("1.0")
        mapper_cmd.append("--Mapper.init_min_tri_angle")
        mapper_cmd.append("0.5")
        mapper_cmd.append("--Mapper.tri_create_max_angle_error")
        mapper_cmd.append("0.5")
        mapper_cmd.append("--Mapper.filter_min_tri_angle")
        mapper_cmd.append("0.5")

    run_cmd(mapper_cmd)

    # Perform conversion per seperated model
    cnt = 0
    converted_base_dir = output_root_dir / "sparse" / "converted"
    converted_base_dir.mkdir(parents=True, exist_ok=True)
    while True:
        # Model conversion
        sparse_out_dir = output_root_dir / "sparse" / str(cnt)
        if not sparse_out_dir.exists():
            break
        converted_out_dir = converted_base_dir / str(cnt)
        converted_out_dir.mkdir(parents=True, exist_ok=True)

        undistort_out_dir = converted_out_dir / "undistort"
        undistort_out_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                COLMAP_PATH,
                "model_converter",
                "--input_path",
                str(sparse_out_dir),
                "--output_path",
                str(converted_out_dir) + "/model.ply",
                "--output_type",
                "PLY",
            ]
        )
        run_cmd(
            [
                COLMAP_PATH,
                "model_converter",
                "--input_path",
                str(sparse_out_dir),
                "--output_path",
                str(converted_out_dir),
                "--output_type",
                "TXT",
            ]
        )

        # Undistort images
        run_cmd(
            [
                COLMAP_PATH,
                "image_undistorter",
                "--image_path",
                str(image_dir),
                "--input_path",
                str(converted_out_dir),
                "--output_path",
                str(undistort_out_dir),
            ]
        )

        # Make undistorted files TXT
        run_cmd(
            [
                COLMAP_PATH,
                "model_converter",
                "--input_path",
                str(undistort_out_dir / "sparse"),
                "--output_path",
                str(undistort_out_dir / "sparse"),
                "--output_type",
                "TXT",
            ]
        )

        cnt += 1


def run_colmap_mvs(
    undistort_dir,
    gpu_index: int = -1,
    geom_consistency: bool = True,
    patchmatch_max_image_size: int = -1,  # COLMAP default
    patchmatch_window_step: int = 1,  # COLMAP default
    patchmatch_num_iterations: int = 5,  # COLMAP default
    patchmatch_window_radius: int = 5,  # COLMAP default
    patchmatch_num_samples: int = 15,  # COLMAP default
    stereofusion_min_num_pixels: int = 5,  # StereoFusion.min_num_pixels default
    use_poisson_mesher: bool = False,  # If True => poisson_mesher; else delaunay_mesher
    poisson_depth: int = 10,  # Typical 9–12
    delaunay_quality_regularization: int = 1,
    delaunay_max_proj_dist: float = 20.0,  # pixels; smaller => stricter
):
    """
    Run COLMAP MVS (PatchMatchStereo -> StereoFusion -> optional meshing)
    for a single model whose undistorted workspace is at `undistort_dir`.

    Args:
        undistort_dir: Path to COLMAP 'undistort' workspace (contains images + stereo/)
        output_dense_dir: Path to write fused point cloud and mesh
    """
    undistort_dir = Path(undistort_dir)

    if not undistort_dir.exists():
        raise FileNotFoundError(f"undistort_dir not found: {undistort_dir}")

    # 1) PatchMatch Stereo (depth/normal estimation)
    pms_cmd = [
        COLMAP_PATH,
        "patch_match_stereo",
        "--workspace_path",
        str(undistort_dir),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.max_image_size",
        str(patchmatch_max_image_size),
        "--PatchMatchStereo.window_step",
        str(patchmatch_window_step),
        "--PatchMatchStereo.num_iterations",
        str(patchmatch_num_iterations),
        "--PatchMatchStereo.window_radius",
        str(patchmatch_window_radius),
        "--PatchMatchStereo.num_samples",
        str(patchmatch_num_samples),
        "--PatchMatchStereo.geom_consistency",
        "1" if geom_consistency else "0",
    ]
    if gpu_index >= 0:
        pms_cmd += [
            "--PatchMatchStereo.gpu_index",
            str(gpu_index),
        ]

    run_cmd(pms_cmd)

    # 2) Stereo Fusion (fuse depth maps into a point cloud)
    fused_ply = undistort_dir / "fused.ply"
    fusion_cmd = [
        COLMAP_PATH,
        "stereo_fusion",
        "--workspace_path",
        str(undistort_dir),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric",  # use "photometric" if you ran photometric stereo
        "--output_path",
        str(fused_ply),
        "--StereoFusion.min_num_pixels",
        str(stereofusion_min_num_pixels),
    ]
    run_cmd(fusion_cmd)

    # 3) (Optional) Meshing
    if use_poisson_mesher:
        mesh_out = undistort_dir / "mesh_poisson.ply"
        poisson_cmd = [
            COLMAP_PATH,
            "poisson_mesher",
            "--input_path",
            str(fused_ply),
            "--output_path",
            str(mesh_out),
            "--PoissonMesher.depth",
            str(poisson_depth),
        ]
        run_cmd(poisson_cmd)
    else:
        # Delaunay reads the COLMAP workspace (needs images + depth/normal maps)
        mesh_out = undistort_dir / "mesh_delaunay.ply"
        delaunay_cmd = [
            COLMAP_PATH,
            "delaunay_mesher",
            "--input_path",
            str(undistort_dir),
            "--output_path",
            str(mesh_out),
            "--DelaunayMeshing.max_proj_dist",
            str(delaunay_max_proj_dist),
            "--DelaunayMeshing.quality_regularization",
            str(delaunay_quality_regularization),
        ]
        run_cmd(delaunay_cmd)


def run_openmvs(colmap_output_dense_dir, output_root_dir, refine_mesh=False):
    # Create output directory
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)

    mvs_sparse_fname = str(output_root_dir) + "/scene.mvs"

    # Convert colmap sparse reconstruction to openmvs format
    run_cmd(
        [
            OPENMVS_DIR + "InterfaceCOLMAP ",
            "-i",
            str(colmap_output_dense_dir),
            "--image-folder",
            str(colmap_output_dense_dir) + "/images",
            "-o",
            mvs_sparse_fname,
            "-w",
            output_root_dir,
        ]
    )

    # Densify point cloud
    run_cmd(
        [OPENMVS_DIR + "DensifyPointCloud", mvs_sparse_fname, "-w", output_root_dir]
    )

    mvs_dense_fname = str(output_root_dir) + "/scene_dense.mvs"

    # Reconstruct mesh
    mesh_name = "scene_dense_mesh.ply"
    run_cmd(
        [
            OPENMVS_DIR + "ReconstructMesh",
            "-i",
            mvs_dense_fname,
            "-o",
            str(output_root_dir) + "/" + mesh_name,
            "-w",
            output_root_dir,
        ]
    )

    if refine_mesh:
        # Refine mesh
        mesh_name = "scene_dense_refine.ply"
        run_cmd(
            [
                OPENMVS_DIR + "RefineMesh",
                "-i",
                mvs_dense_fname,
                "-m",
                str(output_root_dir) + "/scene_dense_mesh.ply",
                "-o",
                str(output_root_dir) + "/" + mesh_name,
                "-w",
                output_root_dir,
            ]
        )

    # Texture mesh
    textured_mesh_name = "scene_dense_texture.ply"
    run_cmd(
        [
            OPENMVS_DIR + "TextureMesh",
            "-i",
            mvs_dense_fname,
            "-m",
            mesh_name,
            "-o",
            textured_mesh_name,
            "-w",
            output_root_dir,
        ]
    )


def main(
    image_dir,
    output_root_dir,
    gpu_index=0,
    camera_mode="1",
    camera_model="SIMPLE_RADIAL",
    intrinsic_prior=None,
    sequential_matching=True,
    forward_motion=False,
    use_openmvs=True,
    fast_colmap_mvs=False
):
    database_path = output_root_dir / "colmap" / "database.db"
    run_colmap_sfm(
        database_path,
        image_dir,
        output_root_dir / "colmap",
        gpu_index=gpu_index,
        camera_mode=camera_mode,
        camera_model=camera_model,
        intrinsic_prior=intrinsic_prior,
        sequential_matching=sequential_matching,
        forward_motion=forward_motion,
    )

    if use_openmvs:
        cnt = 0
        converted_base_dir = output_root_dir / "colmap" / "sparse" / "converted"
        while True:
            converted_dir = converted_base_dir / str(cnt)
            if not converted_dir.exists():
                break
            output_openmvs_dir = output_root_dir / "openmvs" / str(cnt)
            output_openmvs_dir.mkdir(parents=True, exist_ok=True)
            run_openmvs(converted_dir / "undistort", output_openmvs_dir)
            cnt += 1
    else:
        cnt = 0
        converted_base_dir = output_root_dir / "colmap" / "sparse" / "converted"
        while True:
            converted_dir = converted_base_dir / str(cnt)
            if not converted_dir.exists():
                break
            output_dense_dir = output_root_dir / "colmap" / "dense" / str(cnt)
            output_dense_dir.mkdir(parents=True, exist_ok=True)
            if fast_colmap_mvs:
                # https://colmap.github.io/faq.html#speedup-dense-reconstruction
                # Max image size is 1024, and other parameters are set as half of the default value
                run_colmap_mvs(
                    converted_dir / "undistort",
                    patchmatch_max_image_size=1024,
                    patchmatch_window_step=2,
                    patchmatch_num_iterations=3,
                    patchmatch_window_radius=3,
                    patchmatch_num_samples=8,
                )
            else:
                run_colmap_mvs(
                    converted_dir / "undistort")
            cnt += 1


if __name__ == "__main__":
    for image_dir in Path("./data/").iterdir():
        if not image_dir.is_dir():
            continue
        output_root_dir = Path(__file__).parent / "output" / image_dir.stem
        main(image_dir, output_root_dir)
