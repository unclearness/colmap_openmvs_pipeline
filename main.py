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
    run_cmd(
        [
            COLMAP_PATH,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(output_root_dir) + "/sparse",
        ]
    )

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

        cnt += 1


def run_openmvs(colmap_output_dense_dir, output_root_dir, refine_mesh=True):
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

    mesh_name = "scene_dense_refine.ply"

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
    )

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


if __name__ == "__main__":
    for image_dir in Path("./data/").iterdir():
        if not image_dir.is_dir():
            continue
        output_root_dir = Path(__file__).parent / "output" / image_dir.stem
        main(image_dir, output_root_dir)
