from pathlib import Path
import subprocess


COLMAP_PATH = "./colmap-x64-windows-cuda/bin/colmap.exe"
OPENMVS_DIR = "./openmvs_v2.3.0/"


def run_colmap_sfm(
    database_path,
    image_dir,
    output_root_dir,
    gpu_index=-1,
    camera_mode="3",  # [AUTO = 0, SINGLE = 1, PER_FOLDER = 2, PER_IMAGE = 3]
    camera_model="PINHOLE",
    intrinsic_prior=None,
    sequential_matching=False,
):
    # Create output directory
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_root_dir) / "sparse").mkdir(parents=True, exist_ok=True)
    (Path(output_root_dir) / "dense").mkdir(parents=True, exist_ok=True)

    # Create database
    subprocess.run(
        [
            COLMAP_PATH,
            "database_creator",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
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

    # https://github.com/colmap/colmap/blob/107814a1b8ce60c49d0815938934be148d05b5b9/src/colmap/exe/feature.cc#L73
    feature_extraction_cmd.append("--camera_mode")
    feature_extraction_cmd.append(camera_mode)

    feature_extraction_cmd.append("--ImageReader.camera_model")
    feature_extraction_cmd.append(camera_model)
    if intrinsic_prior is not None:
        feature_extraction_cmd.append("--ImageReader.camera_params")
        intrinsic_prior_str = ""
        for intrin in intrinsic_prior:
            intrinsic_prior_str += str(intrin) + " "
        feature_extraction_cmd.append(intrinsic_prior_str)

    subprocess.run(feature_extraction_cmd)

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
    subprocess.run(matching_cmd)

    # Sparse reconstruction
    subprocess.run(
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

    # Model conversion
    subprocess.run(
        [
            COLMAP_PATH,
            "model_converter",
            "--input_path",
            str(output_root_dir) + "/sparse/0/",
            "--output_path",
            str(output_root_dir) + "/sparse/model.ply",
            "--output_type",
            "PLY",
        ]
    )
    subprocess.run(
        [
            COLMAP_PATH,
            "model_converter",
            "--input_path",
            str(output_root_dir) + "/sparse/0/",
            "--output_path",
            str(output_root_dir) + "/sparse/",
            "--output_type",
            "TXT",
        ]
    )

    # Undistort images
    subprocess.run(
        [
            COLMAP_PATH,
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(output_root_dir) + "/sparse",
            "--output_path",
            str(output_root_dir),
        ]
    )


def run_openmvs(colmap_output_dense_dir, output_root_dir, refine_mesh=True):
    # Create output directory
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)

    mvs_sparse_fname = str(output_root_dir) + "/scene.mvs"

    print(
        str(colmap_output_dense_dir) + "/images",
    )

    # Convert colmap sparse reconstruction to openmvs format
    subprocess.run(
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
    subprocess.run(
        [OPENMVS_DIR + "DensifyPointCloud", mvs_sparse_fname, "-w", output_root_dir]
    )

    mvs_dense_fname = str(output_root_dir) + "/scene_dense.mvs"

    # Reconstruct mesh
    mesh_name = "scene_dense_mesh.ply"
    subprocess.run(
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
        subprocess.run(
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
    subprocess.run(
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


def main(image_dir, output_root_dir, gpu_index=0):
    database_path = output_root_dir / "colmap" / "database.db"
    run_colmap_sfm(
        database_path, image_dir, output_root_dir / "colmap", gpu_index=gpu_index
    )
    run_openmvs(output_root_dir / "colmap", output_root_dir / "openmvs")


if __name__ == "__main__":
    for image_dir in Path("./data/").iterdir():
        if not image_dir.is_dir():
            continue
        output_root_dir = Path(__file__).parent / "output" / image_dir.stem
        main(image_dir, output_root_dir)
