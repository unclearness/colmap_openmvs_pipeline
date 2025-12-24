from pathlib import Path
import argparse


def main(images_dir: Path, stereo_dir: Path):
    stereo_dir.mkdir(parents=True, exist_ok=True)

    depth_maps_dir = stereo_dir / "depth_maps"
    depth_maps_dir.mkdir(parents=True, exist_ok=True)

    normal_maps_dir = stereo_dir / "normal_maps"
    normal_maps_dir.mkdir(parents=True, exist_ok=True)

    consistency_graphs_dir = stereo_dir / "consistency_graphs"
    consistency_graphs_dir.mkdir(parents=True, exist_ok=True)

    image_names = images_dir.glob("*.jpg")
    image_names = sorted(image_names)

    fusion_cfg_path = stereo_dir / "fusion.cfg"
    with fusion_cfg_path.open("w") as f:
        for image_path in image_names:
            f.write(f"{image_path.name}\n")

    patchmatch_cfg_path = stereo_dir / "patch-match.cfg"
    with patchmatch_cfg_path.open("w") as f:
        for image_path in image_names:
            f.write(f"{image_path.name}\n")
            f.write("__auto__, 20\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create stub COLMAP stereo config files."
    )
    parser.add_argument(
        "images_dir", type=Path, help="Directory containing input images."
    )
    parser.add_argument(
        "stereo_dir", type=Path, help="Directory to save stereo config files."
    )
    args = parser.parse_args()

    main(args.images_dir, args.stereo_dir)
