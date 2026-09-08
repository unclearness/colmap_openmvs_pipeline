"""Download verified official reconstruction tools and model assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from recon_pipeline.tooling import install_asset, load_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install version-pinned official tools/models into tools/. "
            "Downloads are accepted only when their SHA-256 matches tools/manifest.json."
        )
    )
    parser.add_argument(
        "tools",
        metavar="TOOL",
        nargs="*",
        choices=(
            "colmap",
            "openmvs",
            "azure-kinect",
            "orbbec-k4a",
            "foundationstereo",
        ),
        help="tool(s) to install; defaults to both",
    )
    parser.add_argument(
        "--colmap-variant",
        choices=("cuda", "nocuda"),
        default="cuda",
        help="COLMAP binary edition (default: cuda)",
    )
    parser.add_argument(
        "--openmvs-variant",
        choices=("cuda", "cpu"),
        default="cuda",
        help="OpenMVS binary edition (default: cuda)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root containing tools/ (default: inferred from this script)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="alternate manifest path (default: ROOT/tools/manifest.json)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="archive cache directory (default: ROOT/tools/_downloads)",
    )
    parser.add_argument(
        "--seven-zip",
        type=Path,
        help="path to 7z.exe; otherwise SEVEN_ZIP_PATH, PATH, and Program Files are checked",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else root / "tools" / "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    selected = list(dict.fromkeys(args.tools or ("colmap", "openmvs")))
    variants = {
        "colmap": args.colmap_variant,
        "openmvs": args.openmvs_variant,
        "azure-kinect": "windows",
        "orbbec-k4a": "windows",
        "foundationstereo": "dynamic",
    }

    for tool_name in selected:
        variant = variants[tool_name]
        version = manifest["tools"][tool_name]["version"]
        print(f"Installing {tool_name} {version} ({variant})...")
        install_dir = install_asset(
            root,
            tool_name,
            variant,
            manifest_path=manifest_path,
            download_dir=args.download_dir,
            seven_zip=args.seven_zip,
        )
        print(f"Ready: {install_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
