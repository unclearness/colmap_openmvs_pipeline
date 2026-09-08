# Repository Guidelines

## Scope and purpose

- These instructions apply to the entire repository.
- This is a Windows-oriented Python 3.11+ photogrammetry pipeline with four public backends: COLMAP, COLMAP-to-OpenMVS, RealityScan, and Metashape Professional.
- Keep changes focused. Inspect the worktree before and after editing, and preserve unrelated tracked or untracked user files.

## Architecture and repository map

- `recon_pipeline/cli.py`: the `run`, `batch`, and `doctor` CLI.
- `recon_pipeline/orchestrator.py`: validation, tool resolution, manifests, and backend dispatch.
- `recon_pipeline/backends/`: engine adapters. `openmvs` always uses COLMAP for SfM first.
- `recon_pipeline/workers/metashape_worker.py`: isolated, lazy-import Metashape API worker.
- `recon_pipeline/rgbd_fusion.py`, `rgbd_main.py`: optional Python 3.12 Open3D/K4A RGB-D SLAM and OpenMVS texturing entry point.
- `recon_pipeline/foundation_stereo.py`, `foundation_stereo_main.py`: optional COLMAP/RealityScan-exported SfM + rectified two-view NGC FoundationStereo + multi-view TSDF experiment.
- `recon_pipeline/foundation_stereo_icp_refusion.py`, `foundation_stereo_icp_refusion_main.py`: saved-depth relaxed consistency with RealityScan-prior pose-graph or leave-one-out `(N-1 fixed) + 1 moving` ICP. The mannequin experiment showed that even robust leave-one-out ICP can thicken a bimodal surface; compare projected surface thickness before accepting it.
- `recon_pipeline/rgbd_hybrid.py`, `rgbd_hybrid_main.py`: staged RealityScan + metric ToF + ToF-anchored FoundationStereo experiment. Run `--tof-only` before stereo/texture; ToF-only multiple surfaces mean the pose gate failed.
- `recon_pipeline/foundation_stereo_refusion.py`, `foundation_stereo_refusion_main.py`: re-fuse saved FoundationStereo depths with stronger multi-view depth and world-normal consistency without rerunning ONNX.
- `recon_pipeline/assets/realityscan/`: byte-verified RealityScan export parameter XML.
- `recon_pipeline/tooling.py`, `tools/manifest.json`, `scripts/install_tools.py`: pinned official downloads, SHA-256 verification, safe extraction, and installation discovery.
- `tests/`: standard-library `unittest` suite.
- `main.py` and `main_pycolmap.py`: compatibility entry points; new work belongs in the package.
- `data/ET/` and `data/kermit/`: tracked lightweight integration inputs.

## Tool and capability rules

- The pinned releases are COLMAP 4.1.1, OpenMVS 2.4.0, and NGC FoundationStereo dynamic 2.0. Change versions, URLs, and SHA-256 values together and verify them against official release metadata.
- Downloaded/extracted binaries are local artifacts under ignored versioned `tools/` directories. Do not commit them.
- Do not modify or rebuild third-party tools unless the task explicitly asks for it. The normal workflow uses official binaries via `python scripts/install_tools.py`.
- COLMAP/OpenMVS support `sfm`, `dense`, and `mesh`; RealityScan and Metashape support only `sfm` and `mesh`.
- RealityScan and Metashape mesh flows must not publish a dense point-cloud artifact. In Metashape, call `buildDepthMaps()` then `buildModel(DepthMapsData)` and never `buildPointCloud()`.
- Every backend must export or preserve a valid COLMAP text model under `colmap/sparse/<id>`. A mesh remains a separate PLY/OBJ because COLMAP sparse format cannot represent it.
- Metashape automation requires the Professional Python API. Never bypass licensing or pretend Standard edition is automatable; fail before processing with an actionable message.

## Implementation conventions

- Use four-space indentation, `snake_case`, `UPPER_CASE` constants, `pathlib.Path`, modern type annotations, and specific exceptions.
- Build external commands as argv lists. Run them through `CommandRunner`, check exit status, and validate required artifacts.
- Do not import side-effectful proprietary APIs in the main process. Keep Metashape imports inside its worker.
- Preserve engine-specific native intermediates under `native/`; do not claim interoperability that is not implemented.
- An output directory must be new or empty. Do not add overwrite/cleanup behavior without an explicit design and tests.
- Preserve COLMAP 4.1's `SIFT_BRUTEFORCE` matcher value and OpenMVS 2.4's external `scene_dense.ply` passed to `ReconstructMesh -p`.
- Treat Delaunay as an optional CGAL capability. `--mesher auto` must use Poisson when Delaunay is unavailable; do not require a source build.
- Keep CUDA and CPU OpenMVS runtimes isolated. Auto fallback may occur only in `auto` mode, never after an explicit `cuda` selection.
- RGB-D fusion uses Microsoft K4A playback for K4A-compatible MKV, derives mode-specific intrinsics from the embedded calibration, undistorts aligned RGB/depth, and stores poses as camera-to-world before converting them to COLMAP world-to-camera.
- Preserve the high-density Open3D mesh separately from the simplified OpenMVS texturing mesh. Keep OpenMVS seam leveling disabled for this external-pose path unless a verified upstream fix is available.
- Publish every textured result as standard OBJ + MTL + image. COLMAP/OpenMVS per-face-UV PLY may remain as an intermediate artifact, but common viewers ignore that extension, so never expose it as the primary textured mesh.
- FoundationStereo consumes rectified pairs with positive horizontal disparity. Preserve the COLMAP world-to-camera convention, reject pathological rectification or out-of-range expected disparity, and retain sparse-disparity plus multi-view consistency diagnostics. The model has no native confidence output.
- Keep FoundationStereo reference views spatially distributed around the PCA-estimated camera orbit. For two sources per reference, prefer one source on either azimuth side before falling back to same-side candidates; filename-order stepping is only a compatibility fallback.
- For quality fusion, preserve full rectified FOV (`alpha=1`), prefer short baseline pairs, and require multiple independent depth supports plus world-normal agreement. Coverage diagnostics must z-buffer the nearest surface and reject normals facing away from the camera; raw projection overlap is not a visibility metric.

## Validation

- Run the complete unit suite for code changes:

  ```powershell
  python -m unittest discover -v
  ```

- Run CLI/tool preflight and a dry-run when dispatch or discovery changes:

  ```powershell
  python -m recon_pipeline doctor --json
  python -m recon_pipeline run data\ET runs\dry-check --backend openmvs --target mesh --dry-run
  ```

- `data/ET` and `data/kermit` may be used for real integration tests. Always choose a new ignored `runs/<descriptive-name>` output; never reuse or delete an existing result.
- For COLMAP exports, use COLMAP 4.1.1 `model_analyzer` when practical, in addition to the internal text-model validator.
- The installed Metashape edition in this environment is Standard 2.3.0, so its automated integration test is expected to stop at the Professional-license preflight.
- Open3D 0.19 on Windows requires the dedicated Python 3.12 `.venv-open3d`; do not add Open3D to the default Python 3.13 environment.
- Run `tests.test_foundation_stereo` in `.venv-open3d` when changing rectification, disparity, or fusion code. Real inference must use a new ignored `runs/` output and record the NGC model SHA-256 and active ONNX Runtime provider.
- Always run `git diff --check`, inspect `git status --short --branch`, and review the complete diff. Report tests not run and why.

## Generated files and Git hygiene

- Treat existing untracked files as user-owned. Do not use `git clean`, `git add .`, or `git add -A`.
- Local/generated paths include `.venv/`, caches, `runs/`, `output*/`, root logs, reconstruction databases, PLY/OBJ/MVS/depth-map files, downloaded archives, and extracted tool directories.
- Stage only explicit source, test, documentation, manifest, and small configuration paths when asked to commit.
- Avoid drive-by formatting and line-ending-only churn.
