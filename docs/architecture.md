# Architecture

## Pipeline selection

The CLI separates the engine from the last requested stage:

```text
backend=colmap
  images -> COLMAP SfM -> COLMAP PatchMatch/Fusion -> COLMAP mesher

backend=openmvs
  images -> COLMAP SfM/undistort -> InterfaceCOLMAP
         -> OpenMVS DensifyPointCloud -> ReconstructMesh

backend=realityscan
  images -> collision-safe flat staging -> RealityScan align
         -> optional direct model calculation
         -> exportRegistration (COLMAP text)

backend=metashape
  images -> matchPhotos/alignCameras
         -> optional buildDepthMaps/buildModel (no buildPointCloud)
         -> CamerasFormatColmap
```

`Target.DENSE` is intentionally rejected for RealityScan and Metashape. Their
mesh implementations may use depth maps internally, but the public result has
no dense point-cloud artifact.

## Common contract

Every backend returns `BackendResult` and writes a `run.json` manifest. The
common COLMAP contract is a text model under `colmap/sparse/<id>` and, when
requested/exported, matching images under `colmap/images`. Native engine data
stays under `native/`; it is not treated as interoperable unless a documented
engine converter is used.

Commands are passed as argv lists and nonzero status is fatal. Existing output
directories are never cleaned or reused. A failed run keeps its manifest and
native logs for diagnosis.

## Tool resolution

`ToolResolver` applies deterministic precedence:

1. explicit CLI path;
2. environment variable;
3. versioned installation from `tools/manifest.json`;
4. read-only system discovery (Epic manifest, Windows uninstall registry, and
   known install paths).

Downloaded assets are HTTPS-only and SHA-256 verified. ZIP extraction rejects
absolute paths, traversal, links, special files, duplicate targets, and
encrypted members. 7z archives are listed and audited before/after extraction.

## Compatibility decisions

- COLMAP 4.1 matching uses `FeatureMatching.type=SIFT_BRUTEFORCE`.
- COLMAP dense fusion writes native `fused.ply` inside the dense workspace,
  because Delaunay meshing discovers that fixed filename. A stable copy is
  published under `dense/<id>/fused.ply`.
- Delaunay is capability-checked because CGAL-enabled commands are not present
  in every COLMAP/pycolmap distribution. `auto` falls back to Poisson.
- OpenMVS 2.4 stores dense points in an external view-aware PLY. That PLY is
  explicitly passed to `ReconstructMesh` with `-p`.
- OpenMVS CUDA and CPU releases are isolated in separate runtime directories.
  In `auto` mode, a CUDA densification failure can continue with the matching
  official CPU release; later stages stay on that runtime.
- Metashape is imported only in its worker process. The worker checks
  `Metashape.app.activated` before loading images or starting processing.
