from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from recon_pipeline.artifacts import (
    OutputLayout,
    copy_image_tree,
    numeric_model_dirs,
    require_file,
    validate_mesh,
    validate_colmap_text_model,
)
from recon_pipeline.models import (
    BackendName,
    BackendResult,
    Matcher,
    Mesher,
    PipelineConfig,
    Preset,
    Target,
)
from recon_pipeline.process import CommandRunner


class ColmapBackend:
    """COLMAP 4.1 command-line backend for SfM, dense MVS, and meshing."""

    backend_name = BackendName.COLMAP

    def __init__(self, executable: Path) -> None:
        self.executable = Path(executable)

    def run(self, config: PipelineConfig, runner: CommandRunner) -> BackendResult:
        if not runner.dry_run:
            require_file(self.executable, "COLMAP executable")

        layout = OutputLayout(config.output_dir)
        native_root = layout.native / "colmap"
        database_path = native_root / "database.db"
        sparse_root = native_root / "sparse"
        dense_root = native_root / "dense"
        if not runner.dry_run:
            sparse_root.mkdir(parents=True, exist_ok=True)

        runner.run(
            "colmap.database_creator",
            self._command(
                "database_creator",
                "--database_path",
                database_path,
            ),
        )

        extraction = self._command(
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            config.image_dir,
            "--camera_mode",
            str(config.camera_mode),
            "--ImageReader.camera_model",
            config.camera_model,
            "--FeatureExtraction.type",
            "SIFT",
            "--FeatureExtraction.use_gpu",
            "1" if config.use_gpu else "0",
        )
        if config.use_gpu:
            extraction.extend(["--FeatureExtraction.gpu_index", config.gpu_index])
        if config.intrinsic_prior:
            extraction.extend(
                [
                    "--ImageReader.camera_params",
                    ",".join(str(value) for value in config.intrinsic_prior),
                ]
            )
        runner.run("colmap.feature_extractor", extraction)

        matcher_command = (
            "sequential_matcher"
            if config.matcher is Matcher.SEQUENTIAL
            else "exhaustive_matcher"
        )
        matching = self._command(
            matcher_command,
            "--database_path",
            database_path,
            "--FeatureMatching.type",
            "SIFT_BRUTEFORCE",
            "--FeatureMatching.use_gpu",
            "1" if config.use_gpu else "0",
        )
        if config.use_gpu:
            matching.extend(["--FeatureMatching.gpu_index", config.gpu_index])
        runner.run(f"colmap.{matcher_command}", matching)

        mapping = self._command(
            "mapper",
            "--database_path",
            database_path,
            "--image_path",
            config.image_dir,
            "--output_path",
            sparse_root,
        )
        if config.forward_motion:
            mapping.extend(
                [
                    "--Mapper.init_max_forward_motion",
                    "1.0",
                    "--Mapper.init_min_tri_angle",
                    "0.5",
                    "--Mapper.tri_create_max_angle_error",
                    "0.5",
                    "--Mapper.filter_min_tri_angle",
                    "0.5",
                ]
            )
        if config.fixed_intrinsics:
            mapping.extend(
                [
                    "--Mapper.ba_refine_focal_length",
                    "0",
                    "--Mapper.ba_refine_principal_point",
                    "0",
                    "--Mapper.ba_refine_extra_params",
                    "0",
                ]
            )
        runner.run("colmap.mapper", mapping)

        models = [sparse_root / "0"] if runner.dry_run else numeric_model_dirs(sparse_root)
        if not models:
            raise RuntimeError(f"COLMAP mapper produced no sparse models in {sparse_root}")

        exported_models: list[Path] = []
        model_stats: dict[str, dict[str, int]] = {}
        for model in models:
            export_model = layout.colmap_sparse / model.name
            if not runner.dry_run:
                export_model.mkdir(parents=True, exist_ok=True)
            runner.run(
                f"colmap.model_converter.{model.name}",
                self._command(
                    "model_converter",
                    "--input_path",
                    model,
                    "--output_path",
                    export_model,
                    "--output_type",
                    "TXT",
                ),
            )
            if not runner.dry_run:
                model_stats[model.name] = validate_colmap_text_model(export_model)
            exported_models.append(export_model)

        if config.copy_images and not runner.dry_run:
            copy_image_tree(config.image_dir, layout.colmap_images)

        result = BackendResult(
            backend=self.backend_name,
            target=config.target,
            output_dir=config.output_dir,
            colmap_models=exported_models,
            native_project=database_path,
            metadata={
                "colmap_models": model_stats,
                "database": str(database_path),
                "dense_point_cloud_generated": config.target in {Target.DENSE, Target.MESH},
            },
        )
        if config.target is Target.SFM:
            return result

        if not config.use_gpu:
            raise ValueError("COLMAP PatchMatch dense reconstruction requires the CUDA build")

        dense_models = (
            models
            if runner.dry_run
            else [
                model
                for model in models
                if model_stats.get(model.name, {}).get("images", 0) >= 3
            ]
        )
        skipped_models = [model.name for model in models if model not in dense_models]
        result.metadata["skipped_dense_models"] = skipped_models
        if not dense_models:
            raise RuntimeError(
                "No sparse model has the minimum three registered images required "
                "for dense reconstruction"
            )

        for model in dense_models:
            workspace = dense_root / model.name
            if not runner.dry_run:
                workspace.parent.mkdir(parents=True, exist_ok=True)
            runner.run(
                f"colmap.image_undistorter.{model.name}",
                self._command(
                    "image_undistorter",
                    "--image_path",
                    config.image_dir,
                    "--input_path",
                    model,
                    "--output_path",
                    workspace,
                    "--output_type",
                    "COLMAP",
                    "--copy_policy",
                    "copy",
                ),
            )
            dense_cloud, mesh = self._run_dense_model(
                config, runner, layout, model.name, workspace
            )
            result.dense_clouds.append(dense_cloud)
            if mesh is not None:
                result.meshes.append(mesh)

        return result

    def _run_dense_model(
        self,
        config: PipelineConfig,
        runner: CommandRunner,
        layout: OutputLayout,
        model_id: str,
        workspace: Path,
    ) -> tuple[Path, Path | None]:
        patch_match = self._command(
            "patch_match_stereo",
            "--workspace_path",
            workspace,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.gpu_index",
            config.gpu_index,
            "--PatchMatchStereo.geom_consistency",
            "1",
        )
        if config.preset is Preset.FAST:
            patch_match.extend(
                [
                    "--PatchMatchStereo.max_image_size",
                    "1024",
                    "--PatchMatchStereo.window_step",
                    "2",
                    "--PatchMatchStereo.num_iterations",
                    "3",
                    "--PatchMatchStereo.window_radius",
                    "3",
                    "--PatchMatchStereo.num_samples",
                    "8",
                ]
            )
        elif config.preset is Preset.HIGH:
            patch_match.extend(
                [
                    "--PatchMatchStereo.num_iterations",
                    "7",
                    "--PatchMatchStereo.num_samples",
                    "20",
                ]
            )
        runner.run(f"colmap.patch_match_stereo.{model_id}", patch_match)

        # Delaunay meshing discovers fused.ply inside the dense workspace even
        # when an explicit output path was used for StereoFusion. Keep the
        # native file there and copy it to the stable public artifact layout.
        workspace_cloud = workspace / "fused.ply"
        dense_dir = layout.dense / model_id
        dense_cloud = dense_dir / "fused.ply"
        if not runner.dry_run:
            dense_dir.mkdir(parents=True, exist_ok=True)
        runner.run(
            f"colmap.stereo_fusion.{model_id}",
            self._command(
                "stereo_fusion",
                "--workspace_path",
                workspace,
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--output_type",
                "PLY",
                "--output_path",
                workspace_cloud,
            ),
        )
        if not runner.dry_run:
            require_file(workspace_cloud, "COLMAP workspace fused point cloud")
            shutil.copy2(workspace_cloud, dense_cloud)
            require_file(dense_cloud, "COLMAP fused point cloud")
        if config.target is Target.DENSE:
            return dense_cloud, None

        requested_mesher = config.mesher
        delaunay_available = (
            True if runner.dry_run else self._supports_command("delaunay_mesher")
        )
        selected_mesher = self.select_mesher(requested_mesher, delaunay_available)
        mesh_dir = layout.mesh / model_id
        mesh_path = mesh_dir / f"mesh_{selected_mesher.value}.ply"
        if not runner.dry_run:
            mesh_dir.mkdir(parents=True, exist_ok=True)
        if selected_mesher is Mesher.POISSON:
            mesh_command = self._command(
                "poisson_mesher",
                "--input_path",
                dense_cloud,
                "--output_path",
                mesh_path,
            )
            poisson_settings = {
                Preset.FAST: ("9", "1"),
                Preset.NORMAL: ("11", "5"),
                Preset.HIGH: ("13", "10"),
            }
            depth, trim = poisson_settings[config.preset]
            mesh_command.extend(
                [
                    "--PoissonMeshing.depth",
                    depth,
                    "--PoissonMeshing.trim",
                    trim,
                ]
            )
        else:
            mesh_command = self._command(
                "delaunay_mesher",
                "--input_path",
                workspace,
                "--input_type",
                "dense",
                "--output_path",
                mesh_path,
            )
        runner.run(f"colmap.{selected_mesher.value}_mesher.{model_id}", mesh_command)
        if not runner.dry_run:
            validate_mesh(mesh_path)
        return dense_cloud, mesh_path

    @staticmethod
    def select_mesher(requested: Mesher, delaunay_available: bool) -> Mesher:
        if requested is Mesher.AUTO:
            return Mesher.DELAUNAY if delaunay_available else Mesher.POISSON
        if requested is Mesher.DELAUNAY and not delaunay_available:
            raise RuntimeError(
                "This COLMAP distribution was built without CGAL/Delaunay meshing. "
                "Use --mesher poisson; no source build is required."
            )
        return requested

    def _supports_command(self, command: str) -> bool:
        try:
            completed = subprocess.run(
                [str(self.executable), "-h"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        return any(line.strip() == command for line in output.splitlines())

    def _command(self, subcommand: str, *arguments: object) -> list[str]:
        return [
            str(self.executable),
            subcommand,
            "--log_target",
            "stderr",
            *(str(argument) for argument in arguments),
        ]
