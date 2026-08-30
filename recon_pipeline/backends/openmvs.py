from __future__ import annotations

import shutil
from pathlib import Path

from recon_pipeline.artifacts import OutputLayout, require_file, validate_mesh
from recon_pipeline.backends.colmap import ColmapBackend
from recon_pipeline.models import BackendName, PipelineConfig, Preset, Target
from recon_pipeline.process import CommandExecutionError, CommandRunner


class OpenMVSBackend(ColmapBackend):
    """COLMAP SfM followed by OpenMVS 2.4 dense reconstruction and meshing."""

    backend_name = BackendName.OPENMVS

    def __init__(
        self,
        colmap_executable: Path,
        openmvs_directory: Path,
        *,
        variant: str = "cpu",
        fallback_directory: Path | None = None,
    ) -> None:
        super().__init__(colmap_executable)
        self.openmvs_directory = Path(openmvs_directory)
        self.variant = variant
        self.fallback_directory = (
            Path(fallback_directory) if fallback_directory is not None else None
        )
        self.fallback_used = False

    def run(self, config: PipelineConfig, runner: CommandRunner):
        if not runner.dry_run:
            for name in (
                "InterfaceCOLMAP.exe",
                "DensifyPointCloud.exe",
                "ReconstructMesh.exe",
            ):
                require_file(self.openmvs_directory / name, f"OpenMVS {name}")
        if not runner.dry_run and self.fallback_directory is not None:
            for name in ("DensifyPointCloud.exe", "ReconstructMesh.exe"):
                require_file(
                    self.fallback_directory / name, f"OpenMVS CPU fallback {name}"
                )
        result = super().run(config, runner)
        result.metadata["openmvs_variant"] = self.variant
        result.metadata["openmvs_cpu_fallback_used"] = self.fallback_used
        return result

    def _run_dense_model(
        self,
        config: PipelineConfig,
        runner: CommandRunner,
        layout: OutputLayout,
        model_id: str,
        workspace: Path,
    ) -> tuple[Path, Path | None]:
        work = layout.native / "openmvs" / model_id
        scene = work / "scene.mvs"
        dense_scene = work / "scene_dense.mvs"
        dense_ply = work / "scene_dense.ply"
        mesh_ply = work / "scene_mesh.ply"
        if not runner.dry_run:
            work.mkdir(parents=True, exist_ok=True)

        runner.run(
            f"openmvs.interface_colmap.{model_id}",
            [
                self._exe("InterfaceCOLMAP.exe"),
                "-i",
                workspace,
                "-o",
                scene,
                "-w",
                work,
            ],
            cwd=work,
        )

        active_directory = (
            self.fallback_directory
            if self.fallback_used and self.fallback_directory is not None
            else self.openmvs_directory
        )
        densify = [
            self._exe("DensifyPointCloud.exe", active_directory),
            "-i",
            scene,
            "-o",
            dense_scene,
            "-w",
            work,
        ]
        if self.variant == "cuda" and active_directory == self.openmvs_directory:
            if "," in config.gpu_index:
                raise ValueError("OpenMVS accepts a single CUDA device index")
            densify.extend(
                ["--cuda-device", config.gpu_index if config.use_gpu else "-2"]
            )
        densify.extend(self._densify_preset_arguments(config))
        try:
            runner.run(f"openmvs.densify.{model_id}", densify, cwd=work)
        except CommandExecutionError:
            if self.fallback_directory is None:
                raise
            active_directory = self.fallback_directory
            fallback_command = [
                self._exe("DensifyPointCloud.exe", active_directory),
                "-i",
                scene,
                "-o",
                dense_scene,
                "-w",
                work,
                *self._densify_preset_arguments(config),
            ]
            runner.run(
                f"openmvs.densify_cpu_fallback.{model_id}",
                fallback_command,
                cwd=work,
            )
            self.fallback_used = True
        if not runner.dry_run:
            require_file(dense_scene, "OpenMVS dense scene")
            require_file(dense_ply, "OpenMVS dense point cloud")

        dense_dir = layout.dense / model_id
        dense_result = dense_dir / "fused.ply"
        if not runner.dry_run:
            dense_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dense_ply, dense_result)
        if config.target is Target.DENSE:
            return dense_result, None

        runner.run(
            f"openmvs.reconstruct_mesh.{model_id}",
            [
                self._exe("ReconstructMesh.exe", active_directory),
                "-i",
                dense_scene,
                "-p",
                dense_ply,
                "-o",
                mesh_ply,
                "-w",
                work,
            ],
            cwd=work,
        )
        current_scene = dense_scene
        current_mesh = mesh_ply
        if not runner.dry_run:
            validate_mesh(current_mesh)

        if config.refine_mesh:
            refined_ply = work / "scene_mesh_refined.ply"
            runner.run(
                f"openmvs.refine_mesh.{model_id}",
                [
                    self._exe("RefineMesh.exe", active_directory),
                    "-i",
                    current_scene,
                    "-m",
                    current_mesh,
                    "-o",
                    refined_ply,
                    "-w",
                    work,
                ],
                cwd=work,
            )
            current_mesh = refined_ply
            if not runner.dry_run:
                validate_mesh(current_mesh)

        if config.texture:
            textured_ply = work / "scene_mesh_textured.ply"
            runner.run(
                f"openmvs.texture_mesh.{model_id}",
                [
                    self._exe("TextureMesh.exe", active_directory),
                    "-i",
                    current_scene,
                    "-m",
                    current_mesh,
                    "-o",
                    textured_ply,
                    "-w",
                    work,
                ],
                cwd=work,
            )
            current_mesh = textured_ply
            if not runner.dry_run:
                validate_mesh(current_mesh)

        result_dir = layout.mesh / model_id
        result_mesh = result_dir / "mesh.ply"
        if not runner.dry_run:
            result_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_mesh, result_mesh)
            if config.texture:
                for companion in work.glob(f"{current_mesh.stem}*.png"):
                    shutil.copy2(companion, result_dir / companion.name)
                for companion in work.glob(f"{current_mesh.stem}*.jpg"):
                    shutil.copy2(companion, result_dir / companion.name)
            validate_mesh(result_mesh)
        return dense_result, result_mesh

    @staticmethod
    def _densify_preset_arguments(config: PipelineConfig) -> list[str]:
        if config.preset is Preset.FAST:
            return [
                "--max-resolution",
                "1024",
                "--number-views",
                "4",
                "--iters",
                "2",
            ]
        if config.preset is Preset.HIGH:
            return [
                "--resolution-level",
                "0",
                "--number-views",
                "8",
                "--iters",
                "4",
            ]
        return []

    def _exe(self, name: str, directory: Path | None = None) -> Path:
        return (directory or self.openmvs_directory) / name
