from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from recon_pipeline.artifacts import (
    OutputLayout,
    require_file,
    require_images,
    validate_colmap_text_model,
    write_json,
)
from recon_pipeline.models import BackendName, BackendResult, PipelineConfig, Target
from recon_pipeline.process import CommandRunner


_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "workers" / "metashape_worker.py"


def build_metashape_command(
    *,
    config_path: Path,
    worker_script: Path = _WORKER_SCRIPT,
    executable: Path | str | None = None,
    python_executable: Path | str | None = None,
) -> list[str]:
    """Build the argv for either Metashape's ``-r`` mode or its Python module."""
    if executable is not None and python_executable is not None:
        raise ValueError(
            "Choose one Metashape runner: metashape.exe -r or an explicit Python "
            "interpreter with the Metashape module installed"
        )
    if executable is None and python_executable is None:
        raise ValueError("A Metashape executable or Python interpreter is required")

    script = str(Path(worker_script))
    job = str(Path(config_path))
    if python_executable is not None:
        return [str(python_executable), script, "--config", job]
    return [str(executable), "-r", script, "--config", job]


def discover_metashape_executable() -> Path | None:
    """Find a Professional-edition Metashape executable without launching it."""
    configured = os.environ.get("METASHAPE_EXE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    on_path = shutil.which("metashape.exe") or shutil.which("metashape")
    if on_path:
        return Path(on_path).resolve()

    program_files = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramW6432"),
    ]
    relative_candidates = (
        Path("Agisoft") / "Metashape Pro" / "metashape.exe",
        Path("Agisoft") / "Metashape Professional" / "metashape.exe",
        Path("Agisoft") / "Metashape" / "metashape.exe",
    )
    for root in dict.fromkeys(value for value in program_files if value):
        for relative in relative_candidates:
            candidate = Path(root) / relative
            if candidate.is_file():
                return candidate.resolve()
    return None


def _target_value(target: Target | str) -> str:
    return target.value if isinstance(target, Target) else str(target).lower()


def _preset_value(config: PipelineConfig) -> str:
    value = config.preset
    return value.value if hasattr(value, "value") else str(value).lower()


def _is_standard_install(path: Path | str) -> bool:
    normalized = str(path).replace("_", " ").replace("-", " ").lower()
    return "metashape standard" in normalized


class MetashapeBackend:
    """Run Metashape Professional in a separate, lazily imported worker."""

    def __init__(
        self,
        executable: Path | None = None,
        python_executable: Path | None = None,
    ) -> None:
        self.executable = Path(executable) if executable is not None else None
        self.python_executable = (
            Path(python_executable) if python_executable is not None else None
        )

    def run(self, config: PipelineConfig, runner: CommandRunner) -> BackendResult:
        target = _target_value(config.target)
        if target == Target.DENSE.value:
            raise ValueError(
                "Metashape has no standalone dense point-cloud target in this "
                "pipeline; choose sfm or mesh"
            )
        if target not in {Target.SFM.value, Target.MESH.value}:
            raise ValueError(f"Unsupported Metashape target: {target}")

        layout = OutputLayout(Path(config.output_dir))
        native_dir = layout.native / "metashape"
        project_path = native_dir / "project.psx"
        job_path = native_dir / "job.json"
        result_path = native_dir / "result.json"
        colmap_model = layout.colmap_sparse / "0"
        mesh_dir = layout.mesh / "0"
        mesh_path = mesh_dir / "mesh.obj"
        dry_run = bool(config.dry_run or runner.dry_run)

        if config.metashape_exe is not None or config.metashape_python is not None:
            executable = config.metashape_exe
            python_executable = config.metashape_python
        else:
            executable = self.executable
            python_executable = self.python_executable
        if executable is None and python_executable is None:
            executable = discover_metashape_executable()
            if executable is None and dry_run:
                executable = Path("metashape.exe")

        self._preflight_runner(
            executable=executable,
            python_executable=python_executable,
            dry_run=dry_run,
        )

        payload: dict[str, Any] = {
            "schema_version": 1,
            "image_dir": str(Path(config.image_dir).resolve()),
            "target": target,
            "preset": _preset_value(config),
            "texture": bool(config.texture),
            "copy_images": bool(config.copy_images),
            "project_path": str(project_path.resolve()),
            "mesh_path": str(mesh_path.resolve()),
            "colmap_model_path": str(colmap_model.resolve()),
            "colmap_images_path": str(layout.colmap_images.resolve()),
            "result_path": str(result_path.resolve()),
        }
        argv = build_metashape_command(
            config_path=job_path.resolve(),
            executable=executable,
            python_executable=python_executable,
        )

        if not dry_run:
            native_dir.mkdir(parents=True, exist_ok=True)
            layout.colmap.mkdir(parents=True, exist_ok=True)
            if target == Target.MESH.value:
                mesh_dir.mkdir(parents=True, exist_ok=True)
            write_json(job_path, payload)

        if not dry_run or runner.dry_run:
            runner.run("metashape.run", argv, cwd=Path(config.output_dir))

        metadata: dict[str, Any] = {
            "runner": "python" if python_executable is not None else "metashape-r",
            "command": argv,
            "job_config": str(job_path),
            "dry_run": dry_run,
        }
        if not dry_run:
            require_file(project_path, "Metashape project")
            metadata["colmap"] = validate_colmap_text_model(colmap_model)
            if config.copy_images:
                metadata["colmap_images"] = len(require_images(layout.colmap_images))
            if target == Target.MESH.value:
                require_file(mesh_path, "Metashape mesh")
            require_file(result_path, "Metashape worker result")
            worker_result = json.loads(result_path.read_text(encoding="utf-8"))
            metadata.update(worker_result)

        return BackendResult(
            backend=BackendName.METASHAPE,
            target=config.target,
            output_dir=Path(config.output_dir),
            colmap_models=[colmap_model],
            dense_clouds=[],
            meshes=[mesh_path] if target == Target.MESH.value else [],
            native_project=project_path,
            metadata=metadata,
        )

    @staticmethod
    def _preflight_runner(
        *,
        executable: Path | None,
        python_executable: Path | None,
        dry_run: bool,
    ) -> None:
        if executable is not None and python_executable is not None:
            raise ValueError(
                "Both Metashape execution modes were configured. Choose either "
                "metashape.exe -r or --metashape-python."
            )
        if executable is None and python_executable is None:
            raise FileNotFoundError(
                "Metashape Professional was not found. Set --metashape-exe to its "
                "metashape.exe, or --metashape-python to a Python interpreter with "
                "the licensed Metashape module installed. Metashape Standard cannot "
                "run Python scripts."
            )
        selected = python_executable if python_executable is not None else executable
        assert selected is not None
        if _is_standard_install(selected):
            raise RuntimeError(
                "Metashape Standard was selected, but Python scripting is a "
                "Professional-edition feature. Install and activate Metashape "
                "Professional, then select its executable."
            )
        if dry_run:
            return
        if not selected.is_file():
            raise FileNotFoundError(f"Metashape runner not found: {selected}")


__all__ = [
    "MetashapeBackend",
    "build_metashape_command",
    "discover_metashape_executable",
]
