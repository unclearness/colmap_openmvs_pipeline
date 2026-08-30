from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from recon_pipeline.artifacts import (
    OutputLayout,
    ensure_new_output,
    require_file,
    require_images,
    validate_mesh,
    validate_colmap_text_model,
    write_json,
)
from recon_pipeline.backends.colmap import ColmapBackend
from recon_pipeline.backends.openmvs import OpenMVSBackend
from recon_pipeline.models import BackendName, BackendResult, PipelineConfig
from recon_pipeline.process import CommandRunner
from recon_pipeline.tooling import ToolInfo, ToolResolver


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _tool_path(info: ToolInfo, purpose: str) -> Path:
    if not info.available or info.path is None:
        note = info.details.get("note") if isinstance(info.details, dict) else None
        suffix = f" ({note})" if note else ""
        raise FileNotFoundError(f"{purpose} was not found{suffix}")
    return Path(info.path)


def _resolve_backend(
    config: PipelineConfig, resolver: ToolResolver
) -> tuple[object, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if config.backend is BackendName.COLMAP:
        colmap = resolver.resolve_colmap(config.colmap_exe)
        metadata["colmap"] = colmap.as_dict()
        return ColmapBackend(_tool_path(colmap, "COLMAP")), metadata

    if config.backend is BackendName.OPENMVS:
        colmap = resolver.resolve_colmap(config.colmap_exe)
        openmvs = resolver.resolve_openmvs(
            config.openmvs_dir, variant=config.openmvs_variant
        )
        fallback = None
        if config.openmvs_variant == "auto" and openmvs.variant == "cuda":
            cpu_info = resolver.resolve_openmvs(variant="cpu")
            if cpu_info.available and cpu_info.path is not None:
                fallback = Path(cpu_info.path)
                metadata["openmvs_cpu_fallback"] = cpu_info.as_dict()
        metadata["colmap"] = colmap.as_dict()
        metadata["openmvs"] = openmvs.as_dict()
        return (
            OpenMVSBackend(
                _tool_path(colmap, "COLMAP"),
                _tool_path(openmvs, "OpenMVS"),
                variant=openmvs.variant or "cpu",
                fallback_directory=fallback,
            ),
            metadata,
        )

    if config.backend is BackendName.REALITYSCAN:
        from recon_pipeline.backends.realityscan import RealityScanBackend

        realityscan = resolver.resolve_realityscan(config.realityscan_exe)
        metadata["realityscan"] = realityscan.as_dict()
        return RealityScanBackend(_tool_path(realityscan, "RealityScan")), metadata

    from recon_pipeline.backends.metashape import MetashapeBackend

    if config.metashape_python is not None:
        metadata["metashape"] = {
            "name": "metashape",
            "version": None,
            "path": str(config.metashape_python),
            "variant": "python-module",
            "available": config.metashape_python.is_file(),
            "details": {
                "note": "Professional API activation is checked inside the worker"
            },
        }
        return MetashapeBackend(python_executable=config.metashape_python), metadata

    metashape = resolver.resolve_metashape(config.metashape_exe)
    metadata["metashape"] = metashape.as_dict()
    edition = str(metashape.details.get("edition", ""))
    if "standard" in edition.lower():
        raise RuntimeError(
            "Agisoft Metashape Standard is installed, but automated Python "
            "processing requires Metashape Professional. A Professional/trial "
            "license or --metashape-python runner is required."
        )
    return MetashapeBackend(executable=_tool_path(metashape, "Metashape")), metadata


def execute_pipeline(config: PipelineConfig) -> BackendResult:
    config.normalized()
    config.validate()
    require_images(config.image_dir)
    if config.output_dir == config.image_dir or config.image_dir in config.output_dir.parents:
        raise ValueError("Output directory must not be inside the input image directory")

    # Resolve capabilities before creating the output tree so a missing tool or
    # unsupported license does not leave behind a directory that blocks retry.
    resolver = ToolResolver(PROJECT_ROOT)
    backend, tools = _resolve_backend(config, resolver)
    ensure_new_output(config.output_dir, dry_run=config.dry_run)

    layout = OutputLayout(config.output_dir)
    if not config.dry_run:
        layout.create()
    runner = CommandRunner(
        layout.logs / "pipeline.log",
        dry_run=config.dry_run,
        timeout_seconds=config.timeout_seconds,
    )
    started = datetime.now(timezone.utc).isoformat()

    base_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started,
        "config": _json_value(asdict(config)),
        "tools": tools,
        "commands": [],
    }
    if not config.dry_run:
        write_json(layout.manifest, base_manifest)

    try:
        result = backend.run(config, runner)  # type: ignore[attr-defined]
        if not config.dry_run:
            for model in result.colmap_models:
                validate_colmap_text_model(model)
            for cloud in result.dense_clouds:
                require_file(cloud, "dense point cloud")
            for mesh in result.meshes:
                validate_mesh(mesh)
            if result.native_project is not None:
                require_file(result.native_project, "native project", allow_empty=False)
        final_manifest = {
            **base_manifest,
            "status": "dry-run" if config.dry_run else "complete",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "commands": [record.to_dict() for record in runner.records],
            "result": result.to_dict(),
        }
        result.metadata.setdefault("tools", tools)
        result.metadata.setdefault("manifest", str(layout.manifest))
        if not config.dry_run:
            write_json(layout.manifest, final_manifest)
        return result
    except Exception as exc:
        if not config.dry_run:
            write_json(
                layout.manifest,
                {
                    **base_manifest,
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "commands": [record.to_dict() for record in runner.records],
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        raise


def _doctor_info(call: Callable[[], ToolInfo]) -> dict[str, Any]:
    try:
        return call().as_dict()
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "path": None,
            "variant": None,
            "details": {},
            "note": str(exc),
        }


def doctor(
    *,
    colmap_exe: Path | None = None,
    openmvs_dir: Path | None = None,
    openmvs_variant: str = "auto",
    realityscan_exe: Path | None = None,
    metashape_exe: Path | None = None,
) -> dict[str, dict[str, Any]]:
    resolver = ToolResolver(PROJECT_ROOT)
    report = {
        "colmap": _doctor_info(lambda: resolver.resolve_colmap(colmap_exe)),
        "openmvs": _doctor_info(
            lambda: resolver.resolve_openmvs(openmvs_dir, variant=openmvs_variant)
        ),
        "realityscan": _doctor_info(
            lambda: resolver.resolve_realityscan(realityscan_exe)
        ),
        "metashape": _doctor_info(
            lambda: resolver.resolve_metashape(metashape_exe)
        ),
    }
    metashape = report["metashape"]
    edition = str(metashape.get("details", {}).get("edition", ""))
    if "standard" in edition.lower():
        metashape["note"] = (
            "Installed Standard edition supports manual GUI workflows, but the "
            "automated backend requires Metashape Professional Python scripting."
        )
    return report
