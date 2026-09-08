from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class BackendName(StrEnum):
    COLMAP = "colmap"
    OPENMVS = "openmvs"
    REALITYSCAN = "realityscan"
    METASHAPE = "metashape"


class Target(StrEnum):
    SFM = "sfm"
    DENSE = "dense"
    MESH = "mesh"


class Preset(StrEnum):
    FAST = "fast"
    NORMAL = "normal"
    HIGH = "high"


class Matcher(StrEnum):
    SEQUENTIAL = "sequential"
    EXHAUSTIVE = "exhaustive"


class Mesher(StrEnum):
    AUTO = "auto"
    DELAUNAY = "delaunay"
    POISSON = "poisson"


@dataclass(slots=True)
class PipelineConfig:
    image_dir: Path
    output_dir: Path
    backend: BackendName = BackendName.COLMAP
    target: Target = Target.MESH
    preset: Preset = Preset.NORMAL
    matcher: Matcher = Matcher.SEQUENTIAL
    camera_mode: int = 1
    camera_model: str = "SIMPLE_RADIAL"
    intrinsic_prior: tuple[float, ...] | None = None
    use_gpu: bool = True
    gpu_index: str = "-1"
    forward_motion: bool = False
    fixed_intrinsics: bool = False
    mesher: Mesher = Mesher.AUTO
    refine_mesh: bool = False
    texture: bool = False
    copy_images: bool = True
    dry_run: bool = False
    timeout_seconds: float | None = None
    colmap_exe: Path | None = None
    openmvs_dir: Path | None = None
    openmvs_variant: str = "auto"
    realityscan_exe: Path | None = None
    realityscan_quality: str = "normal"
    realityscan_no_distortion: bool = False
    realityscan_distortion_model: str | None = None
    realityscan_distortion_prior: str | None = None
    realityscan_shared_intrinsics: bool = False
    realityscan_sensitive_alignment: bool = False
    metashape_exe: Path | None = None
    metashape_python: Path | None = None

    def normalized(self) -> PipelineConfig:
        self.image_dir = self.image_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        for name in (
            "colmap_exe",
            "openmvs_dir",
            "realityscan_exe",
            "metashape_exe",
            "metashape_python",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.expanduser().resolve())
        return self

    def validate(self) -> None:
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if self.camera_mode not in {0, 1, 2, 3}:
            raise ValueError("camera_mode must be one of 0, 1, 2, or 3")
        if self.backend in {BackendName.REALITYSCAN, BackendName.METASHAPE}:
            if self.target is Target.DENSE:
                raise ValueError(
                    f"{self.backend.value} has no standalone dense point-cloud target; "
                    "choose sfm or mesh"
                )
        if self.realityscan_quality not in {"normal", "high"}:
            raise ValueError("realityscan_quality must be normal or high")
        distortion_models = {
            "division",
            "brown3",
            "brown4",
            "brown3-tangential2",
            "brown4-tangential2",
        }
        if (
            self.realityscan_distortion_model is not None
            and self.realityscan_distortion_model not in distortion_models
        ):
            raise ValueError("unsupported RealityScan distortion model")
        if (
            self.realityscan_distortion_prior is not None
            and self.realityscan_distortion_prior
            not in {"unknown", "approximate", "fixed"}
        ):
            raise ValueError("unsupported RealityScan distortion prior")
        if self.realityscan_no_distortion and (
            self.realityscan_distortion_model is not None
            or self.realityscan_distortion_prior is not None
        ):
            raise ValueError(
                "realityscan_no_distortion cannot be combined with an explicit "
                "distortion model or prior"
            )
        if self.openmvs_variant not in {"auto", "cuda", "cpu"}:
            raise ValueError("openmvs_variant must be auto, cuda, or cpu")
        try:
            indices = [int(value.strip()) for value in self.gpu_index.split(",")]
        except ValueError as exc:
            raise ValueError("gpu_index must be an integer or comma-separated integers") from exc
        if not indices:
            raise ValueError("gpu_index cannot be empty")


@dataclass(slots=True)
class BackendResult:
    backend: BackendName | str
    target: Target | str
    output_dir: Path
    colmap_models: list[Path] = field(default_factory=list)
    dense_clouds: list[Path] = field(default_factory=list)
    meshes: list[Path] = field(default_factory=list)
    native_project: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": str(self.backend),
            "target": str(self.target),
            "output_dir": str(self.output_dir),
            "colmap_models": [str(path) for path in self.colmap_models],
            "dense_clouds": [str(path) for path in self.dense_clouds],
            "meshes": [str(path) for path in self.meshes],
            "native_project": str(self.native_project) if self.native_project else None,
            "metadata": self.metadata,
        }
