from __future__ import annotations

from typing import Protocol

from recon_pipeline.models import BackendResult, PipelineConfig
from recon_pipeline.process import CommandRunner


class ReconstructionBackend(Protocol):
    def run(self, config: PipelineConfig, runner: CommandRunner) -> BackendResult: ...
