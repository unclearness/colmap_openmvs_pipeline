from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recon_pipeline.models import BackendName, PipelineConfig, Target
from recon_pipeline.orchestrator import execute_pipeline


class OrchestratorTests(unittest.TestCase):
    def test_output_inside_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"a")
            (images / "b.jpg").write_bytes(b"b")
            config = PipelineConfig(
                image_dir=images,
                output_dir=images / "output",
                backend=BackendName.COLMAP,
                target=Target.SFM,
                dry_run=True,
            )
            with self.assertRaisesRegex(ValueError, "must not be inside"):
                execute_pipeline(config)


if __name__ == "__main__":
    unittest.main()
