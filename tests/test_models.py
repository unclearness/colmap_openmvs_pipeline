from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recon_pipeline.models import BackendName, PipelineConfig, Target


class PipelineConfigTests(unittest.TestCase):
    def test_commercial_backends_reject_dense_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_dir = Path(temp) / "images"
            image_dir.mkdir()
            for backend in (BackendName.REALITYSCAN, BackendName.METASHAPE):
                config = PipelineConfig(
                    image_dir=image_dir,
                    output_dir=Path(temp) / backend.value,
                    backend=backend,
                    target=Target.DENSE,
                ).normalized()
                with self.subTest(backend=backend):
                    with self.assertRaisesRegex(ValueError, "no standalone dense"):
                        config.validate()

    def test_gpu_index_may_be_a_comma_separated_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_dir = Path(temp) / "images"
            image_dir.mkdir()
            config = PipelineConfig(
                image_dir=image_dir,
                output_dir=Path(temp) / "output",
                gpu_index="0,1",
            ).normalized()
            config.validate()


if __name__ == "__main__":
    unittest.main()
