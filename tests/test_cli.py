from __future__ import annotations

import unittest

from recon_pipeline.cli import build_parser


class CliTests(unittest.TestCase):
    def test_realityscan_sfm_arguments(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "images",
                "result",
                "--backend",
                "realityscan",
                "--target",
                "sfm",
                "--realityscan-quality",
                "high",
            ]
        )
        self.assertEqual(args.backend, "realityscan")
        self.assertEqual(args.target, "sfm")
        self.assertEqual(args.realityscan_quality, "high")

    def test_openmvs_cpu_switch(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "images",
                "result",
                "--backend",
                "openmvs",
                "--openmvs-variant",
                "cpu",
                "--cpu",
            ]
        )
        self.assertEqual(args.openmvs_variant, "cpu")
        self.assertTrue(args.cpu)


if __name__ == "__main__":
    unittest.main()
