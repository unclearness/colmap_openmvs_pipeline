from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from util import parse_colmap_version, run_and_log, version_tuple


class CompatibilityUtilTests(unittest.TestCase):
    def test_parses_colmap_411_parenthesized_version(self) -> None:
        parsed = parse_colmap_version(
            "COLMAP 4.1.1 (Commit a0d785f on 2026-07-17 with CUDA)"
        )
        self.assertEqual(parsed, ("4.1.1", "a0d785f", "2026-07-17"))
        self.assertGreater(version_tuple("4.1.1"), version_tuple("3.13.0"))

    def test_nonzero_command_is_returned_unless_raise_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log = str(Path(temp) / "command.log")
            command = [sys.executable, "-c", "raise SystemExit(7)"]
            self.assertEqual(run_and_log(command, log), 7)
            with self.assertRaises(subprocess.CalledProcessError):
                run_and_log(command, log, raise_on_error=True)


if __name__ == "__main__":
    unittest.main()
