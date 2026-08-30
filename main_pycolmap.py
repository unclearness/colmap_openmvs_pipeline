"""Deprecated entry point retained for callers of the pre-0.2 layout.

The refactored pipeline uses the versioned COLMAP command-line distribution.
Invoke this file with the same arguments as ``python -m recon_pipeline``.
"""

import sys

from recon_pipeline.cli import main


if __name__ == "__main__":
    print(
        "warning: main_pycolmap.py is deprecated; use `python -m recon_pipeline`",
        file=sys.stderr,
    )
    raise SystemExit(main())
