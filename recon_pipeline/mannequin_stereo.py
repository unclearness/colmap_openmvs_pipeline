"""Run a fully resolved FoundationStereo configuration in its own environment."""
import argparse
import json
import sys
from pathlib import Path

from recon_pipeline.foundation_stereo import run_experiment, validate_arguments


def main():
    values = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for name in ("colmap_model", "images", "output", "model", "colmap_fused"):
        if values.get(name) is not None:
            values[name] = Path(values[name])
    args = argparse.Namespace(**values)
    validate_arguments(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
