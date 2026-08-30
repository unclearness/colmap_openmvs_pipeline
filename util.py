"""Compatibility helpers for code written against the pre-0.2 modules."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from recon_pipeline.process import CommandExecutionError, CommandRunner


COLMAP_VERSION_RE = re.compile(r"^COLMAP\s+([0-9]+(?:\.[0-9]+){1,3})", re.MULTILINE)
COLMAP_COMMIT_RE = re.compile(
    r"Commit\s+([0-9a-fA-F]+)\s+on\s+(\d{4}-\d{2}-\d{2})"
)


def parse_colmap_version(text: str) -> tuple[str | None, str | None, str | None]:
    version_match = COLMAP_VERSION_RE.search(text)
    commit_match = COLMAP_COMMIT_RE.search(text)
    version = version_match.group(1) if version_match else None
    commit = commit_match.group(1) if commit_match else None
    date = commit_match.group(2) if commit_match else None
    return version, commit, date


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def get_colmap_version(
    colmap_path: str | None = None,
) -> tuple[str, str | None, str | None]:
    executable = colmap_path or shutil.which("colmap") or shutil.which("colmap.exe")
    if not executable:
        raise FileNotFoundError("COLMAP executable was not found")
    for arguments in ((executable, "version"), (executable, "-h")):
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            check=False,
        )
        parsed = parse_colmap_version((completed.stdout or "") + (completed.stderr or ""))
        if parsed[0]:
            return parsed[0], parsed[1], parsed[2]
    raise RuntimeError("Could not parse COLMAP version output")


def is_colmap_version_at_least(
    required: str, colmap_path: str | None = None
) -> tuple[bool, str]:
    found, _, _ = get_colmap_version(colmap_path)
    return version_tuple(found) >= version_tuple(required), found


def setup_logger(logfile: str) -> logging.Logger:
    logger = logging.getLogger(f"recon_pipeline.compat.{Path(logfile).resolve()}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    return logger


def level_from_line(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("E") or " [Error" in stripped or " [ERROR" in stripped:
        return logging.ERROR
    if stripped.startswith("W") or " [Warn" in stripped or " [WARN" in stripped:
        return logging.WARNING
    return logging.INFO


def run_and_log(
    cmd: list[str],
    logfile: str = "command.log",
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    raise_on_error: bool = False,
) -> int:
    runner = CommandRunner(Path(logfile), echo=False)
    try:
        record = runner.run(
            "compat.command",
            cmd,
            cwd=Path(cwd) if cwd else None,
            env=env,
            expected_exit_codes=(0,),
        )
    except CommandExecutionError as exc:
        if raise_on_error:
            raise subprocess.CalledProcessError(exc.record.returncode or 1, cmd) from exc
        return int(exc.record.returncode or 1)
    return int(record.returncode or 0)
