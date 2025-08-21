import subprocess
import logging
from logging.handlers import RotatingFileHandler
import sys
from threading import Thread
import locale
import re
import shutil

COLMAP_VERSION_RE = re.compile(
    r"^COLMAP\s+([0-9]+(?:\.[0-9]+){1,3})\s+--.*$", re.MULTILINE
)
COLMAP_COMMIT_RE = re.compile(r"Commit\s+([0-9a-fA-F]+)\s+on\s+(\d{4}-\d{2}-\d{2})")


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    """Run a subprocess command and return combined stdout + stderr as string"""
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return (p.stdout or "") + (p.stderr or "")


def parse_colmap_version(text: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse COLMAP version, commit hash, and date from help output.
    Example lines:
      "COLMAP 3.12.4 -- ..."
      "Commit c4a3b30 on 2025-08-05"
    """
    ver = com = date = None
    m = COLMAP_VERSION_RE.search(text)
    if m:
        ver = m.group(1)
    m2 = COLMAP_COMMIT_RE.search(text)
    if m2:
        com, date = m2.group(1), m2.group(2)
    return ver, com, date


def version_tuple(v: str) -> tuple[int, ...]:
    """Convert version string into integer tuple for comparison, e.g. '3.12.4' -> (3, 12, 4)"""
    return tuple(int(x) for x in v.split("."))


def get_colmap_version(
    colmap_path: str | None = None,
) -> tuple[str, str | None, str | None]:
    """
    Run COLMAP with -h or --help and extract version, commit, and date.
    Returns (version, commit, date).
    """
    exe = colmap_path or shutil.which("colmap") or shutil.which("colmap.exe")
    if not exe:
        raise FileNotFoundError(
            "COLMAP executable not found (not in PATH and no path provided)."
        )

    # Try both -h and --help since outputs differ depending on build/platform
    for args in ([exe, "-h"], [exe, "--help"]):
        try:
            text = _run(args)
            ver, com, date = parse_colmap_version(text)
            if ver:
                return ver, com, date
        except subprocess.TimeoutExpired:
            pass

    raise RuntimeError("Could not parse COLMAP version from output.")


def is_colmap_version_at_least(required: str, colmap_path: str | None = None) -> bool:
    """
    Check whether the installed COLMAP version is >= required version.
    Example: is_colmap_version_at_least("3.12.0") -> True/False
    """
    found, _, _ = get_colmap_version(colmap_path)
    return version_tuple(found) >= version_tuple(required), found


def setup_logger(logfile: str) -> logging.Logger:
    logger = logging.getLogger("proc")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        fh = RotatingFileHandler(
            logfile, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def level_from_line(line: str) -> int:
    s = line.strip()
    # glog like: I2025..., W2025..., E2025...
    if s.startswith(("E", "E20")):
        return logging.ERROR
    if s.startswith(("W", "W20")):
        return logging.WARNING
    # OpenMVS like: "23:05:31 [Error  ]", "[Warn   ]"
    if " [ERROR" in s or " [Error" in s:
        return logging.ERROR
    if " [WARN" in s or " [Warn" in s:
        return logging.WARNING
    return logging.INFO


def pump_stream(stream, logger: logging.Logger):
    try:
        buf = ""
        for chunk in iter(lambda: stream.read(1), ""):
            if chunk == "\r":
                buf += "\n"
                continue
            buf += chunk
            if chunk == "\n":
                for line in buf.splitlines(keepends=False):
                    if line:
                        logger.log(level_from_line(line), line.rstrip())
                buf = ""
        if buf.strip():
            for line in buf.splitlines():
                if line:
                    logger.log(level_from_line(line), line.rstrip())
    finally:
        stream.close()


def run_and_log(
    cmd, logfile="command.log", cwd=None, env=None, raise_on_error=False
) -> int:
    logger = setup_logger(logfile)
    enc = locale.getpreferredencoding(False) or "utf-8"

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding=enc,
        errors="replace",
        bufsize=0,
    )

    t_out = Thread(target=pump_stream, args=(proc.stdout, logger), daemon=True)
    t_err = Thread(target=pump_stream, args=(proc.stderr, logger), daemon=True)
    t_out.start()
    t_err.start()
    rc = proc.wait()
    t_out.join()
    t_err.join()

    if raise_on_error and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc
