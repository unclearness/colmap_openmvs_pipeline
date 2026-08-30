from __future__ import annotations

import locale
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(slots=True)
class CompletedCommand:
    label: str
    argv: list[str]
    cwd: str | None
    returncode: int | None
    duration_seconds: float
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CommandExecutionError(RuntimeError):
    def __init__(self, record: CompletedCommand, log_path: Path):
        self.record = record
        self.log_path = log_path
        super().__init__(
            f"{record.label} exited with code {record.returncode}; see {log_path}"
        )


def format_command(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(value) for value in argv])
    import shlex

    return shlex.join(str(value) for value in argv)


class CommandRunner:
    def __init__(
        self,
        log_path: Path,
        *,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        echo: bool = True,
    ) -> None:
        self.log_path = Path(log_path)
        self.dry_run = dry_run
        self.timeout_seconds = timeout_seconds
        self.echo = echo
        self.records: list[CompletedCommand] = []

    def run(
        self,
        label: str,
        argv: Sequence[str | Path],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        expected_exit_codes: tuple[int, ...] = (0,),
    ) -> CompletedCommand:
        command = [str(value) for value in argv]
        cwd_string = str(cwd) if cwd is not None else None
        rendered = format_command(command)
        if self.echo:
            prefix = "[dry-run]" if self.dry_run else "[run]"
            print(f"{prefix} {label}: {rendered}", flush=True)

        if self.dry_run:
            record = CompletedCommand(label, command, cwd_string, None, 0.0, True)
            self.records.append(record)
            return record

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = locale.getpreferredencoding(False) or "utf-8"
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd_string,
                env=dict(env) if env is not None else None,
                capture_output=True,
                text=True,
                encoding=encoding,
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            record = CompletedCommand(label, command, cwd_string, None, duration, False)
            self.records.append(record)
            self._append_log(label, rendered, exc.stdout or "", exc.stderr or "")
            raise TimeoutError(
                f"{label} exceeded timeout {self.timeout_seconds}s; see {self.log_path}"
            ) from exc

        duration = time.monotonic() - started
        record = CompletedCommand(
            label, command, cwd_string, completed.returncode, duration, False
        )
        self.records.append(record)
        self._append_log(label, rendered, completed.stdout, completed.stderr)
        if self.echo:
            print(
                f"[done] {label}: exit={completed.returncode} time={duration:.2f}s",
                flush=True,
            )
        if completed.returncode not in expected_exit_codes:
            raise CommandExecutionError(record, self.log_path)
        return record

    def _append_log(
        self,
        label: str,
        command: str,
        stdout: str | bytes | None,
        stderr: str | bytes | None,
    ) -> None:
        def as_text(value: str | bytes | None) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value

        with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"\n===== {label} =====\n{command}\n")
            out = as_text(stdout)
            err = as_text(stderr)
            if out:
                stream.write("\n--- stdout ---\n")
                stream.write(out)
                if not out.endswith("\n"):
                    stream.write("\n")
            if err:
                stream.write("\n--- stderr ---\n")
                stream.write(err)
                if not err.endswith("\n"):
                    stream.write("\n")
