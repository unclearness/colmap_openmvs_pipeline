"""Discovery and installation helpers for external reconstruction tools.

The module intentionally depends only on the Python standard library.  It is
safe to import on machines where none of the supported applications are
installed; discovery is read-only and reports unavailable tools as data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MANIFEST = Path("tools") / "manifest.json"

_COLMAP_ENV_VARS = ("COLMAP_EXE", "COLMAP_PATH")
_OPENMVS_ENV_VARS = ("OPENMVS_BIN", "OPENMVS_PATH")
_REALITYSCAN_ENV_VARS = ("REALITYSCAN_EXE", "REALITYSCAN_PATH")
_METASHAPE_ENV_VARS = ("METASHAPE_EXE", "METASHAPE_PATH")

_OPENMVS_COMMANDS = (
    "InterfaceCOLMAP.exe",
    "DensifyPointCloud.exe",
    "ReconstructMesh.exe",
    "RefineMesh.exe",
    "TextureMesh.exe",
)


class ManifestError(ValueError):
    """Raised when the checked-in tool manifest is malformed."""


class ArchiveSafetyError(ValueError):
    """Raised when an archive contains an unsafe member."""


@dataclass(slots=True)
class ToolInfo:
    """Result of resolving one external tool.

    ``path`` points to the COLMAP/proprietary executable, and to the directory
    containing the OpenMVS executables.  Callers must check ``available``
    before launching it.
    """

    name: str
    version: str | None
    path: Path | None
    variant: str | None
    available: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata suitable for a doctor command."""

        return {
            "name": self.name,
            "version": self.version,
            "path": str(self.path) if self.path is not None else None,
            "variant": self.variant,
            "available": self.available,
            "details": self.details,
        }


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a tool manifest."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Tool manifest not found: {manifest_path}") from None
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in tool manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ManifestError("Tool manifest must be an object with schema_version 1")
    tools = payload.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise ManifestError("Tool manifest must contain a non-empty 'tools' object")

    for tool_name, tool in tools.items():
        if not isinstance(tool_name, str) or not isinstance(tool, dict):
            raise ManifestError("Each manifest tool must be a named object")
        version = tool.get("version")
        variants = tool.get("variants")
        default_variant = tool.get("default_variant")
        if not isinstance(version, str) or not version:
            raise ManifestError(f"Tool {tool_name!r} has no version")
        if not isinstance(variants, dict) or not variants:
            raise ManifestError(f"Tool {tool_name!r} has no variants")
        if default_variant not in variants:
            raise ManifestError(
                f"Tool {tool_name!r} default_variant is not present in variants"
            )
        for variant_name, asset in variants.items():
            _validate_manifest_asset(tool_name, variant_name, asset)
    return payload


def _validate_manifest_asset(tool_name: str, variant: str, asset: Any) -> None:
    if not isinstance(variant, str) or not isinstance(asset, dict):
        raise ManifestError(f"Invalid variant in tool {tool_name!r}")
    for key in ("url", "sha256", "archive", "install_dir", "probe"):
        if not isinstance(asset.get(key), str) or not asset[key]:
            raise ManifestError(f"{tool_name}:{variant} is missing {key!r}")
    if not asset["url"].startswith("https://"):
        raise ManifestError(f"{tool_name}:{variant} URL must use HTTPS")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", asset["sha256"]):
        raise ManifestError(f"{tool_name}:{variant} has an invalid SHA-256")
    if asset.get("kind", "archive") not in {"archive", "file"}:
        raise ManifestError(f"{tool_name}:{variant} has an invalid asset kind")
    _validated_relative_parts(asset["archive"], label="archive name")
    _validated_relative_parts(asset["install_dir"], label="install directory")
    _validated_relative_parts(asset["probe"], label="probe path")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    """Verify a file digest and return the actual digest."""

    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Invalid expected SHA-256: {expected!r}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {Path(path)}: expected {expected.lower()}, got {actual}"
        )
    return actual


def _validated_relative_parts(name: str, *, label: str = "archive member") -> tuple[str, ...]:
    if not name or "\x00" in name:
        raise ArchiveSafetyError(f"Unsafe {label}: {name!r}")
    normalized = name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(name)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ArchiveSafetyError(f"Absolute {label} is not allowed: {name!r}")
    parts = tuple(part for part in posix_path.parts if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise ArchiveSafetyError(f"Traversal in {label}: {name!r}")
    if any(":" in part for part in parts):
        raise ArchiveSafetyError(f"Windows stream/drive syntax in {label}: {name!r}")
    return parts


def _safe_member_target(root: Path, member_name: str) -> Path:
    parts = _validated_relative_parts(member_name)
    resolved_root = root.resolve()
    target = (resolved_root.joinpath(*parts)).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        raise ArchiveSafetyError(f"Archive member escapes destination: {member_name!r}") from None
    return target


def safe_extract_zip(archive: str | Path, destination: str | Path) -> Path:
    """Extract a ZIP after rejecting traversal, links, and special files."""

    archive_path = Path(archive)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    with zipfile.ZipFile(archive_path) as bundle:
        for info in bundle.infolist():
            target = _safe_member_target(destination_path, info.filename)
            normalized_target = os.path.normcase(str(target))
            if normalized_target in seen:
                raise ArchiveSafetyError(f"Duplicate archive member: {info.filename!r}")
            seen.add(normalized_target)

            unix_mode = info.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)
            if stat.S_ISLNK(unix_mode):
                raise ArchiveSafetyError(f"Symbolic link in ZIP: {info.filename!r}")
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ArchiveSafetyError(f"Special file in ZIP: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise ArchiveSafetyError(f"Encrypted ZIP member is unsupported: {info.filename!r}")

            if info.is_dir() or stat.S_ISDIR(unix_mode):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return destination_path


def find_7z(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Locate 7-Zip without modifying PATH or the registry."""

    env = os.environ if environ is None else environ
    candidates: list[Path] = []
    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        candidates.extend(
            [explicit_path, explicit_path / "7z.exe"]
            if explicit_path.suffix.lower() != ".exe"
            else [explicit_path]
        )
    else:
        for variable in ("SEVEN_ZIP_PATH", "SEVENZIP"):
            if value := _env_value(env, variable):
                env_path = Path(value).expanduser()
                candidates.extend(
                    [env_path, env_path / "7z.exe"]
                    if env_path.suffix.lower() != ".exe"
                    else [env_path]
                )
        for command in ("7z", "7zz", "7za"):
            if found := shutil.which(command, path=_env_value(env, "PATH")):
                candidates.append(Path(found))
        for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            if value := _env_value(env, variable):
                candidates.append(Path(value) / "7-Zip" / "7z.exe")

    for candidate in _unique_paths(candidates):
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in _unique_paths(candidates)) or "PATH"
    raise FileNotFoundError(
        "7-Zip is required for .7z archives. Install 7-Zip or set "
        f"SEVEN_ZIP_PATH. Searched: {searched}"
    )


def _validate_7z_listing(output: str) -> None:
    in_entries = False
    member_count = 0
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("----------"):
            in_entries = True
            continue
        if not in_entries:
            continue
        if line.startswith("Path = "):
            _validated_relative_parts(line.removeprefix("Path = "))
            member_count += 1
        elif line.startswith(("Symbolic Link = ", "Hard Link = ")):
            raise ArchiveSafetyError(f"Links in 7z archives are unsupported: {line}")
    if member_count == 0:
        raise ArchiveSafetyError("7-Zip did not report any archive members")


def extract_7z(
    archive: str | Path,
    destination: str | Path,
    *,
    seven_zip: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Validate a 7z listing, extract it, and audit the extracted tree."""

    archive_path = Path(archive).resolve()
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    executable = find_7z(seven_zip, environ=environ)
    listing = subprocess.run(
        [str(executable), "l", "-slt", str(archive_path)],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if listing.returncode != 0:
        raise RuntimeError(
            f"7-Zip could not list {archive_path} (exit {listing.returncode}): "
            f"{listing.stderr.strip()}"
        )
    _validate_7z_listing(listing.stdout)

    extraction = subprocess.run(
        [
            str(executable),
            "x",
            "-y",
            "-bd",
            "-bb0",
            f"-o{destination_path.resolve()}",
            str(archive_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if extraction.returncode != 0:
        raise RuntimeError(
            f"7-Zip could not extract {archive_path} (exit {extraction.returncode}): "
            f"{extraction.stderr.strip()}"
        )

    resolved_destination = destination_path.resolve()
    for extracted in destination_path.rglob("*"):
        if extracted.is_symlink():
            raise ArchiveSafetyError(f"Extracted link is unsupported: {extracted}")
        try:
            extracted.resolve().relative_to(resolved_destination)
        except ValueError:
            raise ArchiveSafetyError(f"Extracted path escapes destination: {extracted}") from None
    return destination_path


def extract_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    seven_zip: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Extract a supported archive using the safe extractor."""

    archive_path = Path(archive)
    if archive_path.suffix.lower() == ".zip":
        return safe_extract_zip(archive_path, destination)
    if archive_path.suffix.lower() == ".7z":
        return extract_7z(
            archive_path,
            destination,
            seven_zip=seven_zip,
            environ=environ,
        )
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def download_and_verify(
    url: str,
    destination: str | Path,
    expected_sha256: str,
    *,
    timeout: float = 60.0,
) -> Path:
    """Download to a temporary file, verify it, then atomically publish it.

    A valid existing cache entry is reused.  A mismatching existing entry is
    deliberately left untouched and reported to the caller.
    """

    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS download URL: {url}")
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        verify_sha256(destination_path, expected_sha256)
        return destination_path

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".part",
        dir=destination_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "colmap-openmvs-pipeline-tool-installer/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary_path.open(
            "wb"
        ) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256.lower()}, got {actual}"
            )
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination_path


def install_asset(
    project_root: str | Path,
    tool_name: str,
    variant: str,
    *,
    manifest_path: str | Path | None = None,
    download_dir: str | Path | None = None,
    seven_zip: str | Path | None = None,
) -> Path:
    """Install one manifest asset into its versioned local directory.

    Existing complete installations are reused.  Existing incomplete
    directories are never overwritten or removed automatically.
    """

    root = Path(project_root).resolve()
    selected_manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else (root / DEFAULT_MANIFEST).resolve()
    )
    manifest = load_manifest(selected_manifest)
    try:
        tool = manifest["tools"][tool_name]
        asset = tool["variants"][variant]
    except KeyError as exc:
        raise KeyError(f"Unknown tool variant {tool_name}:{variant}") from exc

    tools_root = (root / "tools").resolve()
    tools_root.mkdir(parents=True, exist_ok=True)
    install_parts = _validated_relative_parts(asset["install_dir"], label="install directory")
    target = tools_root.joinpath(*install_parts)
    probe_parts = _validated_relative_parts(asset["probe"], label="probe path")
    target_probe = target.joinpath(*probe_parts)
    if target.exists():
        if target_probe.is_file():
            return target
        raise FileExistsError(
            f"Refusing to overwrite incomplete installation {target}; move it aside first"
        )

    cache_root = (
        Path(download_dir).resolve()
        if download_dir is not None
        else (tools_root / "_downloads").resolve()
    )
    archive_parts = _validated_relative_parts(asset["archive"], label="archive name")
    if len(archive_parts) != 1:
        raise ManifestError("Archive cache names must not contain directories")
    archive_path = cache_root / archive_parts[0]
    download_and_verify(asset["url"], archive_path, asset["sha256"])

    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=tools_root))
    payload = staging / "payload"
    try:
        if asset.get("kind", "archive") == "file":
            staged_probe = payload.joinpath(*probe_parts)
            staged_probe.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive_path, staged_probe)
        else:
            extract_archive(archive_path, payload, seven_zip=seven_zip)
        staged_probe = payload.joinpath(*probe_parts)
        if not staged_probe.is_file():
            raise RuntimeError(
                f"Archive {archive_path.name} does not contain expected file {asset['probe']}"
            )
        payload.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


class ToolResolver:
    """Resolve supported command-line tools using deterministic precedence."""

    def __init__(
        self,
        project_root: str | Path,
        manifest_path: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        epic_manifest_dirs: Sequence[str | Path] | None = None,
        registry_entries: Iterable[Mapping[str, Any]] | None = None,
        system_detection: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.project_root / DEFAULT_MANIFEST
        )
        self.manifest = load_manifest(self.manifest_path)
        self.environ = dict(os.environ if environ is None else environ)
        self._epic_manifest_dirs_override = (
            tuple(Path(path) for path in epic_manifest_dirs)
            if epic_manifest_dirs is not None
            else None
        )
        self._registry_entries_override = (
            tuple(dict(entry) for entry in registry_entries)
            if registry_entries is not None
            else None
        )
        self.system_detection = system_detection

    @property
    def tools_root(self) -> Path:
        return self.project_root / "tools"

    def resolve_colmap(
        self,
        explicit: str | Path | None = None,
        variant: str = "auto",
    ) -> ToolInfo:
        override = self._override(explicit, _COLMAP_ENV_VARS)
        if override is not None:
            raw_path, source = override
            executable = _colmap_executable(raw_path)
            if executable.is_file():
                resolved_variant = self._infer_variant("colmap", executable, variant)
                return ToolInfo(
                    name="colmap",
                    version=self._version_for_path("colmap", executable),
                    path=executable.resolve(),
                    variant=resolved_variant,
                    available=True,
                    details={
                        "source": source,
                        "edition": _edition_label(resolved_variant),
                    },
                )
            return self._unavailable(
                "colmap", variant, source, [_colmap_executable(raw_path)]
            )

        searched: list[Path] = []
        for candidate_variant, asset in self._manifest_variants("colmap", variant):
            executable = self.tools_root / asset["install_dir"] / asset["probe"]
            searched.append(executable)
            if executable.is_file():
                return ToolInfo(
                    name="colmap",
                    version=self.manifest["tools"]["colmap"]["version"],
                    path=executable.resolve(),
                    variant=candidate_variant,
                    available=True,
                    details={
                        "source": "manifest",
                        "edition": _edition_label(candidate_variant),
                        "install_dir": str(executable.parent.parent),
                    },
                )
        return self._unavailable("colmap", variant, "manifest", searched)

    def resolve_openmvs(
        self,
        explicit: str | Path | None = None,
        variant: str = "auto",
    ) -> ToolInfo:
        override = self._override(explicit, _OPENMVS_ENV_VARS)
        if override is not None:
            raw_path, source = override
            binary_dir = _openmvs_binary_dir(raw_path)
            resolved_variant = self._infer_variant("openmvs", binary_dir, variant)
            return self._openmvs_info(binary_dir, resolved_variant, source)

        searched: list[Path] = []
        for candidate_variant, asset in self._manifest_variants("openmvs", variant):
            probe = self.tools_root / asset["install_dir"] / asset["probe"]
            binary_dir = probe.parent
            searched.append(binary_dir)
            info = self._openmvs_info(
                binary_dir,
                candidate_variant,
                "manifest",
                version=self.manifest["tools"]["openmvs"]["version"],
            )
            if info.available:
                return info
        return self._unavailable("openmvs", variant, "manifest", searched)

    def resolve_realityscan(
        self,
        explicit: str | Path | None = None,
    ) -> ToolInfo:
        override = self._override(explicit, _REALITYSCAN_ENV_VARS)
        if override is not None:
            raw_path, source = override
            executable = _named_executable(raw_path, "RealityScan.exe")
            if executable.is_file():
                return ToolInfo(
                    "realityscan",
                    _version_from_path(executable),
                    executable.resolve(),
                    "desktop",
                    True,
                    {"source": source, "edition": "RealityScan"},
                )
            return self._unavailable("realityscan", "desktop", source, [executable])

        searched: list[Path] = []
        for installation in self._epic_installations():
            executable = installation["path"]
            searched.append(executable)
            if executable.is_file():
                details = dict(installation["details"])
                details.update({"source": "epic_manifest", "edition": "RealityScan"})
                return ToolInfo(
                    "realityscan",
                    installation.get("version"),
                    executable.resolve(),
                    "desktop",
                    True,
                    details,
                )

        for installation in self._registry_installations("realityscan"):
            executable = installation["path"]
            searched.append(executable)
            if executable.is_file():
                details = dict(installation["details"])
                details["source"] = "registry"
                return ToolInfo(
                    "realityscan",
                    installation.get("version"),
                    executable.resolve(),
                    "desktop",
                    True,
                    details,
                )

        if self.system_detection:
            for executable in self._realityscan_defaults():
                searched.append(executable)
                if executable.is_file():
                    return ToolInfo(
                        "realityscan",
                        _version_from_path(executable),
                        executable.resolve(),
                        "desktop",
                        True,
                        {"source": "default_path", "edition": "RealityScan"},
                    )
        return self._unavailable("realityscan", "desktop", "system", searched)

    def resolve_metashape(
        self,
        explicit: str | Path | None = None,
    ) -> ToolInfo:
        override = self._override(explicit, _METASHAPE_ENV_VARS)
        if override is not None:
            raw_path, source = override
            executable = _named_executable(raw_path, "metashape.exe")
            if executable.is_file():
                metadata = self._matching_registry_metadata(executable, "metashape")
                version = metadata.get("version") or _version_from_path(executable)
                edition = metadata.get("edition") or "unknown"
                return ToolInfo(
                    "metashape",
                    version,
                    executable.resolve(),
                    str(edition).lower(),
                    True,
                    {"source": source, "edition": edition},
                )
            return self._unavailable("metashape", None, source, [executable])

        searched: list[Path] = []
        registry_installations = self._registry_installations("metashape")
        for installation in registry_installations:
            executable = installation["path"]
            searched.append(executable)
            if executable.is_file():
                details = dict(installation["details"])
                details["source"] = "registry"
                edition = details.get("edition", "unknown")
                return ToolInfo(
                    "metashape",
                    installation.get("version"),
                    executable.resolve(),
                    str(edition).lower(),
                    True,
                    details,
                )

        if self.system_detection:
            registry_metadata = self._best_registry_metadata("metashape")
            for executable in self._metashape_defaults():
                searched.append(executable)
                if executable.is_file():
                    version = registry_metadata.get("version") or _version_from_path(executable)
                    edition = registry_metadata.get("edition") or "unknown"
                    return ToolInfo(
                        "metashape",
                        version,
                        executable.resolve(),
                        str(edition).lower(),
                        True,
                        {
                            "source": "default_path",
                            "edition": edition,
                            "registry_display_name": registry_metadata.get("display_name"),
                        },
                    )
        return self._unavailable("metashape", None, "system", searched)

    def collect_tool_metadata(self) -> dict[str, ToolInfo]:
        """Resolve all tools for status/doctor reporting."""

        return {
            "colmap": self.resolve_colmap(),
            "openmvs": self.resolve_openmvs(),
            "realityscan": self.resolve_realityscan(),
            "metashape": self.resolve_metashape(),
        }

    def _override(
        self,
        explicit: str | Path | None,
        environment_variables: Sequence[str],
    ) -> tuple[Path, str] | None:
        if explicit is not None:
            return Path(explicit).expanduser(), "explicit"
        for variable in environment_variables:
            if value := _env_value(self.environ, variable):
                return Path(value).expanduser(), f"env:{variable}"
        return None

    def _manifest_variants(
        self,
        tool_name: str,
        requested: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        tool = self.manifest["tools"].get(tool_name)
        if not isinstance(tool, dict):
            raise ManifestError(f"Tool {tool_name!r} is not present in the manifest")
        variants = tool["variants"]
        if requested != "auto":
            if requested not in variants:
                allowed = ", ".join(sorted(variants))
                raise ValueError(
                    f"Unknown {tool_name} variant {requested!r}; choose one of: {allowed}, auto"
                )
            return [(requested, variants[requested])]
        default = tool["default_variant"]
        order = [default, *(name for name in variants if name != default)]
        return [(name, variants[name]) for name in order]

    def _infer_variant(self, tool_name: str, path: Path, requested: str) -> str:
        if requested != "auto":
            return requested
        lowered = str(path).lower()
        if "nocuda" in lowered or re.search(r"(?:^|[-_\\/])cpu(?:[-_\\/]|$)", lowered):
            return "nocuda" if tool_name == "colmap" else "cpu"
        if "cuda" in lowered:
            return "cuda"
        for variant_name, asset in self._manifest_variants(tool_name, "auto"):
            expected = (self.tools_root / asset["install_dir"]).resolve()
            try:
                path.resolve().relative_to(expected)
                return variant_name
            except ValueError:
                pass
        return self.manifest["tools"][tool_name]["default_variant"]

    def _version_for_path(self, tool_name: str, path: Path) -> str | None:
        for _, asset in self._manifest_variants(tool_name, "auto"):
            expected = (self.tools_root / asset["install_dir"]).resolve()
            try:
                path.resolve().relative_to(expected)
                return self.manifest["tools"][tool_name]["version"]
            except ValueError:
                pass
        inferred = _version_from_path(path)
        if inferred is not None:
            return inferred
        if tool_name == "colmap":
            return _probe_colmap_version(path)
        return None

    def _openmvs_info(
        self,
        binary_dir: Path,
        variant: str,
        source: str,
        *,
        version: str | None = None,
    ) -> ToolInfo:
        commands = {name.removesuffix(".exe"): binary_dir / name for name in _OPENMVS_COMMANDS}
        missing = [str(path) for path in commands.values() if not path.is_file()]
        available = not missing
        return ToolInfo(
            "openmvs",
            version or self._version_for_path("openmvs", binary_dir),
            binary_dir.resolve() if available else None,
            variant,
            available,
            {
                "source": source,
                "edition": _edition_label(variant),
                "commands": {name: str(path) for name, path in commands.items()},
                "missing_executables": missing,
                "searched_path": str(binary_dir),
            },
        )

    def _epic_manifest_directories(self) -> tuple[Path, ...]:
        if self._epic_manifest_dirs_override is not None:
            return self._epic_manifest_dirs_override
        if not self.system_detection:
            return ()
        program_data = _env_value(self.environ, "ProgramData")
        if not program_data:
            return ()
        return (
            Path(program_data) / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests",
        )

    def _epic_installations(self) -> list[dict[str, Any]]:
        installations: list[dict[str, Any]] = []
        for directory in self._epic_manifest_directories():
            if not directory.is_dir():
                continue
            for item_path in sorted(directory.glob("*.item")):
                try:
                    item = json.loads(item_path.read_text(encoding="utf-8-sig"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                display_name = str(item.get("DisplayName", ""))
                launch_executable = str(item.get("LaunchExecutable", ""))
                if (
                    "realityscan" not in display_name.replace(" ", "").lower()
                    and Path(launch_executable).name.lower() != "realityscan.exe"
                ):
                    continue
                if item.get("bIsIncompleteInstall") is True:
                    continue
                location = item.get("InstallLocation")
                if not isinstance(location, str) or not location:
                    continue
                relative_launch = launch_executable or "RealityScan.exe"
                try:
                    launch_parts = _validated_relative_parts(
                        relative_launch, label="Epic launch executable"
                    )
                except ArchiveSafetyError:
                    continue
                executable = Path(location).joinpath(*launch_parts)
                version = item.get("AppVersionString")
                if not isinstance(version, str) or not version:
                    version = _version_from_text(display_name)
                installations.append(
                    {
                        "path": executable,
                        "version": version,
                        "details": {
                            "display_name": display_name,
                            "epic_app_name": item.get("AppName"),
                            "epic_catalog_item_id": item.get("CatalogItemId"),
                            "epic_manifest": str(item_path),
                            "install_location": location,
                        },
                    }
                )
        return installations

    def _registry_entries(self) -> list[dict[str, Any]]:
        if self._registry_entries_override is not None:
            return [dict(entry) for entry in self._registry_entries_override]
        if not self.system_detection or os.name != "nt":
            return []
        try:
            import winreg
        except ImportError:
            return []

        entries: list[dict[str, Any]] = []
        roots = (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER)
        views = [0]
        for flag_name in ("KEY_WOW64_64KEY", "KEY_WOW64_32KEY"):
            flag = getattr(winreg, flag_name, 0)
            if flag and flag not in views:
                views.append(flag)
        uninstall = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
        value_names = (
            "DisplayName",
            "DisplayVersion",
            "InstallLocation",
            "DisplayIcon",
            "Publisher",
        )
        for root in roots:
            for view in views:
                try:
                    parent = winreg.OpenKey(root, uninstall, 0, winreg.KEY_READ | view)
                except OSError:
                    continue
                with parent:
                    index = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(parent, index)
                        except OSError:
                            break
                        index += 1
                        try:
                            child = winreg.OpenKey(parent, subkey_name)
                        except OSError:
                            continue
                        entry: dict[str, Any] = {"RegistryKey": subkey_name}
                        with child:
                            for value_name in value_names:
                                try:
                                    entry[value_name] = winreg.QueryValueEx(child, value_name)[0]
                                except OSError:
                                    pass
                        if entry.get("DisplayName"):
                            entries.append(entry)
        return entries

    def _registry_installations(self, product: str) -> list[dict[str, Any]]:
        installations: list[dict[str, Any]] = []
        for entry in self._registry_entries():
            display_name = str(entry.get("DisplayName", ""))
            compact_name = display_name.replace(" ", "").lower()
            if product == "metashape":
                if "metashape" not in compact_name:
                    continue
                executable_name = "metashape.exe"
                edition = _metashape_edition(display_name)
            else:
                if "realityscan" not in compact_name:
                    continue
                executable_name = "RealityScan.exe"
                edition = "RealityScan"

            candidates: list[Path] = []
            if location := entry.get("InstallLocation"):
                candidates.append(Path(str(location)) / executable_name)
            if display_icon := entry.get("DisplayIcon"):
                candidates.append(_display_icon_path(str(display_icon)))
            for candidate in _unique_paths(candidates):
                installations.append(
                    {
                        "path": candidate,
                        "version": _optional_string(entry.get("DisplayVersion")),
                        "details": {
                            "display_name": display_name,
                            "edition": edition,
                            "publisher": entry.get("Publisher"),
                            "registry_key": entry.get("RegistryKey"),
                            "install_location": entry.get("InstallLocation"),
                        },
                    }
                )
        return installations

    def _best_registry_metadata(self, product: str) -> dict[str, Any]:
        for entry in self._registry_entries():
            display_name = str(entry.get("DisplayName", ""))
            compact_name = display_name.replace(" ", "").lower()
            if product == "metashape" and "metashape" in compact_name:
                return {
                    "version": _optional_string(entry.get("DisplayVersion")),
                    "edition": _metashape_edition(display_name),
                    "display_name": display_name,
                }
            if product == "realityscan" and "realityscan" in compact_name:
                return {
                    "version": _optional_string(entry.get("DisplayVersion")),
                    "edition": "RealityScan",
                    "display_name": display_name,
                }
        return {}

    def _matching_registry_metadata(self, executable: Path, product: str) -> dict[str, Any]:
        resolved = executable.resolve()
        for installation in self._registry_installations(product):
            try:
                if installation["path"].resolve() == resolved:
                    return {
                        "version": installation.get("version"),
                        "edition": installation["details"].get("edition"),
                    }
            except OSError:
                continue
        return self._best_registry_metadata(product)

    def _program_files_roots(self) -> list[Path]:
        roots: list[Path] = []
        for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            if value := _env_value(self.environ, variable):
                roots.append(Path(value))
        if not roots and os.name == "nt" and self.system_detection:
            roots.append(Path(r"C:\Program Files"))
        return _unique_paths(roots)

    def _realityscan_defaults(self) -> list[Path]:
        candidates: list[Path] = []
        for root in self._program_files_roots():
            epic_root = root / "Epic Games"
            candidates.append(epic_root / "RealityScan" / "RealityScan.exe")
            if epic_root.is_dir():
                candidates.extend(
                    directory / "RealityScan.exe"
                    for directory in sorted(epic_root.glob("RealityScan*"))
                )
        return _unique_paths(candidates)

    def _metashape_defaults(self) -> list[Path]:
        candidates: list[Path] = []
        for root in self._program_files_roots():
            agisoft = root / "Agisoft"
            for directory in ("Metashape", "Metashape Pro", "Metashape Standard"):
                candidates.append(agisoft / directory / "metashape.exe")
        return _unique_paths(candidates)

    @staticmethod
    def _unavailable(
        name: str,
        variant: str | None,
        source: str,
        searched: Sequence[Path],
    ) -> ToolInfo:
        return ToolInfo(
            name,
            None,
            None,
            None if variant == "auto" else variant,
            False,
            {
                "source": source,
                "searched_paths": [str(path) for path in _unique_paths(searched)],
            },
        )


def collect_tool_metadata(
    project_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, ToolInfo]:
    """Convenience wrapper for callers that do not need a resolver instance."""

    return ToolResolver(project_root, manifest_path).collect_tool_metadata()


def _env_value(environ: Mapping[str, str], name: str) -> str | None:
    for key, value in environ.items():
        if key.casefold() == name.casefold() and value.strip():
            return value.strip().strip('"')
    return None


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _colmap_executable(path: Path) -> Path:
    if path.suffix.lower() in (".exe", ".bat"):
        return path
    for relative in (Path("bin") / "colmap.exe", Path("colmap.exe"), Path("COLMAP.bat")):
        candidate = path / relative
        if candidate.is_file():
            return candidate
    return path / "bin" / "colmap.exe"


def _openmvs_binary_dir(path: Path) -> Path:
    if path.suffix.lower() == ".exe":
        return path.parent
    candidates = (
        path,
        path / "bin",
        path / "vc17" / "x64" / "Release",
        path / "vc16" / "x64" / "Release",
    )
    for candidate in candidates:
        if (candidate / "DensifyPointCloud.exe").is_file():
            return candidate
    return path


def _named_executable(path: Path, executable_name: str) -> Path:
    if path.suffix.lower() == ".exe":
        return path
    return path / executable_name


def _version_from_text(text: str) -> str | None:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+){1,3})(?!\d)", text)
    return match.group(1) if match else None


def _version_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if version := _version_from_text(part):
            return version
    return None


def _probe_colmap_version(executable: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(executable), "version"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _version_from_text(f"{result.stdout}\n{result.stderr}")


def _edition_label(variant: str | None) -> str | None:
    if variant == "cuda":
        return "CUDA"
    if variant in ("cpu", "nocuda"):
        return "CPU"
    return variant


def _metashape_edition(display_name: str) -> str:
    lowered = display_name.lower()
    if "professional" in lowered or re.search(r"\bpro\b", lowered):
        return "Professional"
    if "standard" in lowered:
        return "Standard"
    return "Unknown"


def _display_icon_path(value: str) -> Path:
    stripped = value.strip().strip('"')
    stripped = re.sub(r"[\"']?\s*,\s*-?\d+\s*$", "", stripped).strip().strip('"')
    return Path(stripped)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "ArchiveSafetyError",
    "ManifestError",
    "ToolInfo",
    "ToolResolver",
    "collect_tool_metadata",
    "download_and_verify",
    "extract_7z",
    "extract_archive",
    "find_7z",
    "install_asset",
    "load_manifest",
    "safe_extract_zip",
    "sha256_file",
    "verify_sha256",
]
