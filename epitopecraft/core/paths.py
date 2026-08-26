"""Resolve user paths and package-shipped resources without repository assumptions."""

from __future__ import annotations

from pathlib import Path


def resolve_package_path(value: str | Path) -> Path:
    """Resolve an existing path, then fall back to the installed package root."""

    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    package_root = Path(__file__).resolve().parents[1]
    parts = path.parts
    if parts and parts[0] == "epitopecraft":
        candidate = package_root.joinpath(*parts[1:])
    else:
        candidate = package_root / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(path)
