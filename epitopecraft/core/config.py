"""Strict, instance-scoped configuration for composable pipelines."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, Sequence, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigurationError(ValueError):
    """Base error for invalid pipeline configuration."""


class UnknownParameterError(ConfigurationError):
    """Raised when configuration contains an unconsumed key."""


class InvalidParameterError(ConfigurationError):
    """Raised when a parameter has an incompatible value or shape."""


@dataclass(frozen=True)
class NoConfig:
    """Configuration model for Steps without behavior parameters."""


@dataclass(frozen=True)
class ConfigValue:
    """One resolved leaf in a printable configuration inventory."""

    path: str
    type_name: str
    default: Any
    value: Any
    source: str
    description: str = ""


@dataclass(frozen=True)
class ResolvedConfig:
    """A typed Step config plus provenance for each leaf value."""

    value: Any
    sources: Mapping[str, str]


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _leaf_paths(value: Mapping[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            paths.update(_leaf_paths(item, path))
        else:
            paths.add(path)
    return paths


def _type_name(annotation: Any) -> str:
    if annotation is Any:
        return "Any"
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    args = ", ".join(_type_name(arg) for arg in get_args(annotation))
    return f"{getattr(origin, '__name__', str(origin))}[{args}]"


def _matches_type(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return any(_matches_type(value, arg) for arg in get_args(annotation))
    if origin is not None:
        if origin in (list, tuple, set, frozenset, dict):
            return isinstance(value, origin)
        return True
    if annotation is Path:
        return isinstance(value, (str, Path))
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(annotation, type):
        return isinstance(value, annotation)
    return True


def _default_for_field(field_def) -> Any:
    if field_def.default is not MISSING:
        return deepcopy(field_def.default)
    if field_def.default_factory is not MISSING:
        return field_def.default_factory()
    return MISSING


def _model_defaults(model_type: type, path: str = "") -> dict[str, Any]:
    if not is_dataclass(model_type):
        raise TypeError(f"Config model {model_type!r} must be a dataclass")
    hints = get_type_hints(model_type)
    result: dict[str, Any] = {}
    for field_def in fields(model_type):
        annotation = hints.get(field_def.name, field_def.type)
        default = _default_for_field(field_def)
        if default is MISSING:
            continue
        if is_dataclass(annotation):
            result[field_def.name] = asdict(default)
        else:
            result[field_def.name] = deepcopy(default)
    return result


def _materialize(model_type: type, values: Mapping[str, Any], path: str) -> Any:
    if not isinstance(values, Mapping):
        raise InvalidParameterError(f"{path} must be a mapping")
    field_map = {item.name: item for item in fields(model_type)}
    unknown = set(values) - set(field_map)
    if unknown:
        key = sorted(unknown)[0]
        raise UnknownParameterError(f"Unknown parameter: {path}.{key}")

    hints = get_type_hints(model_type)
    kwargs: dict[str, Any] = {}
    for name, field_def in field_map.items():
        annotation = hints.get(name, field_def.type)
        field_path = f"{path}.{name}"
        if name in values:
            value = values[name]
        else:
            value = _default_for_field(field_def)
            if value is MISSING:
                raise InvalidParameterError(f"Missing required parameter: {field_path}")
        if is_dataclass(annotation):
            if is_dataclass(value):
                kwargs[name] = value
            else:
                kwargs[name] = _materialize(annotation, value, field_path)
        else:
            origin = get_origin(annotation)
            if origin is tuple and isinstance(value, list):
                value = tuple(value)
            elif origin is list and isinstance(value, tuple):
                value = list(value)
            if annotation is Path and isinstance(value, str):
                value = Path(value)
            if not _matches_type(value, annotation):
                raise InvalidParameterError(
                    f"{field_path} expects {_type_name(annotation)}, got {type(value).__name__}"
                )
            kwargs[name] = value
    return model_type(**kwargs)


def _iter_model_leaves(
    model_type: type,
    value: Any,
    *,
    prefix: str,
    sources: Mapping[str, str],
):
    hints = get_type_hints(model_type)
    for field_def in fields(model_type):
        annotation = hints.get(field_def.name, field_def.type)
        field_value = getattr(value, field_def.name)
        relative = field_def.name
        full_path = f"{prefix}.{relative}"
        if is_dataclass(annotation):
            nested_sources = {
                key.removeprefix(f"{relative}."): source
                for key, source in sources.items()
                if key.startswith(f"{relative}.")
            }
            yield from _iter_model_leaves(
                annotation,
                field_value,
                prefix=full_path,
                sources=nested_sources,
            )
            continue
        default = _default_for_field(field_def)
        yield ConfigValue(
            path=full_path,
            type_name=_type_name(annotation),
            default=None if default is MISSING else default,
            value=field_value,
            source=sources.get(relative, "default"),
            description=str(field_def.metadata.get("description", "")),
        )


class PipelineConfig:
    """Resolve YAML parameters only for Steps present in one Pipeline."""

    def __init__(self, mapping: Mapping[str, Any] | None = None):
        data = deepcopy(dict(mapping or {}))
        unknown = set(data) - {"profiles", "steps"}
        if unknown:
            raise UnknownParameterError(f"Unknown top-level parameter: {sorted(unknown)[0]}")
        self.profiles = dict(data.get("profiles", {}))
        self.steps = dict(data.get("steps", {}))
        if not isinstance(self.profiles, dict) or not isinstance(self.steps, dict):
            raise InvalidParameterError("profiles and steps must be mappings")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PipelineConfig":
        """Build configuration from an already parsed mapping."""

        return cls(mapping)

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        """Load YAML configuration without importing modeling backends."""

        with Path(path).open() as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise InvalidParameterError("Pipeline config root must be a mapping")
        return cls(data)

    def assert_only_steps(self, step_ids: Sequence[str]) -> None:
        """Reject configuration blocks that no Pipeline Step consumes."""

        unknown = set(self.steps) - set(step_ids)
        if unknown:
            raise UnknownParameterError(f"Unknown configured Step: steps.{sorted(unknown)[0]}")

    def resolve(
        self,
        step_id: str,
        model_type: type,
        constructor_params: Mapping[str, Any] | None = None,
    ) -> ResolvedConfig:
        """Resolve defaults, profiles, YAML params, and constructor overrides."""

        block = self.steps.get(step_id, {})
        if block is None:
            block = {}
        if not isinstance(block, Mapping):
            raise InvalidParameterError(f"steps.{step_id} must be a mapping")
        unknown_block = set(block) - {"profile", "params"}
        if unknown_block:
            key = sorted(unknown_block)[0]
            raise UnknownParameterError(f"Unknown parameter: steps.{step_id}.{key}")

        merged = _model_defaults(model_type)
        sources = {path: "default" for path in _leaf_paths(merged)}
        profile_names = block.get("profile", [])
        if isinstance(profile_names, str):
            profile_names = [profile_names]
        if not isinstance(profile_names, Sequence):
            raise InvalidParameterError(f"steps.{step_id}.profile must be a string or list")
        for profile_name in profile_names:
            if profile_name not in self.profiles:
                raise UnknownParameterError(f"Unknown profile: profiles.{profile_name}")
            profile = self.profiles[profile_name]
            if not isinstance(profile, Mapping):
                raise InvalidParameterError(f"profiles.{profile_name} must be a mapping")
            merged = _deep_merge(merged, profile)
            sources.update({path: f"profile:{profile_name}" for path in _leaf_paths(profile)})

        params = block.get("params", {})
        if not isinstance(params, Mapping):
            raise InvalidParameterError(f"steps.{step_id}.params must be a mapping")
        merged = _deep_merge(merged, params)
        sources.update({path: "step" for path in _leaf_paths(params)})
        if constructor_params:
            merged = _deep_merge(merged, constructor_params)
            sources.update(
                {path: "constructor" for path in _leaf_paths(constructor_params)}
            )
        value = _materialize(model_type, merged, f"steps.{step_id}.params")
        return ResolvedConfig(value=value, sources=sources)

    def explain(
        self,
        step_id: str,
        model_type: type,
        resolved: ResolvedConfig,
    ) -> list[ConfigValue]:
        """Return printable leaf values for one resolved Step."""

        return list(
            _iter_model_leaves(
                model_type,
                resolved.value,
                prefix=f"steps.{step_id}",
                sources=resolved.sources,
            )
        )
