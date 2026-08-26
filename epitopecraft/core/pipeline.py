"""Low-code pipeline graph construction and local execution."""

from __future__ import annotations

import json
import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping

from .config import ConfigValue, PipelineConfig
from .artifacts import StructureArtifact
from .design import DesignSet
from .step import PortSpec, Step


class PipelineDefinitionError(ValueError):
    """Base error for an invalid graph or port binding."""


class AmbiguousBindingError(PipelineDefinitionError):
    """Raised when automatic wiring finds zero or multiple candidates."""


class DuplicateStepIdError(PipelineDefinitionError):
    """Raised when two Step instances use the same id."""


@dataclass(frozen=True)
class Handle:
    """A typed reference to a Pipeline input or Step output."""

    producer: str
    port: str
    data_type: Any

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable producer-port lookup key."""
        return (self.producer, self.port)


class PortNamespace(SimpleNamespace):
    """Attribute access for a collection of named Handles or values."""


@dataclass
class PipelineNode:
    """One configured Step and its graph bindings."""

    step: Step
    bindings: dict[str, Handle]
    outputs: PortNamespace


@dataclass
class ExecutionContext:
    """Runner-owned paths and mutable run-scoped coordination state."""

    run_dir: Path | None = None
    state: dict[str, Any] = field(default_factory=dict)
    current_step_id: str | None = None

    def step_dir(self, step_id: str | None = None) -> Path | None:
        """Return and create the output directory for one Step instance."""

        if self.run_dir is None:
            return None
        target = self.run_dir / "steps" / (step_id or self.current_step_id or "unknown")
        target.mkdir(parents=True, exist_ok=True)
        return target


class Flow(ABC):
    """A reusable graph fragment that exposes ordinary Pipeline handles."""

    @abstractmethod
    def build(self, pipeline: "Pipeline", *, flow_id: str, **bindings) -> PortNamespace:
        """Install child Steps and return the fragment's public outputs."""


def _port_types_compatible(produced: Any, expected: Any) -> bool:
    if expected is Any or produced is Any:
        return True
    try:
        return issubclass(produced, expected)
    except TypeError:
        return produced == expected


def _runtime_type_matches(value: Any, expected: Any) -> bool:
    if expected is Any:
        return True
    try:
        return isinstance(value, expected)
    except TypeError:
        return True


class Pipeline:
    """A declarative DAG whose Step ids define configuration namespaces."""

    def __init__(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        config: PipelineConfig | None = None,
    ):
        self.name = name
        self.config = config or PipelineConfig()
        input_handles = {
            key: Handle("$input", key, data_type)
            for key, data_type in dict(inputs or {}).items()
        }
        self.inputs = PortNamespace(**input_handles)
        self.nodes: list[PipelineNode] = []
        self.steps: dict[str, Step] = {}
        self.outputs: dict[str, Handle] = {}

    def _available_handles(self) -> list[Handle]:
        handles = list(vars(self.inputs).values())
        for node in self.nodes:
            handles.extend(vars(node.outputs).values())
        return handles

    def _auto_bind(self, step: Step, port_name: str, port: PortSpec) -> Handle:
        candidates = [
            handle
            for handle in self._available_handles()
            if _port_types_compatible(handle.data_type, port.data_type)
        ]
        if len(candidates) != 1:
            rendered = ", ".join(f"{item.producer}.{item.port}" for item in candidates)
            raise AmbiguousBindingError(
                f"Cannot auto-bind {step.id}.{port_name}: expected one compatible value, "
                f"found {len(candidates)} ({rendered or 'none'})"
            )
        return candidates[0]

    def add(self, step: Step, **bindings: Handle) -> PortNamespace:
        """Add one Step and return typed handles for its outputs."""

        if step.id in self.steps:
            raise DuplicateStepIdError(f"Duplicate Step id: {step.id}")
        unknown = set(bindings) - set(step.input_ports)
        if unknown:
            raise PipelineDefinitionError(
                f"Unknown input port for {step.id}: {sorted(unknown)[0]}"
            )
        resolved: dict[str, Handle] = {}
        for port_name, port in step.input_ports.items():
            handle = bindings.get(port_name)
            if handle is None:
                if not port.required:
                    continue
                handle = self._auto_bind(step, port_name, port)
            if not isinstance(handle, Handle):
                raise PipelineDefinitionError(
                    f"{step.id}.{port_name} must bind to a Pipeline handle"
                )
            if not _port_types_compatible(handle.data_type, port.data_type):
                raise PipelineDefinitionError(
                    f"{step.id}.{port_name} expects {port.data_type}, got {handle.data_type}"
                )
            resolved[port_name] = handle

        output_handles = {
            name: Handle(step.id, name, port.data_type)
            for name, port in step.output_ports.items()
        }
        namespace = PortNamespace(**output_handles)
        self.nodes.append(PipelineNode(step=step, bindings=resolved, outputs=namespace))
        self.steps[step.id] = step
        return namespace

    def use(self, flow: Flow, flow_id: str, **bindings: Handle) -> PortNamespace:
        """Install a reusable subflow under a stable id namespace."""

        return flow.build(self, flow_id=flow_id, **bindings)

    def output(self, name: str, handle: Handle) -> None:
        """Expose one internal handle as a Pipeline result."""

        if not isinstance(handle, Handle):
            raise PipelineDefinitionError("Pipeline output must be a handle")
        self.outputs[name] = handle

    def resolve_config(self) -> None:
        """Resolve every Step config and reject all unconsumed blocks."""

        self.config.assert_only_steps(tuple(self.steps))
        for step in self.steps.values():
            step.resolve_config(self.config)

    def explain_config(self) -> list[ConfigValue]:
        """Return the effective, source-annotated parameter inventory."""

        self.resolve_config()
        rows: list[ConfigValue] = []
        for step in self.steps.values():
            rows.extend(
                self.config.explain(step.id, step.config_type, step._resolved_config)
            )
        return rows

    def config_schema(self) -> dict[str, Any]:
        """Return the accepted parameter schema for instantiated Steps only."""

        schema: dict[str, Any] = {"pipeline": self.name, "steps": {}}
        for step in self.steps.values():
            resolved = self.config.resolve(
                step.id,
                step.config_type,
                constructor_params=step.params,
            )
            rows = self.config.explain(step.id, step.config_type, resolved)
            prefix = f"steps.{step.id}."
            schema["steps"][step.id] = {
                row.path.removeprefix(prefix): {
                    "type": row.type_name,
                    "default": row.default,
                    "description": row.description,
                }
                for row in rows
            }
        return schema

    def describe(self) -> str:
        """Render graph edges and effective parameters for debugging."""

        lines = [f"Pipeline: {self.name}", "Steps:"]
        for node in self.nodes:
            edges = ", ".join(
                f"{name} <- {handle.producer}.{handle.port}"
                for name, handle in node.bindings.items()
            )
            lines.append(f"  {node.step.id} ({node.step.__class__.__name__}): {edges}")
        lines.append("Parameters:")
        for row in self.explain_config():
            lines.append(f"  {row.path} = {row.value!r} [{row.source}]")
        return "\n".join(lines)


class PipelineRunner:
    """Execute a Pipeline, own run state, and persist a compact manifest."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        run_dir: str | Path | None = None,
        resume: bool = True,
        overwrite_steps: tuple[str, ...] = (),
    ):
        self.pipeline = pipeline
        self.context = ExecutionContext(Path(run_dir) if run_dir is not None else None)
        self.resume = resume
        self.overwrite_steps = frozenset(overwrite_steps)
        self.executions: list[dict[str, Any]] = []

    def plan(self) -> list[dict[str, Any]]:
        """Return the ordered execution plan without running a backend."""

        return [
            {
                "step": node.step.id,
                "type": node.step.__class__.__name__,
                "inputs": {
                    name: f"{handle.producer}.{handle.port}"
                    for name, handle in node.bindings.items()
                },
                "outputs": list(vars(node.outputs)),
            }
            for node in self.pipeline.nodes
        ]

    def run(self, **inputs: Any) -> PortNamespace:
        """Validate inputs, execute the DAG, and return declared outputs."""

        expected_inputs = vars(self.pipeline.inputs)
        unknown = set(inputs) - set(expected_inputs)
        missing = set(expected_inputs) - set(inputs)
        if unknown or missing:
            raise PipelineDefinitionError(
                f"Pipeline inputs mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        values: dict[tuple[str, str], Any] = {}
        for name, handle in expected_inputs.items():
            value = inputs[name]
            if not _runtime_type_matches(value, handle.data_type):
                raise TypeError(
                    f"Pipeline input {name} expects {handle.data_type}, got {type(value)}"
                )
            values[handle.key] = value

        self.pipeline.resolve_config()
        self.executions = []
        if self.context.run_dir is not None:
            self.context.run_dir.mkdir(parents=True, exist_ok=True)
        for node in self.pipeline.nodes:
            self.context.current_step_id = node.step.id
            node_inputs = {name: values[handle.key] for name, handle in node.bindings.items()}
            signature = self._node_signature(node, node_inputs)
            cache_path = self._cache_path(node.step.id, signature)
            cached = (
                self.resume
                and node.step.cacheable
                and node.step.id not in self.overwrite_steps
                and cache_path is not None
                and cache_path.exists()
            )
            if cached:
                with cache_path.open("rb") as handle:
                    payload = pickle.load(handle)
                if payload.get("signature") != signature:
                    raise RuntimeError(f"Invalid cache signature in {cache_path}")
                outputs = payload["outputs"]
                duration = 0.0
            else:
                started = perf_counter()
                outputs = node.step.execute(node_inputs, self.context)
                duration = perf_counter() - started
            if not isinstance(outputs, Mapping):
                raise TypeError(f"Step {node.step.id} must return a mapping")
            expected_outputs = node.step.output_ports
            if set(outputs) != set(expected_outputs):
                raise PipelineDefinitionError(
                    f"Step {node.step.id} returned {sorted(outputs)}; "
                    f"expected {sorted(expected_outputs)}"
                )
            for name, value in outputs.items():
                port = expected_outputs[name]
                if not _runtime_type_matches(value, port.data_type):
                    raise TypeError(
                        f"Step output {node.step.id}.{name} expects {port.data_type}, "
                        f"got {type(value)}"
                    )
                values[(node.step.id, name)] = value
            if not cached and node.step.cacheable and cache_path is not None:
                self._write_cache(cache_path, signature, dict(outputs))
            self.executions.append(
                {
                    "step": node.step.id,
                    "type": node.step.__class__.__name__,
                    "seconds": duration,
                    "cached": cached,
                    "signature": signature,
                    "output_types": {
                        key: type(value).__name__ for key, value in outputs.items()
                    },
                }
            )

        result = PortNamespace(
            **{name: values[handle.key] for name, handle in self.pipeline.outputs.items()}
        )
        self._write_manifest()
        return result

    def _node_signature(self, node: PipelineNode, inputs: Mapping[str, Any]) -> str:
        """Hash Step implementation identity, resolved config, and input values."""

        config_value = node.step.config
        payload = {
            "schema": 1,
            "step_class": (
                f"{node.step.__class__.__module__}.{node.step.__class__.__qualname__}"
            ),
            "step_id": node.step.id,
            "config": _fingerprint_value(config_value),
            "inputs": _fingerprint_value(dict(inputs)),
        }
        try:
            serialized = pickle.dumps(payload, protocol=5)
        except (pickle.PickleError, TypeError, AttributeError) as error:
            raise RuntimeError(
                f"Cannot create cache signature for Step {node.step.id}; "
                "set cacheable=False for non-serializable inputs"
            ) from error
        return hashlib.sha256(serialized).hexdigest()

    def _cache_path(self, step_id: str, signature: str) -> Path | None:
        if self.context.run_dir is None:
            return None
        safe_id = step_id.replace("/", "_")
        return self.context.run_dir / "cache" / safe_id / f"{signature}.pkl"

    @staticmethod
    def _write_cache(path: Path, signature: str, outputs: Mapping[str, Any]) -> None:
        """Atomically persist trusted local run outputs for resume."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(
                {"schema": 1, "signature": signature, "outputs": dict(outputs)},
                handle,
                protocol=5,
            )
        temporary.replace(path)

    def _write_manifest(self) -> None:
        if self.context.run_dir is None:
            return
        manifest = {
            "pipeline": self.pipeline.name,
            "plan": self.plan(),
            "executions": self.executions,
            "config": [row.__dict__ for row in self.pipeline.explain_config()],
        }
        with (self.context.run_dir / "run_manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2, default=str)


def _fingerprint_value(value: Any) -> Any:
    """Normalize scientific inputs so cache signatures follow their content."""

    if isinstance(value, StructureArtifact):
        return {
            "type": "StructureArtifact",
            "id": value.id,
            "format": value.structure_format,
            "content_sha256": hashlib.sha256(value.read_text().encode()).hexdigest(),
            "residue_map": [
                (
                    entry.local_id.chain_id,
                    entry.local_id.sequence_id,
                    entry.local_id.insertion_code,
                    entry.canonical_id.entity_id if entry.canonical_id else None,
                    entry.canonical_id.position if entry.canonical_id else None,
                    entry.status,
                )
                for entry in value.residue_map.entries
            ],
            "reference": value.reference.to_dict() if value.reference else None,
            "provenance": _fingerprint_value(dict(value.provenance)),
        }
    if isinstance(value, DesignSet):
        return [
            {
                "id": design.id,
                "candidate": _fingerprint_value(design.candidate),
                "artifacts": _fingerprint_value(design.artifacts),
                "metrics": _fingerprint_value(design.metrics),
                "tracks": _fingerprint_value(design.tracks),
                "parent_id": design.parent_id,
            }
            for design in value
        ]
    if isinstance(value, Path):
        if value.exists() and value.is_file():
            stat = value.stat()
            return {
                "path": str(value.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        return {"path": str(value)}
    if is_dataclass(value):
        return _fingerprint_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_fingerprint_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)
