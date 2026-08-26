"""Step and port contracts independent of modeling backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, Mapping

from .config import NoConfig, PipelineConfig, ResolvedConfig


@dataclass(frozen=True)
class PortSpec:
    """The runtime type and purpose of a named Step port."""

    data_type: Any
    required: bool = True
    description: str = ""


class Step(ABC):
    """A configured transformation with typed named inputs and outputs.

    Subclasses declare ``config_type``, ``input_ports``, and ``output_ports``.
    Runner-owned persistence and scheduling stay outside ``execute``.
    """

    config_type: type = NoConfig
    input_ports: Mapping[str, PortSpec] = {}
    output_ports: Mapping[str, PortSpec] = {}
    cacheable: bool = True

    def __init__(self, step_id: str, *, params: Mapping[str, Any] | None = None):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", step_id):
            raise ValueError(
                "Step id must start with a letter and contain only letters, "
                "numbers, dots, underscores, or hyphens"
            )
        self.id = step_id
        self.params = dict(params or {})
        self._resolved_config: ResolvedConfig | None = None

    @property
    def config(self):
        """Return this instance's immutable resolved config."""

        if self._resolved_config is None:
            raise RuntimeError(f"Step {self.id!r} has not been configured")
        return self._resolved_config.value

    def resolve_config(self, pipeline_config: PipelineConfig) -> ResolvedConfig:
        """Resolve and retain the parameters scoped to this Step id."""

        self._resolved_config = pipeline_config.resolve(
            self.id,
            self.config_type,
            constructor_params=self.params,
        )
        return self._resolved_config

    @abstractmethod
    def execute(self, inputs: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        """Transform validated port values and return every declared output."""
