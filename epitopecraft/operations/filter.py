"""Recipe-driven filtering with explicit passed and rejected branches."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.design import DesignSet
from ..core.recipes import FilterRecipeBook, TrustedCallableRegistry
from ..core.step import PortSpec, Step


@dataclass(frozen=True)
class MetricFilterConfig:
    """Parameters consumed by one metric-filter instance."""

    recipe_path: str = "epitopecraft/pipelines/config/default_filter.yaml"
    recipe: str = "all"
    allow_python_hooks: bool = False


class MetricFilter(Step):
    """Evaluate a named metric recipe and prune downstream computation."""

    config_type = MetricFilterConfig
    input_ports = {"designs": PortSpec(DesignSet)}
    output_ports = {
        "passed": PortSpec(DesignSet),
        "rejected": PortSpec(DesignSet),
    }

    def __init__(
        self,
        step_id: str,
        *,
        params=None,
        callable_registry: TrustedCallableRegistry | None = None,
    ):
        super().__init__(step_id, params=params)
        self.callable_registry = callable_registry

    def execute(self, inputs, context):
        """Evaluate one recipe and partition designs into passed and rejected sets."""
        designs: DesignSet = inputs["designs"]
        recipes = FilterRecipeBook.from_file(
            self.config.recipe_path,
            callable_registry=self.callable_registry,
            allow_python_hooks=self.config.allow_python_hooks,
        )
        passed = []
        rejected = []
        for design in designs:
            decision = recipes.evaluate(self.config.recipe, design.metrics)
            design.metrics[self.id] = {
                "passed": decision.passed,
                "rules": dict(decision.rules),
            }
            (passed if decision.passed else rejected).append(design)
        return {
            "passed": DesignSet.from_designs(passed),
            "rejected": DesignSet.from_designs(rejected),
        }
