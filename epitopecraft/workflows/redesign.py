"""Recipe-driven protein redesign workflows."""

from __future__ import annotations

from dataclasses import dataclass

from ..backends.colabdesign.mpnn import ProteinMPNN, ProteinMPNNConfig
from ..core.config import PipelineConfig
from ..core.design import DesignSet
from ..core.pipeline import Flow, Pipeline


@dataclass(frozen=True)
class RemoveCysteineConfig(ProteinMPNNConfig):
    """ProteinMPNN defaults for redesigning existing cysteine positions."""

    recipe_path: str = "epitopecraft/pipelines/config/mpnn-rmC-recipe.json"
    include_input: bool = False


class RemoveCysteineMPNN(ProteinMPNN):
    """ProteinMPNN with remove-cysteine defaults, still overrideable from YAML."""

    config_type = RemoveCysteineConfig


def build_remove_cysteine_workflow(
    *,
    validation: Flow | None = None,
    config: PipelineConfig | None = None,
) -> Pipeline:
    """Build MPNN cysteine redesign with an optional reusable validation flow."""

    pipeline = Pipeline("remove_cysteine", inputs={"designs": DesignSet}, config=config)
    redesigned = pipeline.add(
        RemoveCysteineMPNN("remove_c"),
        designs=pipeline.inputs.designs,
    )
    output = redesigned
    if validation is not None:
        output = pipeline.use(validation, "validation", designs=redesigned.designs)
    result = getattr(output, "passed", getattr(output, "designs", None))
    if result is None:
        raise TypeError("Validation flow must expose passed or designs")
    pipeline.output("designs", result)
    return pipeline
