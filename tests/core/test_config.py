from dataclasses import dataclass, field

import pytest

from epitopecraft.core.config import (
    PipelineConfig,
    UnknownParameterError,
)
from epitopecraft.core.pipeline import Pipeline
from epitopecraft.core.step import PortSpec, Step


@dataclass(frozen=True)
class ModelConfig:
    recycles: int = field(default=3, metadata={"description": "Model recycles."})
    diffusion_samples: int = 1


@dataclass(frozen=True)
class RefoldConfig:
    backend: str = "af2"
    model: ModelConfig = field(default_factory=ModelConfig)


class Refold(Step):
    """Minimal refold Step used to exercise configuration resolution."""

    config_type = RefoldConfig
    input_ports = {"structure": PortSpec(str)}
    output_ports = {"structure": PortSpec(str)}

    def execute(self, inputs, context):
        return {"structure": inputs["structure"]}


def test_same_step_type_has_independent_instance_configuration():
    config = PipelineConfig.from_mapping(
        {
            "profiles": {"thorough": {"model": {"recycles": 6}}},
            "steps": {
                "refold_initial": {"params": {"model": {"recycles": 2}}},
                "refold_after_mpnn": {
                    "profile": "thorough",
                    "params": {"backend": "boltz2"},
                },
            },
        }
    )
    pipeline = Pipeline("two_refolds", inputs={"structure": str}, config=config)
    first = pipeline.add(Refold("refold_initial"), structure=pipeline.inputs.structure)
    pipeline.add(Refold("refold_after_mpnn"), structure=first.structure)
    pipeline.output("structure", pipeline.nodes[-1].outputs.structure)

    pipeline.resolve_config()

    assert pipeline.steps["refold_initial"].config.model.recycles == 2
    assert pipeline.steps["refold_after_mpnn"].config.model.recycles == 6
    assert pipeline.steps["refold_after_mpnn"].config.backend == "boltz2"


def test_pipeline_rejects_unknown_step_and_parameter():
    unknown_step = PipelineConfig.from_mapping({"steps": {"typo": {}}})
    pipeline = Pipeline("invalid", inputs={"structure": str}, config=unknown_step)
    pipeline.add(Refold("refold"), structure=pipeline.inputs.structure)
    with pytest.raises(UnknownParameterError, match="steps.typo"):
        pipeline.resolve_config()

    unknown_parameter = PipelineConfig.from_mapping(
        {"steps": {"refold": {"params": {"unknown": 1}}}}
    )
    pipeline = Pipeline("invalid", inputs={"structure": str}, config=unknown_parameter)
    pipeline.add(Refold("refold"), structure=pipeline.inputs.structure)
    with pytest.raises(UnknownParameterError, match="steps.refold.params.unknown"):
        pipeline.resolve_config()


def test_parameter_inventory_reports_effective_value_and_source():
    config = PipelineConfig.from_mapping(
        {"steps": {"refold": {"params": {"model": {"recycles": 5}}}}}
    )
    pipeline = Pipeline("describe", inputs={"structure": str}, config=config)
    pipeline.add(Refold("refold"), structure=pipeline.inputs.structure)

    rows = pipeline.explain_config()
    by_path = {row.path: row for row in rows}

    assert by_path["steps.refold.model.recycles"].value == 5
    assert by_path["steps.refold.model.recycles"].source == "step"
    assert by_path["steps.refold.model.recycles"].description == "Model recycles."
    assert by_path["steps.refold.model.diffusion_samples"].source == "default"

    schema = pipeline.config_schema()
    assert schema["steps"]["refold"]["model.recycles"] == {
        "type": "int",
        "default": 3,
        "description": "Model recycles.",
    }
