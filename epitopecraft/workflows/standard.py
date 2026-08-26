"""Standard design workflow assembled from replaceable backend components."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.artifacts import StructureArtifact
from ..core.config import PipelineConfig
from ..core.pipeline import Flow, Pipeline
from ..core.step import Step


@dataclass(frozen=True)
class StandardDesignComponents:
    """Replaceable Steps and validation flows used by the standard workflow."""

    design: Step
    after_design_filter: Step
    graft: Step
    initial_validation: Flow
    redesign: Step
    final_validation: Flow


def build_standard_design_workflow(
    components: StandardDesignComponents,
    *,
    config: PipelineConfig | None = None,
) -> Pipeline:
    """Build design → filter → graft → validate → redesign → validate.

    Validation flows receive both ``target`` and ``designs`` and must expose a
    ``passed`` output. They may internally compose refold, scoring, filtering,
    relaxation, MD, or other reusable stages.
    """

    pipeline = Pipeline(
        "standard_design",
        inputs={"target": StructureArtifact},
        config=config,
    )
    generated = pipeline.add(components.design, target=pipeline.inputs.target)
    selected = pipeline.add(
        components.after_design_filter,
        designs=generated.designs,
    )
    grafted = pipeline.add(
        components.graft,
        target=pipeline.inputs.target,
        designs=selected.passed,
    )
    initial = pipeline.use(
        components.initial_validation,
        "initial",
        target=pipeline.inputs.target,
        designs=grafted.designs,
    )
    redesigned = pipeline.add(components.redesign, designs=initial.passed)
    final = pipeline.use(
        components.final_validation,
        "post_redesign",
        target=pipeline.inputs.target,
        designs=redesigned.designs,
    )
    pipeline.output("designs", final.passed)
    return pipeline
