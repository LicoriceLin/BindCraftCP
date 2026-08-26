"""Known-site candidate screening workflow factory."""

from __future__ import annotations

from ..backends.boltz.screen import Boltz2Screen
from ..core.artifacts import SelectionArtifact, StructureArtifact
from ..core.config import PipelineConfig
from ..core.pipeline import Pipeline
from ..core.step import Step


def build_epitope_screening_workflow(
    *,
    screen: Step | None = None,
    config: PipelineConfig | None = None,
) -> Pipeline:
    """Build known canonical site + protein/SMILES candidates → scored designs."""

    pipeline = Pipeline(
        "epitope_screening",
        inputs={
            "target": StructureArtifact,
            "site": SelectionArtifact,
            "candidates": tuple,
        },
        config=config,
    )
    screened = pipeline.add(
        screen or Boltz2Screen("screen"),
        target=pipeline.inputs.target,
        site=pipeline.inputs.site,
        candidates=pipeline.inputs.candidates,
    )
    pipeline.output("designs", screened.designs)
    return pipeline
