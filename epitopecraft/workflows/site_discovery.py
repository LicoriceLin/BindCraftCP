"""Binding-site discovery workflow with a pluggable design generator."""

from __future__ import annotations

from ..analysis.binding_sites import BindingSiteDiscovery
from ..core.artifacts import StructureArtifact
from ..core.config import PipelineConfig
from ..core.pipeline import Pipeline
from ..core.step import Step


def build_site_discovery_workflow(
    generator: Step,
    *,
    analysis: BindingSiteDiscovery | None = None,
    config: PipelineConfig | None = None,
) -> Pipeline:
    """Build target → no-hotspot designs → canonical binding-site patches.

    ``generator`` may be a ColabDesign hallucination Step, BoltzGen design Step,
    or any custom Step with ``target`` input and ``designs`` output.
    """

    pipeline = Pipeline(
        "binding_site_discovery",
        inputs={"target": StructureArtifact},
        config=config,
    )
    samples = pipeline.add(generator, target=pipeline.inputs.target)
    if not hasattr(samples, "designs"):
        raise TypeError(f"Generator {generator.id} must expose a designs output")
    sites = pipeline.add(
        analysis or BindingSiteDiscovery("binding_sites"),
        target=pipeline.inputs.target,
        designs=samples.designs,
    )
    pipeline.output("designs", samples.designs)
    pipeline.output("patches", sites.patches)
    pipeline.output("motifs", sites.motifs)
    pipeline.output("table", sites.table)
    return pipeline
