from dataclasses import dataclass

from epitopecraft.core.config import NoConfig, PipelineConfig
from epitopecraft.core.design import (
    Design,
    DesignSet,
    ProteinCandidate,
    SmallMoleculeCandidate,
)
from epitopecraft.core.pipeline import Flow, Pipeline, PipelineRunner
from epitopecraft.core.step import PortSpec, Step


class MakeDesigns(Step):
    config_type = NoConfig
    input_ports = {"target": PortSpec(str)}
    output_ports = {"designs": PortSpec(DesignSet)}

    def execute(self, inputs, context):
        designs = [
            Design(
                id=f"d{index}",
                candidate=ProteinCandidate(sequence=sequence),
                metrics={"quality": quality},
            )
            for index, (sequence, quality) in enumerate(
                [("ACA", 0.9), ("GGG", 0.2), ("CCG", 0.8), ("AAA", 0.1)]
            )
        ]
        return {"designs": DesignSet.from_designs(designs)}


@dataclass(frozen=True)
class FilterConfig:
    metric: str = "quality"
    minimum: float = 0.5


class FilterDesigns(Step):
    config_type = FilterConfig
    input_ports = {"designs": PortSpec(DesignSet)}
    output_ports = {
        "passed": PortSpec(DesignSet),
        "rejected": PortSpec(DesignSet),
    }

    def execute(self, inputs, context):
        return inputs["designs"].partition(
            lambda design: design.metrics[self.config.metric] >= self.config.minimum
        )


class TraceTransform(Step):
    config_type = NoConfig
    input_ports = {"designs": PortSpec(DesignSet)}
    output_ports = {"designs": PortSpec(DesignSet)}

    def execute(self, inputs, context):
        seen = context.state.setdefault("seen", {})
        seen[self.id] = tuple(inputs["designs"].ids)
        for design in inputs["designs"]:
            design.metrics[self.id] = True
        return {"designs": inputs["designs"]}


class RemoveCysteine(Step):
    config_type = NoConfig
    input_ports = {"designs": PortSpec(DesignSet)}
    output_ports = {"designs": PortSpec(DesignSet)}

    def execute(self, inputs, context):
        children = []
        for parent in inputs["designs"]:
            candidate = ProteinCandidate(sequence=parent.candidate.sequence.replace("C", "S"))
            children.append(parent.child(f"{parent.id}-mpnn", candidate=candidate))
        return {"designs": DesignSet.from_designs(children)}


class ValidationFlow(Flow):
    def build(self, pipeline, *, flow_id, designs):
        refold = pipeline.add(TraceTransform(f"{flow_id}.refold"), designs=designs)
        score = pipeline.add(TraceTransform(f"{flow_id}.score"), designs=refold.designs)
        return pipeline.add(FilterDesigns(f"{flow_id}.filter"), designs=score.designs)


def test_standard_workflow_prunes_before_expensive_stages(tmp_path):
    config = PipelineConfig.from_mapping(
        {
            "steps": {
                "after_hallu": {"params": {"minimum": 0.5}},
                "initial.filter": {"params": {"minimum": 0.5}},
                "post_mpnn.filter": {"params": {"minimum": 0.5}},
            }
        }
    )
    pipeline = Pipeline("standard", inputs={"target": str}, config=config)
    hallu = pipeline.add(MakeDesigns("hallu"), target=pipeline.inputs.target)
    selected = pipeline.add(FilterDesigns("after_hallu"), designs=hallu.designs)
    graft = pipeline.add(TraceTransform("graft"), designs=selected.passed)
    initial = pipeline.use(ValidationFlow(), "initial", designs=graft.designs)
    mpnn = pipeline.add(RemoveCysteine("mpnn"), designs=initial.passed)
    final = pipeline.use(ValidationFlow(), "post_mpnn", designs=mpnn.designs)
    relaxed = pipeline.add(TraceTransform("relax"), designs=final.passed)
    pipeline.output("designs", relaxed.designs)

    runner = PipelineRunner(pipeline, run_dir=tmp_path)
    result = runner.run(target="target.pdb")

    assert result.designs.ids == ("d0-mpnn", "d2-mpnn")
    assert runner.context.state["seen"]["graft"] == ("d0", "d2")
    assert runner.context.state["seen"]["relax"] == ("d0-mpnn", "d2-mpnn")


def test_remove_cysteine_is_a_recipe_shaped_redesign_workflow():
    input_design = Design(
        id="complex",
        candidate=ProteinCandidate(sequence="ACCG"),
        metrics={"quality": 1.0},
    )
    pipeline = Pipeline("remove_c", inputs={"designs": DesignSet})
    redesigned = pipeline.add(RemoveCysteine("mpnn"), designs=pipeline.inputs.designs)
    validated = pipeline.use(ValidationFlow(), "validation", designs=redesigned.designs)
    pipeline.output("designs", validated.passed)

    result = PipelineRunner(pipeline).run(
        designs=DesignSet.from_designs([input_design])
    )

    output = result.designs["complex-mpnn"]
    assert output.candidate.sequence == "ASSG"
    assert output.parent_id == "complex"


class DiscoverSites(Step):
    config_type = NoConfig
    input_ports = {"designs": PortSpec(DesignSet)}
    output_ports = {"patches": PortSpec(tuple)}

    def execute(self, inputs, context):
        return {"patches": (("target:X", 10, 11), ("target:X", 20, 21))}


def test_target_analysis_accepts_any_compatible_design_generator():
    pipeline = Pipeline("site_discovery", inputs={"target": str})
    samples = pipeline.add(MakeDesigns("no_hotspot_designs"), target=pipeline.inputs.target)
    sites = pipeline.add(DiscoverSites("binding_sites"), designs=samples.designs)
    pipeline.output("patches", sites.patches)

    result = PipelineRunner(pipeline).run(target="target.pdb")

    assert result.patches[0] == ("target:X", 10, 11)


class ScreenCandidates(Step):
    config_type = NoConfig
    input_ports = {"candidates": PortSpec(tuple), "site": PortSpec(tuple)}
    output_ports = {"designs": PortSpec(DesignSet)}

    def execute(self, inputs, context):
        designs = []
        for index, candidate in enumerate(inputs["candidates"]):
            score = len(candidate.sequence) if isinstance(candidate, ProteinCandidate) else len(candidate.smiles)
            designs.append(Design(f"candidate-{index}", candidate, metrics={"score": score}))
        return {"designs": DesignSet.from_designs(designs)}


def test_epitope_screening_supports_protein_and_small_molecule_candidates():
    pipeline = Pipeline(
        "screen",
        inputs={"candidates": tuple, "site": tuple},
    )
    screened = pipeline.add(
        ScreenCandidates("screen"),
        candidates=pipeline.inputs.candidates,
        site=pipeline.inputs.site,
    )
    pipeline.output("designs", screened.designs)

    result = PipelineRunner(pipeline).run(
        candidates=(
            ProteinCandidate(sequence="ACDE"),
            SmallMoleculeCandidate(smiles="CCO"),
        ),
        site=("target:X", 11, 12),
    )

    assert isinstance(result.designs["candidate-0"].candidate, ProteinCandidate)
    assert isinstance(result.designs["candidate-1"].candidate, SmallMoleculeCandidate)
