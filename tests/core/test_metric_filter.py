from epitopecraft.core.design import Design, DesignSet, ProteinCandidate
from epitopecraft.core.pipeline import Pipeline, PipelineRunner
from epitopecraft.operations.filter import MetricFilter


def test_metric_filter_returns_branches_that_prune_downstream_work():
    designs = DesignSet.from_designs(
        [
            Design(
                "pass",
                ProteinCandidate("AAA"),
                metrics={
                    "halu:pLDDT": 0.9,
                    "halu:i-pAE": 0.2,
                },
            ),
            Design(
                "fail",
                ProteinCandidate("AAA"),
                metrics={
                    "halu:pLDDT": 0.5,
                    "halu:i-pAE": 0.2,
                },
            ),
        ]
    )
    pipeline = Pipeline("filter", inputs={"designs": DesignSet})
    filtered = pipeline.add(
        MetricFilter(
            "after_hallu",
            params={"recipe": "after:hallucinate"},
        ),
        designs=pipeline.inputs.designs,
    )
    pipeline.output("passed", filtered.passed)
    pipeline.output("rejected", filtered.rejected)

    result = PipelineRunner(pipeline).run(designs=designs)

    assert result.passed.ids == ("pass",)
    assert result.rejected.ids == ("fail",)
    assert designs["fail"].metrics["after_hallu"]["rules"]["halu:pLDDT"] is False
