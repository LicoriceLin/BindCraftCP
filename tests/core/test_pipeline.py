from dataclasses import dataclass

import pytest

from epitopecraft.core.config import NoConfig, PipelineConfig
from epitopecraft.core.pipeline import AmbiguousBindingError, Flow, Pipeline, PipelineRunner
from epitopecraft.core.step import PortSpec, Step


@dataclass(frozen=True)
class AddConfig:
    amount: int = 1


class Add(Step):
    config_type = AddConfig
    input_ports = {"value": PortSpec(int)}
    output_ports = {"value": PortSpec(int)}

    def execute(self, inputs, context):
        return {"value": inputs["value"] + self.config.amount}


class AsText(Step):
    config_type = NoConfig
    input_ports = {"value": PortSpec(int)}
    output_ports = {"text": PortSpec(str)}

    def execute(self, inputs, context):
        return {"text": str(inputs["value"])}


class AddTwice(Flow):
    """Reusable two-Step flow with one public output."""

    def build(self, pipeline, *, flow_id, value):
        first = pipeline.add(Add(f"{flow_id}.first"), value=value)
        second = pipeline.add(Add(f"{flow_id}.second"), value=first.value)
        return second


def test_runner_executes_a_readable_composite_flow(tmp_path):
    config = PipelineConfig.from_mapping(
        {
            "steps": {
                "increment.first": {"params": {"amount": 2}},
                "increment.second": {"params": {"amount": 3}},
            }
        }
    )
    pipeline = Pipeline("math", inputs={"value": int}, config=config)
    added = pipeline.use(AddTwice(), "increment", value=pipeline.inputs.value)
    text = pipeline.add(AsText("format"), value=added.value)
    pipeline.output("text", text.text)

    result = PipelineRunner(pipeline, run_dir=tmp_path).run(value=4)

    assert result.text == "9"
    assert (tmp_path / "run_manifest.json").exists()


def test_auto_wiring_requires_exactly_one_compatible_output():
    pipeline = Pipeline("ambiguous", inputs={"left": int, "right": int})
    with pytest.raises(AmbiguousBindingError, match="value"):
        pipeline.add(Add("add"))


def test_explicit_binding_resolves_ambiguity():
    pipeline = Pipeline("explicit", inputs={"left": int, "right": int})
    added = pipeline.add(Add("add"), value=pipeline.inputs.left)
    pipeline.output("value", added.value)

    assert PipelineRunner(pipeline).run(left=2, right=100).value == 3


def test_runner_resumes_from_instance_scoped_cache(tmp_path):
    class CountingAdd(Add):
        calls = 0

        def execute(self, inputs, context):
            type(self).calls += 1
            return super().execute(inputs, context)

    pipeline = Pipeline("resume", inputs={"value": int})
    value = pipeline.add(CountingAdd("expensive"), value=pipeline.inputs.value)
    pipeline.output("value", value.value)

    assert PipelineRunner(pipeline, run_dir=tmp_path).run(value=3).value == 4
    assert PipelineRunner(pipeline, run_dir=tmp_path).run(value=3).value == 4

    assert CountingAdd.calls == 1
    assert (tmp_path / "cache" / "expensive").is_dir()


def test_config_change_invalidates_only_affected_step_cache(tmp_path):
    class First(Add):
        calls = 0

        def execute(self, inputs, context):
            type(self).calls += 1
            return super().execute(inputs, context)

    class Second(Add):
        calls = 0

        def execute(self, inputs, context):
            type(self).calls += 1
            return super().execute(inputs, context)

    def build(second_amount):
        config = PipelineConfig.from_mapping(
            {"steps": {"second": {"params": {"amount": second_amount}}}}
        )
        pipeline = Pipeline("selective_resume", inputs={"value": int}, config=config)
        first = pipeline.add(First("first"), value=pipeline.inputs.value)
        second = pipeline.add(Second("second"), value=first.value)
        pipeline.output("value", second.value)
        return pipeline

    assert PipelineRunner(build(2), run_dir=tmp_path).run(value=1).value == 4
    assert PipelineRunner(build(5), run_dir=tmp_path).run(value=1).value == 7

    assert First.calls == 1
    assert Second.calls == 2
