# Step contract

A Step instance is identified by `id`. The id is the configuration namespace,
output namespace, cache namespace, and provenance label. Two instances of the
same class therefore cannot overwrite each other.

```python
@dataclass(frozen=True)
class RefoldConfig:
    recycles: int = 3


class Refold(Step):
    """Refold protein candidates against one target.

    Consumes a target and designs. Produces the same design lineage with a new
    structure artifact and namespaced metrics. Runner owns persistence.
    """

    config_type = RefoldConfig
    input_ports = {
        "target": PortSpec(StructureArtifact),
        "designs": PortSpec(DesignSet),
    }
    output_ports = {"designs": PortSpec(DesignSet)}

    def execute(self, inputs, context):
        ...
```

Configuration resolution order is schema default, profiles in listed order,
YAML Step parameters, then constructor parameters. Unknown Step ids and unknown
fields are errors.

Public class and method docstrings should state purpose, semantic inputs and
outputs, side effects, and important errors. They should not repeat type hints
or narrate private implementation details.

Steps should not mutate shared configuration. Files must be written under
`context.step_dir()` unless the user explicitly supplies an external location.
Set `cacheable = False` only when outputs cannot be serialized or the Step has
non-repeatable external side effects.
