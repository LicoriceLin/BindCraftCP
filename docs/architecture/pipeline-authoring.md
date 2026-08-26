# Pipeline authoring

Python defines data flow; YAML defines behavior parameters. A customized design
workflow should remain short enough to review as one unit.

```python
def build_standard(config=None):
    p = Pipeline(
        "standard",
        inputs={"target": StructureArtifact},
        config=config,
    )
    hallu = p.add(Hallucinate("hallu"), target=p.inputs.target)
    selected = p.add(Filter("after_hallu"), designs=hallu.designs)
    grafted = p.add(
        Graft("graft"),
        target=p.inputs.target,
        designs=selected.passed,
    )
    initial = p.use(ValidationFlow(), "initial", designs=grafted.designs)
    mpnn = p.add(ProteinMPNN("mpnn"), designs=initial.passed)
    final = p.use(ValidationFlow(), "post_mpnn", designs=mpnn.designs)
    p.output("designs", final.passed)
    return p
```

A compatible `BoltzGenDesign`, `RFDiffusionDesign`, or project-specific design
Step can replace `Hallucinate` without changing downstream validation. OpenMM
MD should produce a `TrajectoryArtifact`; MMPBSA should consume a structure or
trajectory and add namespaced metrics.

Automatic wiring is allowed only when exactly one compatible value exists.
Bind ports explicitly whenever two target/design values are in scope. Filters
return distinct `passed` and `rejected` handles, so rejected designs never
trigger later expensive Steps.

Useful inspection commands:

```bash
epitopecraft pipeline describe myproject.workflow:build --config run.yaml
epitopecraft pipeline describe myproject.workflow:build --schema
epitopecraft pipeline plan myproject.workflow:build --config run.yaml
```
