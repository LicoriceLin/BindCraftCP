# EpitopeCraft

EpitopeCraft is a composable framework for epitope-focused protein design,
refolding, sequence redesign, structure analysis, and protein/small-molecule
screening. It grew from BindCraft and is being reorganized around explicit Step
configuration, typed artifacts, canonical residue coordinates, and readable
pipeline definitions.

Preprint: [arXiv:2509.25479](https://arxiv.org/abs/2509.25479)

> The `refactor/core-v2` branch is an active API-breaking refactor. The legacy
> HalluDesign command remains available while individual backends move to the
> new contracts. See [MIGRATION.md](MIGRATION.md).

## Core model

- Every Step declares its own dataclass configuration and typed input/output
  ports. The Step instance id scopes config, metrics, cache, and provenance.
- `StructureArtifact` maps backend-local chain/residue ids to an immutable
  design reference. Complete target author numbering is preferred over local
  epitope numbering; binders receive a non-conflicting canonical entity.
- Python describes the data flow. YAML contains behavior parameters. Filters
  expose `passed` and `rejected` branches so expensive downstream work only
  sees selected designs.
- `PipelineRunner` owns run directories, manifests, and instance-scoped resume
  caches.

```python
pipeline = Pipeline(
    "screen",
    inputs={
        "target": StructureArtifact,
        "site": SelectionArtifact,
        "candidates": tuple,
    },
    config=PipelineConfig.from_file("screen.yaml"),
)
screened = pipeline.add(
    Boltz2Screen("screen"),
    target=pipeline.inputs.target,
    site=pipeline.inputs.site,
    candidates=pipeline.inputs.candidates,
)
pipeline.output("designs", screened.designs)
```

The same downstream contracts can be used by ColabDesign, BoltzGen,
RFDiffusion, another refold backend, OpenMM MD, or MMPBSA without adding backend
branches to the Runner.

## Installation

The lightweight core requires Python 3.10 or newer:

```bash
python -m pip install -e '.[test]'
```

Modeling backends intentionally remain optional because they require different
CUDA and scientific environments. On the development cluster:

- ColabDesign, ProteinMPNN, AF2, PyRosetta: `BindCraft`
- BoltzGen/Boltz2 co-fold runtime: `boltzgen`
- Boltz-2 structure and affinity screening: `binding_affinity`

Backend imports are lazy, so config, artifact, cache, and pipeline tooling work
without loading those environments.

## CLI

```bash
# Inspect a project-defined core-v2 Pipeline without loading a model.
epitopecraft pipeline describe myproject.workflow:build --config run.yaml
epitopecraft pipeline describe myproject.workflow:build --schema
epitopecraft pipeline plan myproject.workflow:build --config run.yaml

# Repository-owned BoltzGen protein/peptide co-folding runtime.
epitopecraft refold boltzgen --help

# Curated legacy utilities, now with one package implementation.
epitopecraft inspect animation --help
epitopecraft refold repair-missing --help
epitopecraft redesign mpnn-one --help

# Legacy standard workflow during migration.
epitopecraft standard-design --help
```

## Recipes

Filter and ProteinMPNN recipes retain nested boolean logic, arithmetic,
residue selectors, and overlapping amino-acid biases. Expressions are parsed by
a restricted AST evaluator rather than raw `eval`. Scientific logic that
cannot be represented in YAML can use an explicitly enabled, registered Python
hook.

Current compatibility fixtures include:

- `epitopecraft/pipelines/config/default_filter.yaml`
- `epitopecraft/pipelines/config/mpnn-sat-n-charge.json`
- `epitopecraft/pipelines/config/mpnn-rmC-recipe.json`

## Testing

Run tests on an interactive compute allocation:

```bash
conda activate BindCraft
python -m pytest
```

Contract tests are lightweight. GPU workflow tests run by default when the
required GPU and conda environment are available. They currently exercise real
ProteinMPNN cysteine redesign, BoltzGen/Boltz2 protein co-folding, and Boltz-2
known-site small-molecule structure plus affinity prediction.

## Developer documentation

- [Executable core-v2 walkthrough](demo.ipynb)
- [Core architecture](docs/architecture/core-v2.md)
- [Step contract](docs/architecture/step-contract.md)
- [Artifact and residue mapping](docs/architecture/artifact-model.md)
- [Pipeline authoring](docs/architecture/pipeline-authoring.md)
- [CATH BoltzGen site-discovery experiment](experiments/cath_site_discovery/README.md)

## License

See [LICENSE](LICENSE).
