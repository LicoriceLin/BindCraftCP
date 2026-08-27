# Core v2 architecture

Core v2 is a deliberate API break. New development should target the modules
under `epitopecraft.core`; legacy pipelines remain available only while their
backends are moved behind the new contracts.

The architecture has three boundaries:

1. A `Step` declares its instance id, typed ports, and dataclass config.
2. An `Artifact` carries scientific data, provenance, and coordinate mapping.
3. A `Pipeline` describes a DAG; `PipelineRunner` owns execution, resume cache,
   run directories, and manifests.

Backend packages may import heavy modeling libraries. Core modules must not,
apart from the required Gemmi coordinate model used by `StructureArtifact`.
Adding a design, refold, MD, or scoring implementation should require a new
Step with compatible ports, not an edit to a central enum or runner branch.

## Current backend boundary

Boltz-2 known-site screening accepts both protein sequences and SMILES. Direct
target templates are opt-in because the currently installed Boltz PDB template
reader does not reliably accept local or discontinuously numbered structures.
The adapter therefore defaults to sequence plus canonical pocket constraints;
a future processed-template adapter can enable templates without changing the
pipeline contract.

## Stability policy

During the refactor, correctness and internal consistency take precedence over
old Python interfaces and serialized settings. Migration hazards are recorded
in `MIGRATION.md`; compatibility shims are not a design requirement.

## Persistence safety

Runner resume caches use Python pickle and are intended only for run directories
created locally by EpitopeCraft. Never resume from an untrusted or downloaded
run directory. Portable scientific outputs should be stored as structures,
tables, YAML/JSON manifests, and residue-map TSV files instead.
