# Artifact and residue-reference model

Artifacts are backend-neutral values passed between Steps:

- `StructureArtifact` stores raw PDB/mmCIF and a `ResidueMap`.
- `SequenceArtifact` names a canonical protein entity.
- `LigandArtifact` stores a small molecule without pretending it is a protein.
- `SelectionArtifact` stores canonical residue positions.
- `TrajectoryArtifact` links an MD trajectory to its topology.
- `TableArtifact` stores portable tabular results.

## Canonical reference

`ReferenceModel.from_target()` uses the complete target structure when both a
complete structure and local epitope structure are supplied. Target author
chain ids, residue ids, and insertion codes are retained. A de novo binder gets
a new non-conflicting entity and chain.

Every predicted structure is mapped from backend-local residue ids to
`CanonicalResidueId(entity_id, position)`. Chain names are not trusted:
`map_structure_auto()` matches backend chains to entities by sequence before
building residue correspondences. This covers HalluDesign A/B renumbering,
grafting a local epitope into a complete target, and BoltzGen renaming B/Z to
A/B during refolding.

Use `summary()`, `validate()`, `export(numbering="reference")`, and
`write_debug_bundle()` when diagnosing a mapping. The debug bundle contains the
raw structure, canonicalized structure when supported, `residue_map.tsv`,
`entities.yaml`, validation results, and provenance.
