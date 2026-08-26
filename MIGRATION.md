# Core v2 migration notes

Core v2 intentionally does not preserve all historical Python or settings
interfaces.

- Import backend Steps from their defining modules. The package root no longer
  star-imports ColabDesign, PyRosetta, or PyMOL dependencies.
- New Step configuration is scoped under `steps.<instance-id>.params`; dynamic
  keys such as `refold-prefix` are legacy-only.
- New workflows exchange typed artifacts and `DesignSet`, not implicit
  `pdb_files` string keys. Legacy adapters will be removed after their backend
  Steps are migrated.
- Residue selections should use canonical entity positions. Backend chain and
  residue ids are artifact-local and may change after hallucination, graft, or
  refolding.
- Runner pickle caches are local resume state, not a portable interchange or
  compatibility format.
- Boltz-2 screening defaults to no target template. Canonical pocket
  constraints are supported; direct PDB templates remain opt-in until a robust
  processed-template adapter replaces the current Boltz reader path.
- Dataset-specific CATH target-analysis scripts live under
  `experiments/cath_site_discovery`; their CSV interfaces are transitional,
  while reusable site aggregation lives in `BindingSiteDiscovery`.

The checkpoint before this refactor is commit `ad40be0` on `reconstruct`.
