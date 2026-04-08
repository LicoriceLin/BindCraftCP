---
name: epitopecraft-development
description: Use when modifying epitopecraft pipelines, settings, record or cache structures, step and scorer implementations, step wiring, CLI behavior, or when reviewing or refactoring the package architecture as a whole.
---

# Epitopecraft Development

Use this as the canonical development skill for `epitopecraft`, including Step and scorer work.
Prefer the current code over the root `README.md`; the README is useful for intent but not fully aligned with the package anymore.

## Development Workflow

1. Trace the real execution path first:
   - `epitopecraft/main.py`
   - `epitopecraft/pipelines/hallu_design.py`
   - `epitopecraft/utils/settings.py`
   - `epitopecraft/utils/design_record.py`
2. Decide which layer the change belongs to:
   - CLI and config loading
   - pipeline orchestration
   - settings schema
   - record or cache model
   - Step or scorer implementation
3. Keep data contracts stable unless the task explicitly includes a migration:
   - metric keys
   - `ana_tracks` names
   - `pdb_files` keys
   - run-folder layout under `design_path`
4. Prefer repo-local config files under `epitopecraft/pipelines/config`
5. Validate with focused checks such as `python -m py_compile` on touched files

## Step And Scorer Work

When the change is inside `epitopecraft.steps.*`:

1. Read the base class first:
   - `epitopecraft/steps/basestep.py`
   - `epitopecraft/steps/scorer/basescorer.py` for scorer-like steps
2. Read the caller next:
   - usually `epitopecraft/pipelines/hallu_design.py`
   - also `epitopecraft/utils/design_record.py` and `epitopecraft/utils/settings.py`
3. Match the implementation shape before changing behavior:
   - record-wise generation: `hallucinate.py`, `refold.py`, `relax.py`
   - record-wise scoring: `scorer/rmsd.py`, `scorer/annot_surf.py`, `scorer/aux_scores.py`
   - batch-only transformation or analysis: `filter.py`, `pseudo_hotspot.py`, `mpnn.py`

Keep these contracts stable unless the task explicitly changes them:

- `name` owns config keys like `f'{self.name}-prefix'` and `f'{self.name}-pdb-input'`
- `metrics_prefix` owns output namespaces
- `params_to_take` must cover every `adv` key the Step reads
- `process_batch(...)` is the public entry point and usually returns a `DesignBatch` or `DesignBatchSlice`
- `record.metrics` is for scalar or nested summaries
- `record.ana_tracks` is for residue-wise arrays or labels
- persisted structures should go through normal `pdb_to_add` plus purge flow

For `pdb_to_take`:

- use a plain string when one structure key is enough
- use a dict only when the Step truly needs named multi-input state, like `Relax`, `AnnotRMSD`, or `AnnotSurf`

## Package Rules

- Treat `HalluDesign` as the current source of truth for the standard workflow
- Preserve the `DesignBatch` plus `DesignRecord` cache model unless you are intentionally redesigning it
- Keep batch-level analyses relative to `batch.cache_dir.parent`
- Avoid introducing more heavy imports into lightweight modules such as config or cache helpers
- When changing keys or defaults, check both the caller pipeline and the default filter recipes

## References

- Read `references/code-map.md` for the package layout and current runtime contracts
- Read `references/step-patterns.md` when touching `epitopecraft.steps.*`
- Read `references/refactor-notes.md` for concrete refactor suggestions based on the current code
- Read `references/testing-notes.md` for the current, still-discussed testing direction before large refactors
