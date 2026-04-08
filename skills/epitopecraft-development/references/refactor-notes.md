# Refactor Notes

## Highest-Value Refactor Targets

### 1. Separate lightweight config and cache code from heavyweight imports

Today, importing `epitopecraft` pulls in `steps`, which in turn pulls in heavy runtime dependencies such as PyRosetta, ColabDesign, and PyMOL through `epitopecraft/__init__.py` and `epitopecraft/steps/basestep.py`.

Suggested direction:

- make `epitopecraft/__init__.py` minimal
- avoid star-importing all Steps at package import time
- keep `settings.py`, `design_record.py`, and small utilities importable without the full modeling stack

Why it matters:

- faster tooling and tests
- easier static analysis
- easier scripting for config and cache inspection

### 2. Split orchestration from key remapping

`HalluDesign._config_steps()` currently mixes several different responsibilities:

- decides which structure counts as template versus best-refold versus relax input
- overrides Step IO such as `pdb_to_take`, `metrics_prefix`, and `pdb_purge_stem`
- derives chain mappings for downstream steps
- writes some derived values back into shared config

Suggested direction:

- add a dedicated layout-resolution phase before running the pipeline
- compute a small immutable pipeline layout object once, for example `template_pdb_key`, `best_refold_key`, `relaxed_pdb_key`, `template_target_chain`, and `template_binder_chain`
- derive per-step `StepBinding` objects from that layout instead of mutating each Step ad hoc
- keep Step behavior parameters separate from pipeline IO bindings, even if they still live in one user-visible config tree

The key design goal is not "no overrides at all". The goal is "overrides happen once, explicitly, and without action at a distance".

### 3. Replace scattered `adv.setdefault(...)` mutation with one config-resolution pass

All advanced parameters may remain user-callable. The problem is not that some values are hidden; the problem is that default resolution is currently scattered across constructors, pipeline setup, and runtime methods.

Today, reading config often mutates config:

- Step constructors call `config_metrics_prefix()` and `config_pdb_input_key()`
- many methods call `adv.setdefault(...)` while running
- some pipeline-derived values are written back into `adv`

Suggested direction:

- keep one user-visible config surface
- resolve defaults and derived values once into a single effective config view near pipeline initialization
- let Steps read from that resolved view without mutating shared state during normal execution

Good implementation options:

- `OmegaConf` without Hydra, using merge plus resolve plus dotlist override only
- a lightweight in-repo registry plus deep-merge approach if you want zero new dependency

What matters more than the library choice is this contract:

- every parameter may be set explicitly by the user
- defaults are materialized once
- runtime code does not keep rewriting shared config as it runs

### 4. Clean up settings serialization and persistence semantics

Suggested direction:

- define explicit `to_dict()` or `settings` behavior for each settings class
- save a single resolved config view instead of relying on whatever has been accumulated through `setdefault(...)`
- decide whether saved files are meant for restart, audit, or both
- avoid path-dependent settings snapshots created at multiple lifecycle points unless they are clearly named as checkpoints

This is not about hiding internal parameters from users. It is about making the saved config deterministic and easier to reason about.

### 5. Unify run layout and analysis persistence

Per-record JSON files are the main source of truth, while batch-level `metrics.json` is optional and easy to miss.

Suggested direction:

- make batch log persistence consistent
- clearly separate per-record cache, pipeline metadata, and batch analyses
- consider a dedicated run manifest at the run root

## Suggested Refactor Sequence

1. Make top-level imports lighter where possible
2. Extract pipeline layout resolution into a clearer, testable helper
3. Move Step IO binding to an explicit layer instead of scattered overrides
4. Introduce one-time config resolution and cleaner save semantics
5. Only then widen the public CLI or add new pipelines

## Status Notes

- Several concrete correctness bugs and the worst save-side-effect issue were already fixed in the current codebase
- Testing strategy before the larger refactor is still under discussion; see `testing-notes.md`
