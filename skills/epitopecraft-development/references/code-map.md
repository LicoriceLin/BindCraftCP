# Code Map

## Main Layers

### CLI

- `epitopecraft/main.py`

Current commands:

- `standard-design`
- `design-all-in-one`

`standard-design` is the clearest and most direct path to the current `HalluDesign` pipeline.

### Pipelines

- `epitopecraft/pipelines/base_pipeline.py`: pipeline base class and settings persistence
- `epitopecraft/pipelines/hallu_design.py`: standard design pipeline
- `epitopecraft/pipelines/epitope_detect.py`: hotspot-oriented pipeline, still rough
- `epitopecraft/pipelines/removeC.py`: specialized follow-up pipeline, also rough

### Settings and cache

- `epitopecraft/utils/settings.py`
- `epitopecraft/utils/design_record.py`
- `epitopecraft/utils/utils.py`

Important contracts:

- `GlobalSettings.adv` is the runtime advanced-settings dict
- `DesignRecord` stores `sequence`, `pdb_strs`, `pdb_files`, `metrics`, and `ana_tracks`
- `DesignBatch.cache_dir` points at the JSON cache directory, usually `design_path/metrics`

### Steps

- `epitopecraft/steps/basestep.py`: base contract
- `epitopecraft/steps/hallucinate.py`
- `epitopecraft/steps/refold.py`
- `epitopecraft/steps/relax.py`
- `epitopecraft/steps/mpnn.py`
- `epitopecraft/steps/filter.py`
- `epitopecraft/steps/pseudo_hotspot.py`
- `epitopecraft/steps/scorer/*.py`

## Real Runtime Flow

`main.py -> HalluDesign.__init__ -> HalluDesign._init_steps() -> HalluDesign._config_steps() -> HalluDesign.run()`

Most key remapping happens inside `HalluDesign._config_steps()`, not inside the default Step constructors.
If a key looks surprising, check `_config_steps()` before assuming the default is authoritative.

## Current File And Key Conventions

### Run root

`BinderSettings.design_path`

### Cache root

`design_path / metrics_stem`

Usually:

`design_path/metrics`

### PDB directories

- `design_path/hallu`
- `design_path/graft`
- `design_path/refold/multimer`
- `design_path/refold/monomer`
- `design_path/relax`

### Per-record JSON

`design_path/metrics/<record-id>.json`

### Batch-level metrics

`design_path/metrics/metrics.json`

This only exists if the caller actually saves batch-level logs.

## Current Stale Or Legacy Areas To Treat Carefully

- Root `README.md`
- `epitope_detect.py`
- `removeC.py`
- top-level `epitopecraft/__init__.py` imports, because they still pull in heavy runtime dependencies

## Validation Habits

- For Step changes, compile the touched files and verify key names against `HalluDesign`
- For settings changes, verify both `from_file` and `save`
- For cache changes, test `DesignBatch.from_cache(...)`, `save_record(...)`, and any symlink helpers
- For larger refactors, read `testing-notes.md` first; the concrete suite shape is not fully settled yet
