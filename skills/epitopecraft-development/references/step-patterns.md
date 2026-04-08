# Step Patterns

## Files To Compare Against

- `epitopecraft/steps/basestep.py`
- `epitopecraft/pipelines/hallu_design.py`
- `epitopecraft/utils/design_record.py`
- `epitopecraft/steps/refold.py`
- `epitopecraft/steps/relax.py`
- `epitopecraft/steps/mpnn.py`
- `epitopecraft/steps/filter.py`
- `epitopecraft/steps/scorer/basescorer.py`

## Common Patterns

### 1. Config keys vs output keys

- Config keys are usually tied to `self.name`
- Output metric keys are usually tied to `self.metrics_prefix`

Examples:

- config: `f'{self.name}-prefix'`
- config: `f'{self.name}-pdb-input'`
- output metric: `f'{self.metrics_prefix}score'`
- output metric: `f'{self.metrics_prefix}target_hotspot_residues'`

Do not collapse these two concepts together.

### 2. Batch-driven steps

If a Step analyzes many designs together:

- accept `input: DesignBatch`
- iterate over `input.records`
- use `record.pdb_files[self.pdb_to_take]` to find structures
- place derived artifacts under `input.cache_dir.parent`
- return `input`

This is the right shape for run-level analyses such as pseudo-hotspot mining, filtering, or MPNN expansion.

### 3. Record-driven steps

If a Step transforms one design at a time:

- implement `process_record`
- let `process_batch` orchestrate looping, saving, and optional purging

See `Refold`, `Relax`, and most scorers.

Typical current examples:

- `Hallucinate`: creates sequence plus a new in-memory PDB, then purges it to `design_path/hallu/`
- `Refold`: creates multiple PDB keys and metrics, then purges multimer and monomer models under `design_path/refold/`
- `Relax`: takes a dict input and writes one relaxed structure keyed as `f'{input_pdb_key}:{relax_stem}'`

### 4. Advanced settings

If a Step reads a value from `self.settings.adv`, it should usually:

- define a default with `adv.setdefault(...)`
- list the key in `params_to_take`

Typical examples:

- `dalphaball_path`
- `cyclize_peptide`
- `mpnn_sampling_temp`
- `pseudo_hotspot-freq-threshold`

### 5. pdb_to_take

Use `self.pdb_to_take` for the logical input structure key.

Default:

- implement `_default_pdb_input_key`

Complex cases:

- keep `self.pdb_to_take` as a dict only if the Step truly needs multiple named input fields, like `Relax`

Current examples:

- string: `Hallucinate`, `MPNN`, `AnnotGyration`
- dict: `Relax` uses `{'pdb_key', 'binder_chain'}`
- dict: `AnnotRMSD` uses `{'mobile', 'target'}`
- dict: `AnnotSurf` uses `{'pdb_key', 'binder_chain'}`

### 6. Output placement

For batch-level side products:

- use `root_dir = input.cache_dir.parent`
- write outputs to `root_dir / analysis_stem`

For record-level persisted PDBs:

- use normal `pdb_to_add` / purge flow from `BaseStep`

Current run layout conventions:

- cache JSON: `design_path/metrics/<record-id>.json`
- hallucinated PDBs: `design_path/hallu/<record-id>.pdb`
- grafted templates: `design_path/graft/<record-id>.pdb`
- refolds: `design_path/refold/multimer/` and `design_path/refold/monomer/`
- relaxed PDBs: `design_path/relax/<record-id>.pdb`
- batch analyses such as pseudo hotspot: `design_path/<analysis_stem>/...`

### 7. Metrics vs tracks

Use `record.metrics` for summary numbers and booleans.
Use `record.ana_tracks` for residue-wise arrays or labels.

Current examples:

- `AnnotRMSD` writes metrics like `refold:best:target_rmsd`
- `AnnotGyration` writes metrics like `aux:rog`
- `AnnotSurf` writes track arrays like `ppi`, `core`, `surf`
- `AnnotBCAux` writes tracks like `SS8`, `SS3`, `pLDDT`

## Common Pitfalls

- Writing metrics under `self.name` instead of `self.metrics_prefix`
- Reading design PDBs from hard-coded folders instead of `record.pdb_files`
- Using absolute or repo-fixed output paths when the run root is already available from `batch.cache_dir`
- Reading `adv` keys without listing them in `params_to_take`
- Returning analysis dicts instead of the processed batch when the Step is part of a pipeline
- Forgetting that some scorers intentionally use an empty default prefix, so tracks like `ppi` and `surf` may be top-level names
- Changing a Step key without checking `HalluDesign._config_steps()` and default filter recipes
- Confusing a persisted alias key such as `refold:best` with a separate generated file family

## Minimal Design Questions

Before implementing a new Step, answer:

1. Is this record-wise or batch-wise?
2. What is the default `pdb_to_take`?
3. Which values live in `adv` and therefore belong in `params_to_take`?
4. Which outputs are metrics, which are files, and which should use `metrics_prefix`?
5. Should outputs live on records, on batch metrics, or on disk under the run directory?
