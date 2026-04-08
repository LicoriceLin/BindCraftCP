# Running Designs

## Code Map

- `epitopecraft/main.py`: CLI entrypoints
- `epitopecraft/pipelines/hallu_design.py`: the real standard design workflow
- `epitopecraft/utils/settings.py`: config dataclasses and file loading
- `epitopecraft/utils/design_record.py`: run cache format
- `epitopecraft/pipelines/config/base_advanced_settings.yaml`: default advanced knobs
- `epitopecraft/pipelines/config/default_filter.yaml`: current filter recipes

## Recommended Run Command

Use the environment wrapper on this cluster:

```bash
tools/BindCraft-python -m epitopecraft.main standard-design \
  -t epitopecraft/test/test_target_setting.json \
  -b epitopecraft/test/test_binder_setting.json \
  -a epitopecraft/pipelines/config/base_advanced_settings.yaml \
  -f epitopecraft/pipelines/config/default_filter.yaml
```

Add more `-a` flags to stack patches such as:

- `epitopecraft/pipelines/config/patch_templated.json`
- `epitopecraft/pipelines/config/patch_cyc.json`
- `epitopecraft/pipelines/config/patch_4s_mc.json`

Later `-a` files override earlier ones.

## Settings Notes

### Target settings

- `starting_pdb`: trimmed or design-time target PDB
- `chains`: target chains exposed to Hallucinate and non-templated Refold
- `target_hotspot_residues`: optional hotspot string like `A12,A17,B5`
- `full_target_pdb`: intact target, needed for templated grafting and some analyses
- `full_target_chain`: chain ids in the intact target

If `full_target_chain` already contains `B`, the package auto-picks a different `new_binder_chain`.

### Binder settings

- `design_path`: run root
- `binder_name`: record-id prefix
- `binder_lengths`: sampled binder lengths
- `random_seeds`: sampled Hallucinate seeds
- `helix_values`: helicity-loss values; set this explicitly in the current codebase
- `global_seed`: shared fallback seed

Initial hallucination IDs are:

```text
<binder_name>-000
<binder_name>-001
...
```

The zero-padding width depends on the total number of hallucination combinations.

MPNN-derived designs are appended as:

```text
<old-id>-mpnn1
<old-id>-mpnn2
...
```

## Current Stage Outputs

### Hallucinate

- sequence on `record.sequence`
- PDB key: `halu`
- metrics:
  - `config:helix`
  - `config:length`
  - `config:seed`
  - `halu:pLDDT`
  - `halu:pTM`
  - `halu:i-pTM`
  - `halu:pAE`
  - `halu:i-pAE`

### Graft

- only when `templated: true`
- grafted PDB key: `template`

### Refold

- PDB keys:
  - `refold:multimer-1`, `refold:multimer-2`, ...
  - `refold:monomer-1`, `refold:monomer-2`, ...
  - `refold:best`
- metrics include:
  - `refold:multimer-*:pLDDT`
  - `refold:multimer-*:i-pAE`
  - `refold:monomer-*:pLDDT`
  - `refold:best:i-pAE`

### RMSD and Gyration

- `refold:best:target_rmsd`
- `refold:best:binder_rmsd`
- `aux:rog`
- `aux:kappa2`

### Surface annotation

Default metrics prefix is empty, so the residue-wise tracks are top-level:

- `rSASA_in_monomer`
- `rSASA_in_complex`
- `ppi`
- `core`
- `surf`

### Relax

- relaxed PDB key: `refold:best:relax`

### Polar occupancy

Default tracks:

- `h-bond`
- `salt-bridge`
- `unsat_ppi_polar`
- `aatype`

### MPNN

- summary metrics on both old and new records:
  - `mpnn:score`
  - `mpnn:seqid`

### Final auxiliary scoring

- metrics:
  - `aux:relaxed_clashes`
  - `aux:surf_hydro`
  - `aux:interface_hydro`
  - `rst:binder_score`
  - `rst:packstat`
  - `rst:shape_complementarity`
  - `rst:dG`
  - `rst:dSASA`
  - `rst:hbond`
  - `rst:unsat_hbond`
- tracks:
  - `SS8`
  - `SS3`
  - `pLDDT`
- pI:
  - `pi-fold`

### Filter

Filter booleans are nested under `filter:*`.
The current default recipes are:

- `after:hallucinate`: `halu`
- `after:refold`: `refold`, `aux:kappa2`
- `final`: `rst`, `aux:surf_hydro`, `pi-fold`

## Inspecting Results In Python

```python
from epitopecraft.utils.design_record import DesignBatch

batch = DesignBatch.from_cache("output/test/metrics")
pass_df = batch.pass_df()
score_df = batch.df(["halu:pLDDT", "refold:best:i-pAE", "rst:dG", "pi-fold"])
record = batch["tseg6-000"]

print(record.sequence)
print(record.get_metrics("refold:best:binder_rmsd"))
print(record.ana_tracks["ppi"])
print(record.pdb_files["refold:best:relax"])
```

## Batch Analyses

`PseudoHotspot.process_batch(...)` writes batch-level outputs under `batch.cache_dir.parent / analysis_stem`.
The default stem is `hotspot_analysis`, and outputs include:

- `*_hotspot_table.csv`
- `*_hotspot_group_summary.csv`
- `*_hotspot_method.txt`
- `*_hotspot_groups.pse`
- `*_target_hotspot_residues.txt`
- `<binder_name>_motif/` with motif PDBs and a PyMOL session
