---
name: epitopecraft-user
description: Use when running epitopecraft designs, preparing target or binder or advanced or filter settings, understanding the HalluDesign workflow, inspecting per-design metrics and ana_tracks, or locating outputs under design_path.
---

# Epitopecraft User

Use this skill when the task is about running designs or reading design outputs.
Prefer the real code in `epitopecraft/` over the root `README.md`; the README is partly stale.

## Quick Start

- Prefer running from the repo root
- On this cluster, prefer `tools/BindCraft-python` because it loads modules and the BindCraft env
- Main CLI:
  - `tools/BindCraft-python -m epitopecraft.main standard-design -t TARGET.json -b BINDER.json -a epitopecraft/pipelines/config/base_advanced_settings.yaml -f epitopecraft/pipelines/config/default_filter.yaml`
- `-a/--advanced-settings` may be repeated; later files override earlier ones
- There is also `design-all-in-one`, but the most exercised path in the current codebase is `standard-design`

## Before Running

- `TargetSettings.starting_pdb` is the structure Hallucinate and non-templated Refold use
- If `templated: true`, also provide `full_target_pdb` and `full_target_chain`
- `BinderSettings.helix_values` should be set explicitly; current Hallucinate sampling loops over it
- Initial hallucination count is `len(binder_lengths) * len(random_seeds) * len(helix_values)`

## Workflow Summary

`HalluDesign` currently wires the run as:

1. `Hallucinate`
2. `Filter(after:hallucinate)`
3. optional `Graft`
4. `Refold`
5. `AnnotRMSD`
6. `AnnotGyration`
7. `Filter(after:refold)`
8. `AnnotSurf`
9. `Relax`
10. `AnnotPolarOccupy`
11. optional PTM scorers
12. `MPNN`
13. optional `Graft` again
14. `Refold` again
15. the post-refold scorers again
16. `AnnotBCAux`
17. `AnnotPI`
18. `Filter(final)`

## Result Layout

Under `BinderSettings.design_path`, expect a layout like:

```text
design_path/
  <binder_name>.yaml
  backup-settinga.yaml
  metrics/
    <design-id>.json
    metrics.json        # only if a caller saves batch-level logs
  hallu/
    <design-id>.pdb
  graft/                # only when templated=true
    <design-id>.pdb
  refold/
    multimer/<design-id>-<model>.pdb
    monomer/<design-id>-<model>.pdb
  relax/
    <design-id>.pdb
  hotspot_analysis/     # if PseudoHotspot is run
```

Per-design JSON files in `metrics/` are the main source of truth. Most scorers write into `record.metrics` or `record.ana_tracks` inside those JSON files instead of creating separate output folders.

## Analysis Pointers

- Load a run with `DesignBatch.from_cache(<design_path>/metrics)`
- Use `batch.pass_df()` to expand `filter:*`
- Use `batch.df([...])` for selected metrics
- Use `record.metrics`, `record.ana_tracks`, and `record.pdb_files` for per-design inspection

## References

Read `references/run-design.md` for config fields, current metric keys, and the output folder map.
