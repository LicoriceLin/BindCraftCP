# CATH site-discovery experiment

These scripts preserve the repository's BoltzGen-based target-analysis
experiment without checking its large PDB/mmCIF inputs or generated results
into source control. The experiment runs many unconstrained designs, extracts
frequently contacted target residues, creates local-motif and full-target
BoltzGen conditions, prepares refold validation jobs, and summarizes success
rates.

The scripts are experiment drivers, not stable package APIs. Their default
paths still point at `input/cath_testset_boltz` so the existing data layout can
be reused. Public implementations belong under `epitopecraft`; in particular,
new workflows should use `BindingSiteDiscovery`, `StructureArtifact`, and the
repository-owned BoltzGen refold backend instead of copying scientific logic
from this directory.

A typical sequence is:

1. Generate unconstrained BoltzGen designs.
2. Run `extract_diffusion_hotspots.py`.
3. Run `prepare_motif_boltzgen_design.py` and/or
   `prepare_motif_full_boltzgen_design.py`.
4. Run the generated design manifests with the site's scheduler wrapper.
5. Run `prepare_core_refold1.py` or `prepare_motif_refold1.py`, validate with
   `python -m epitopecraft.backends.boltzgen.peptide_refold_cli`, and use the
   repair-manifest scripts for incomplete cases.
6. Run the two `summarize_*_success.py` scripts.

Scheduler scripts are intentionally not copied here: the originals encode
site-specific partitions, absolute paths, and environment setup. Pass a local
scheduler template to `prepare_motif_boltzgen_design.py --submit-template` or
consume the CSV manifests from a portable submit wrapper.
