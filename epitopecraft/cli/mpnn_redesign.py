#!/usr/bin/env python
"""Run MPNN redesign plus templated refold scoring for one cached design."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--input-record", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--recipe",
        default="epitopecraft/pipelines/config/mpnn-default-recipe.json",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-sequences", type=int, default=30)
    parser.add_argument("--prefix", default="mpnn:")
    parser.add_argument("--relax-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_batch(input_record: Path, out_dir: Path, overwrite: bool):
    from epitopecraft.utils import DesignBatch

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dest_record = metrics_dir / input_record.name
    if overwrite or not dest_record.exists():
        shutil.copy2(input_record, dest_record)
    batch = DesignBatch.from_cache(metrics_dir)
    batch.set_overwrite(overwrite)
    return batch


def configure_settings(args: argparse.Namespace):
    from epitopecraft.utils import GlobalSettings

    settings = GlobalSettings.from_file(args.settings)
    settings.binder_settings.design_path = args.out_dir
    adv = settings.adv
    adv["templated"] = True
    adv["mpnn_bias_recipe"] = args.recipe
    adv["n_mpnn_samples"] = args.samples
    adv["max_mpnn_sequences"] = args.max_sequences
    adv["mpnn-prefix"] = args.prefix
    adv["mpnn-pdb-input"] = "template"
    adv["overwrite"] = args.overwrite
    adv["overwrite_refold_only"] = args.overwrite
    adv["relax-threads"] = args.relax_threads
    return settings


def build_steps(settings):
    from epitopecraft.steps import (
        AnnotBCAux,
        AnnotGyration,
        AnnotPI,
        AnnotPolarOccupy,
        AnnotRMSD,
        AnnotSurf,
        Filter,
        Graft,
        MPNN,
        Refold,
        Relax,
    )
    from epitopecraft.utils import NEST_SEP

    adv = settings.adv
    filter_step = Filter(settings)
    graft = Graft(settings)
    refold = Refold(settings)
    mpnn = MPNN(settings)
    annot_rmsd = AnnotRMSD(settings)
    annot_surf = AnnotSurf(settings)
    annot_polar = AnnotPolarOccupy(settings)
    annot_gyr = AnnotGyration(settings)
    relax = Relax(settings)
    annot_aux = AnnotBCAux(settings)
    annot_pi = AnnotPI(settings)

    graft.config_pdb_purge(adv.setdefault("graft_stem", "graft"))
    template_pdb = graft.pdb_to_add[0]
    template_target_chain = settings.target_settings.full_target_chain
    template_binder_chain = settings.target_settings.new_binder_chain

    refold.config_pdb_input_key(template_pdb)
    refold.config_pdb_purge(adv.setdefault("refold_stem", "refold"))
    best_refold_pdb = refold.metrics_prefix + "best"

    annot_rmsd.config_pdb_input_key(
        pdb_to_take={"mobile": best_refold_pdb, "target": template_pdb}
    )
    annot_rmsd.config_metrics_prefix(best_refold_pdb + NEST_SEP)
    annot_surf.config_pdb_input_key(
        {"pdb_key": template_pdb, "binder_chain": template_binder_chain}
    )

    relax.config_pdb_input_key({"pdb_key": best_refold_pdb, "binder_chain": "B"})
    relax.config_pdb_purge(adv.setdefault("relax_stem", "relax"))
    relaxed_pdb = relax.pdb_to_add[0]
    annot_polar.config_pdb_input_key(relaxed_pdb)

    mpnn.config_pdb_input_key(template_pdb)
    mpnn.config_metrics_prefix(adv["mpnn-prefix"])
    adv["mpnn_binder_chain"] = template_binder_chain
    adv["mpnn_target_chain"] = template_target_chain

    annot_pi.config_params(pdb_to_take=relaxed_pdb)
    annot_aux.config_params(pdb_to_take={"pdb_key": relaxed_pdb, "binder_chain": "B"})

    return {
        "filter": filter_step,
        "graft": graft,
        "refold": refold,
        "mpnn": mpnn,
        "annot_rmsd": annot_rmsd,
        "annot_surf": annot_surf,
        "annot_polar": annot_polar,
        "annot_gyr": annot_gyr,
        "relax": relax,
        "annot_aux": annot_aux,
        "annot_pi": annot_pi,
    }


def main() -> None:
    from epitopecraft.utils import NEST_SEP

    args = parse_args()
    input_record = Path(args.input_record)
    out_dir = Path(args.out_dir)
    settings = configure_settings(args)
    batch = prepare_batch(input_record, out_dir, args.overwrite)
    record_id = input_record.stem
    if record_id not in batch.records:
        raise KeyError(f"{record_id} not found in {out_dir / 'metrics'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    settings.save(out_dir / "backup-settings.yaml")
    steps = build_steps(settings)
    work = batch[[record_id]]

    steps["mpnn"].process_batch(work)
    mpnn_suffix = args.prefix.strip(NEST_SEP)
    mpnn_ids = sorted(
        record_id_
        for record_id_ in batch.records
        if re.search(rf"-{re.escape(mpnn_suffix)}\d+$", record_id_)
    )
    print(f"Generated/loaded {len(mpnn_ids)} MPNN records.")
    if not mpnn_ids:
        return

    work = batch[mpnn_ids]
    steps["graft"].process_batch(work)
    steps["refold"].process_batch(work)
    steps["annot_rmsd"].process_batch(work)
    steps["annot_gyr"].process_batch(work)
    steps["filter"].set_recipe("after:refold").process_batch(work)
    steps["annot_surf"].process_batch(work)
    steps["relax"].process_batch(work)
    steps["annot_polar"].process_batch(work)
    steps["annot_aux"].process_batch(work)
    steps["annot_pi"].process_batch(work)
    steps["filter"].set_recipe("final").process_batch(work)
    print("Done.")


if __name__ == "__main__":
    main()
