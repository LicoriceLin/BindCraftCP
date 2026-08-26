#!/usr/bin/env python
"""Resume refold-passing base designs through a new MPNN recipe."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from epitopecraft.utils import DesignBatch, GlobalSettings, NEST_SEP


BASE_EXCLUDE_RE = re.compile(r"-(?:mpnn|rmC|satnc\w*)\d*$")
AFTER_REFOLD_FILTERS = (
    "filter:refold:multimer-1:pLDDT",
    "filter:refold:multimer-1:i-pAE",
    "filter:refold:multimer-2:pLDDT",
    "filter:refold:multimer-2:i-pAE",
    "filter:refold:best:binder_rmsd",
    "filter:aux:kappa2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="output/WDR5-t/backup-settinga.yaml")
    parser.add_argument("--design-path", default="output/WDR5-t")
    parser.add_argument(
        "--recipe",
        default="epitopecraft/pipelines/config/mpnn-sat-n-charge.json",
    )
    parser.add_argument("--prefix", default="satnc:")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument(
        "--selection",
        choices=["filter-kappa", "numeric-refold"],
        default="filter-kappa",
        help="filter-kappa matches the earlier WDR5-t rerun; numeric-refold uses pLDDT/i-pAE/binder RMSD values.",
    )
    parser.add_argument("--exclude-id-regex", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--relax-threads",
        type=int,
        default=1,
        help="Use 1 for GPU jobs unless the allocation has spare CPUs.",
    )
    return parser.parse_args()


def is_base_record(record_id: str) -> bool:
    return BASE_EXCLUDE_RE.search(record_id) is None


def passed_refold(record) -> bool:
    return all(record.get_metrics(key) is True for key in AFTER_REFOLD_FILTERS)


def passed_refold_numeric(record) -> bool:
    return (
        (record.get_metrics("refold:multimer-1:pLDDT", 0) >= 0.8)
        and (record.get_metrics("refold:multimer-1:i-pAE", 1) <= 0.35)
        and (record.get_metrics("refold:multimer-2:pLDDT", 0) >= 0.8)
        and (record.get_metrics("refold:multimer-2:i-pAE", 1) <= 0.35)
        and (record.get_metrics("refold:best:binder_rmsd", 999) <= 3.5)
    )


def select_ids(
    batch: DesignBatch,
    shard_index: int,
    shard_count: int,
    selection: str,
    exclude_id_regex: str = "",
) -> list[str]:
    predicate = passed_refold if selection == "filter-kappa" else passed_refold_numeric
    exclude_re = re.compile(exclude_id_regex) if exclude_id_regex else None
    selected = [
        record_id
        for record_id, record in sorted(batch.records.items())
        if is_base_record(record_id)
        and (exclude_re is None or not exclude_re.search(record_id))
        and predicate(record)
    ]
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < count")
    return [record_id for i, record_id in enumerate(selected) if i % shard_count == shard_index]


def configure_settings(args: argparse.Namespace) -> GlobalSettings:
    settings = GlobalSettings.from_file(args.settings)
    settings.binder_settings.design_path = args.design_path
    adv = settings.adv
    adv["mpnn_bias_recipe"] = args.recipe
    adv["n_mpnn_samples"] = args.samples
    adv["max_mpnn_sequences"] = args.max_sequences
    adv["mpnn-prefix"] = args.prefix
    adv["overwrite"] = False
    adv["overwrite_refold_only"] = False
    adv["relax-threads"] = args.relax_threads
    return settings


def build_steps(settings: GlobalSettings):
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
    args = parse_args()
    metrics_dir = Path(args.design_path) / "metrics"
    batch = DesignBatch.from_cache(metrics_dir)
    selected = select_ids(
        batch,
        args.shard_index,
        args.shard_count,
        args.selection,
        args.exclude_id_regex,
    )
    if args.limit is not None:
        selected = selected[: args.limit]

    print(
        f"Selected {len(selected)} base refold-passing records "
        f"from {metrics_dir} (shard {args.shard_index}/{args.shard_count})."
    )
    if selected:
        print("First IDs:", ", ".join(selected[:10]))
    if args.dry_run or not selected:
        return

    settings = configure_settings(args)
    steps = build_steps(settings)
    work = batch[selected]

    work = steps["mpnn"].process_batch(work)
    mpnn_suffix = args.prefix.strip(NEST_SEP)
    mpnn_ids = [
        record_id
        for record_id in work.records
        if re.search(rf"-{re.escape(mpnn_suffix)}\d+$", record_id)
    ]
    work = batch[mpnn_ids]
    print(f"Processing {len(mpnn_ids)} MPNN records with suffix -{mpnn_suffix}N.")

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
