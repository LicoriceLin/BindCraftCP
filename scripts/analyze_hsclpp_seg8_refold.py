#!/usr/bin/env python
"""Filter hsClpP seg8 Boltz refolds and run templated AF2 refold."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from epitopecraft.steps.refold import Refold
from epitopecraft.utils.design_record import DesignBatch, DesignRecord
from epitopecraft.utils.settings import (
    AdvancedSettings,
    BinderSettings,
    FilterSettings,
    GlobalSettings,
    TargetSettings,
)


DEFAULT_BASE_DIR = Path("input/hsClpP/scaffold-seg8-1500-fullAB-refold")
DEFAULT_TARGET_PDB = Path("input/hsClpP/scaffold-full-1500/scaffold-full.cif")
DEFAULT_ADVANCED = Path("epitopecraft/pipelines/config/base_advanced_settings.yaml")
DEFAULT_TEMPLATE_PATCH = Path("epitopecraft/pipelines/config/patch_templated.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter refold_metrics.csv, generate DesignRecord JSON cache from "
            "design CIFs, run epitopecraft Refold with templating, and plot "
            "Boltz iPTM vs AF2 refold iPAE."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--target-pdb", type=Path, default=DEFAULT_TARGET_PDB)
    parser.add_argument("--advanced-settings", type=Path, default=DEFAULT_ADVANCED)
    parser.add_argument("--template-patch", type=Path, default=DEFAULT_TEMPLATE_PATCH)
    parser.add_argument("--pass-csv", type=Path, default=None)
    parser.add_argument("--out-dir-name", default="af2_refold")
    parser.add_argument("--target-chains", default="A,B")
    parser.add_argument("--binder-chain", default="C")
    parser.add_argument("--iptm-threshold", type=float, default=0.35)
    parser.add_argument("--bb-rmsd-threshold", type=float, default=3.5)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument(
        "--run-monomer",
        action="store_true",
        help="Also run monomer refolds. Off by default because iPAE is from complex models.",
    )
    parser.add_argument(
        "--af2-multimer-refold",
        action="store_true",
        help=(
            "Set use_multimer_design=False so Refold validates with AF2-multimer "
            "models 1-5 instead of AF2-ptm models 1-2."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate records and rerun Refold even if cached outputs exist.",
    )
    parser.add_argument(
        "--skip-refold",
        action="store_true",
        help="Only write pass.csv and DesignRecord JSON files.",
    )
    return parser.parse_args()


def json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_sequence(seq: str) -> str:
    return "".join(ch for ch in seq if ch.isalpha()).upper()


def cif_chain_sequence(cif_path: Path, chain_id: str) -> str:
    cif_dict = MMCIF2Dict(str(cif_path))
    asym_ids = cif_dict["_struct_asym.id"]
    entity_ids = cif_dict["_struct_asym.entity_id"]
    entity_by_chain = dict(zip(asym_ids, entity_ids))
    if chain_id not in entity_by_chain:
        raise ValueError(f"{cif_path} does not contain chain {chain_id!r}")

    seq_entity_ids = cif_dict["_entity_poly.entity_id"]
    seqs = cif_dict["_entity_poly.pdbx_seq_one_letter_code"]
    seq_by_entity = dict(zip(seq_entity_ids, seqs))
    entity_id = entity_by_chain[chain_id]
    if entity_id not in seq_by_entity:
        raise ValueError(f"{cif_path} chain {chain_id!r} has no polymer sequence")
    return clean_sequence(seq_by_entity[entity_id])


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path, overwrite: bool = False) -> Path:
    if pdb_path.exists() and not overwrite:
        return pdb_path

    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))
    return pdb_path


def source_metrics(row: pd.Series, template_cif: Path, template_pdb: Path) -> dict[str, Any]:
    return {
        "boltz": {col: json_scalar(row[col]) for col in row.index},
        "source": {
            "template_cif": str(template_cif),
            "template_pdb": str(template_pdb),
        },
    }


def build_settings(args: argparse.Namespace) -> GlobalSettings:
    extra_patch = {
        "templated": True,
        "refold-run-monomer": bool(args.run_monomer),
    }
    if args.af2_multimer_refold:
        extra_patch["use_multimer_design"] = False
    return GlobalSettings(
        target_settings=TargetSettings(
            starting_pdb=str(args.target_pdb),
            chains=args.target_chains,
            full_target_pdb=str(args.target_pdb),
            full_target_chain=args.target_chains,
        ),
        binder_settings=BinderSettings(
            design_path=str(args.base_dir),
            binder_name="scaffold-seg8",
            binder_lengths=[],
            random_seeds=[],
            global_seed=args.global_seed,
        ),
        advanced_settings=AdvancedSettings(
            advanced_paths=[str(args.advanced_settings), str(args.template_patch)],
            extra_patch=extra_patch,
        ),
        filter_settings=FilterSettings(filters_path=None),
    )


def load_cached_record(cache_dir: Path, record_id: str) -> DesignRecord | None:
    record_json = cache_dir / f"{record_id}.json"
    if not record_json.exists():
        return None
    return DesignRecord.from_json(record_json)


def build_batch(args: argparse.Namespace, pass_df: pd.DataFrame) -> DesignBatch:
    out_dir = args.base_dir / args.out_dir_name
    cache_dir = out_dir / "metrics"
    template_pdb_dir = out_dir / "templates_pdb"
    batch = DesignBatch(cache_dir=cache_dir, overwrite=args.overwrite)

    for _, row in pass_df.iterrows():
        record_id = str(row["id"])
        template_cif = args.base_dir / "designs" / f"{record_id}.cif"
        if not template_cif.exists():
            raise FileNotFoundError(f"Template CIF not found: {template_cif}")

        template_pdb = convert_cif_to_pdb(
            template_cif,
            template_pdb_dir / f"{record_id}.pdb",
            overwrite=args.overwrite,
        )
        sequence = cif_chain_sequence(template_cif, args.binder_chain)

        record = None if args.overwrite else load_cached_record(cache_dir, record_id)
        if record is None:
            record = DesignRecord(id=record_id, sequence=sequence)
        else:
            record.sequence = sequence
        record.pdb_files["template"] = str(template_pdb)
        record.metrics.update(source_metrics(row, template_cif, template_pdb))

        batch.add_record(record)
        batch.save_record(record_id)

    return batch


def read_ipae(record: DesignRecord, model_num: int) -> float:
    value = record.get_metrics(f"refold:multimer-{model_num}:i-pAE")
    if value is None:
        return math.nan
    return float(value)


def update_pass_csv(
    pass_df: pd.DataFrame,
    batch: DesignBatch,
    pass_csv: Path,
    model_nums: list[int],
) -> pd.DataFrame:
    records = batch.records
    pass_df = pass_df.copy()
    pass_df["template_cif"] = [
        records[str(record_id)].metrics["source"]["template_cif"] for record_id in pass_df["id"]
    ]
    pass_df["template_pdb"] = [
        records[str(record_id)].metrics["source"]["template_pdb"] for record_id in pass_df["id"]
    ]
    ipae_cols = []
    for model_num in model_nums:
        col = f"af2_refold_multimer_{model_num}_ipae"
        ipae_cols.append(col)
        pass_df[col] = [read_ipae(records[str(record_id)], model_num) for record_id in pass_df["id"]]
    pass_df["af2_refold_mean_ipae"] = pass_df[ipae_cols].mean(axis=1)
    pass_df.to_csv(pass_csv, index=False)
    return pass_df


def plot_correlation(pass_df: pd.DataFrame, plot_path: Path) -> tuple[float, float]:
    valid = pass_df[["design_to_target_iptm", "af2_refold_mean_ipae"]].dropna()
    pearson = valid["design_to_target_iptm"].corr(valid["af2_refold_mean_ipae"], method="pearson")
    spearman = valid["design_to_target_iptm"].corr(valid["af2_refold_mean_ipae"], method="spearman")

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
    ax.scatter(
        valid["design_to_target_iptm"],
        valid["af2_refold_mean_ipae"],
        s=38,
        color="#226f54",
        edgecolor="#0b1f17",
        linewidth=0.5,
    )
    ax.set_xlabel("boltz-iptm")
    ax.set_ylabel("mean AF2 refold iPAE")
    ax.set_title("Boltz iPTM vs AF2 Refold iPAE")
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {pearson:.3f}\nSpearman rho = {spearman:.3f}\nn = {len(valid)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#c7c7c7"},
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    return float(pearson), float(spearman)


def main() -> None:
    args = parse_args()
    args.base_dir = args.base_dir.absolute()
    args.target_pdb = args.target_pdb.absolute()
    args.advanced_settings = args.advanced_settings.absolute()
    args.template_patch = args.template_patch.absolute()
    metrics_csv = (args.metrics_csv or (args.base_dir / "refold_metrics.csv")).absolute()
    pass_csv = (args.pass_csv or (args.base_dir / "pass.csv")).absolute()
    out_dir = args.base_dir / args.out_dir_name
    plot_path = out_dir / "boltz_iptm_vs_af2_mean_ipae.png"

    df = pd.read_csv(metrics_csv)
    pass_df = df[
        (df["design_to_target_iptm"] > args.iptm_threshold)
        & (df["bb_rmsd"] < args.bb_rmsd_threshold)
    ].copy()
    pass_df.to_csv(pass_csv, index=False)
    print(f"filtered_entries={len(pass_df)}")
    print(f"pass_csv={pass_csv}")

    settings = build_settings(args)
    batch = build_batch(args, pass_df)
    print(f"record_json_dir={batch.cache_dir}")
    model_nums = [n + 1 for n in Refold(settings).prediction_models]
    print(f"use_multimer_design={settings.adv['use_multimer_design']}")
    print(f"refold_model_nums={model_nums}")

    if not args.skip_refold and len(batch) > 0:
        refold = Refold(settings)
        refold.process_batch(batch, pdb_purge_stem=args.out_dir_name, pdb_to_take="template")

    updated = update_pass_csv(pass_df, batch, pass_csv, model_nums)
    if updated["af2_refold_mean_ipae"].notna().any():
        pearson, spearman = plot_correlation(updated, plot_path)
        print(f"plot={plot_path}")
        print(f"pearson_r={pearson:.6f}")
        print(f"spearman_rho={spearman:.6f}")
    else:
        print("No AF2 refold iPAE values available; skipped correlation plot.")

    summary_path = out_dir / "summary.json"
    summary = {
        "filtered_entries": int(len(pass_df)),
        "pass_csv": str(pass_csv),
        "plot": str(plot_path),
        "use_multimer_design": bool(settings.adv["use_multimer_design"]),
        "refold_model_nums": model_nums,
    }
    if updated["af2_refold_mean_ipae"].notna().any():
        summary.update(
            {
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
