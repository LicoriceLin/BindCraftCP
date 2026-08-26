#!/usr/bin/env python
"""Extract pseudo-hotspots from CATH BoltzGen diffusion outputs."""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd
import yaml
from pymol import cmd

if (Path.cwd() / "epitopecraft").is_dir():
    sys.path.insert(0, str(Path.cwd()))

from epitopecraft.steps.pseudo_hotspot import PseudoHotspot
from epitopecraft.utils.design_record import DesignBatch, DesignRecord
from epitopecraft.utils.settings import (
    AdvancedSettings,
    BinderSettings,
    FilterSettings,
    GlobalSettings,
    TargetSettings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run epitopecraft.steps.pseudo_hotspot.PseudoHotspot on "
            "input/cath_testset_boltz/outputs/*/intermediate_designs/*.cif."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("input/cath_testset_boltz/outputs"),
    )
    parser.add_argument(
        "--inputs-root",
        type=Path,
        default=Path("input/cath_testset_boltz/inputs"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("input/cath_testset_boltz/outputs/hotspot_run_summary.csv"),
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional target IDs to process. Defaults to every output target.",
    )
    parser.add_argument(
        "--motif-sizes",
        nargs="*",
        type=int,
        default=[],
        help="Optional top-k motif sizes. Empty means only hotspot outputs.",
    )
    parser.add_argument(
        "--freq-threshold",
        type=float,
        default=0.5,
        help="Frequency threshold for target_hotspot_residues.txt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even if hotspot outputs already exist.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove each target hotspot directory before recomputing.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first target failure.",
    )
    return parser.parse_args()


def load_target_input(input_yaml: Path) -> tuple[Path, str]:
    config = yaml.safe_load(input_yaml.read_text())
    file_entity = next(entity["file"] for entity in config["entities"] if "file" in entity)
    target_pdb = Path(file_entity["path"])
    chains = [
        include["chain"]["id"]
        for include in file_entity.get("include", [])
        if "chain" in include
    ]
    return target_pdb, ",".join(chains) if chains else "A"


def build_settings(
    target_id: str,
    target_dir: Path,
    target_pdb: Path,
    target_chains: str,
    motif_sizes: list[int],
    freq_threshold: float,
) -> GlobalSettings:
    adv_patch = {
        "pseudo_hotspot-pdb-input": "diffusion",
        "pseudo_hotspot-analysis-stem": "hotspot",
        "pseudo_hotspot-motif-sizes": motif_sizes,
        "pseudo_hotspot-freq-threshold": freq_threshold,
    }
    return GlobalSettings(
        target_settings=TargetSettings(
            full_target_pdb=str(target_pdb),
            chains=target_chains,
            full_target_chain=target_chains,
        ),
        binder_settings=BinderSettings(
            design_path=str(target_dir),
            binder_name=target_id,
            binder_lengths=[],
            random_seeds=[],
        ),
        advanced_settings=AdvancedSettings(
            advanced_paths=["none"],
            extra_patch=adv_patch,
        ),
        filter_settings=FilterSettings(filters_path="none"),
    )


def build_batch(target_dir: Path, cif_files: list[Path]) -> DesignBatch:
    batch = DesignBatch(target_dir / "config")
    for cif_file in cif_files:
        batch.add_record(
            DesignRecord(
                id=cif_file.stem,
                sequence="",
                pdb_files={"diffusion": str(cif_file)},
            )
        )
    return batch


def target_ids(outputs_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return sorted(requested)
    return sorted(
        path.name
        for path in outputs_root.iterdir()
        if path.is_dir() and (path / "intermediate_designs").is_dir()
    )


def expected_outputs(target_dir: Path, target_id: str) -> list[Path]:
    hotspot_dir = target_dir / "hotspot"
    return [
        hotspot_dir / f"{target_id}_hotspot_table.csv",
        hotspot_dir / f"{target_id}_hotspot_group_summary.csv",
        hotspot_dir / f"{target_id}_hotspot_method.txt",
        hotspot_dir / f"{target_id}_hotspot_groups.pse",
        hotspot_dir / f"{target_id}_target_hotspot_residues.txt",
    ]


def process_target(args: argparse.Namespace, target_id: str) -> dict[str, object]:
    target_dir = args.outputs_root / target_id
    input_yaml = args.inputs_root / f"{target_id}.yaml"
    hotspot_dir = target_dir / "hotspot"
    summary: dict[str, object] = {
        "target": target_id,
        "status": "unknown",
        "n_designs": 0,
        "n_hotspot_rows": 0,
        "n_selected_residues": 0,
        "n_groups": 0,
        "target_chains": "",
        "target_pdb": "",
        "analysis_dir": str(hotspot_dir),
        "error": "",
    }

    if not input_yaml.exists():
        summary.update(status="failed", error=f"missing input yaml: {input_yaml}")
        return summary

    cif_files = sorted((target_dir / "intermediate_designs").glob("*.cif"))
    summary["n_designs"] = len(cif_files)
    if not cif_files:
        summary.update(status="failed", error="no intermediate_designs/*.cif files")
        return summary

    if not args.overwrite and all(path.exists() for path in expected_outputs(target_dir, target_id)):
        summary.update(status="skipped")
        return summary

    if args.clean and hotspot_dir.exists():
        shutil.rmtree(hotspot_dir)

    target_pdb, target_chains = load_target_input(input_yaml)
    summary["target_pdb"] = str(target_pdb)
    summary["target_chains"] = target_chains
    if not target_pdb.exists():
        summary.update(status="failed", error=f"missing target pdb: {target_pdb}")
        return summary

    settings = build_settings(
        target_id=target_id,
        target_dir=target_dir,
        target_pdb=target_pdb,
        target_chains=target_chains,
        motif_sizes=args.motif_sizes,
        freq_threshold=args.freq_threshold,
    )
    batch = build_batch(target_dir, cif_files)
    step = PseudoHotspot(settings)
    step.process_batch(batch, analysis_stem="hotspot", pdb_to_take="diffusion")
    cmd.delete("all")

    if step.hotspot_df is not None:
        summary["n_hotspot_rows"] = int(len(step.hotspot_df))
    if step.method_info:
        summary["n_groups"] = int(step.method_info.get("n_groups", 0))
    hotspot_residues = batch.metrics.get("pseudo_hotspot:target_hotspot_residues", "")
    summary["n_selected_residues"] = 0 if not hotspot_residues else len(hotspot_residues.split(","))
    summary["analysis_dir"] = str(step.analysis_dir or hotspot_dir)
    summary["status"] = "ok"
    return summary


def write_summary(summary_csv: Path, rows: list[dict[str, object]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv, index=False)


def main() -> int:
    args = parse_args()
    rows = []
    ids = target_ids(args.outputs_root, args.targets)
    print(f"Processing {len(ids)} targets from {args.outputs_root}", flush=True)
    for index, target_id in enumerate(ids, start=1):
        print(f"[{index}/{len(ids)}] {target_id}", flush=True)
        try:
            row = process_target(args, target_id)
        except Exception as exc:
            cmd.delete("all")
            row = {
                "target": target_id,
                "status": "failed",
                "n_designs": 0,
                "n_hotspot_rows": 0,
                "n_selected_residues": 0,
                "n_groups": 0,
                "target_chains": "",
                "target_pdb": "",
                "analysis_dir": str(args.outputs_root / target_id / "hotspot"),
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
            if args.fail_fast:
                rows.append(row)
                write_summary(args.summary_csv, rows)
                raise
        rows.append(row)
        print(
            f"  {row['status']} designs={row['n_designs']} "
            f"hotspot_rows={row['n_hotspot_rows']} "
            f"selected={row['n_selected_residues']} groups={row['n_groups']}",
            flush=True,
        )
        write_summary(args.summary_csv, rows)
    write_summary(args.summary_csv, rows)
    failed = [row for row in rows if row["status"] == "failed"]
    print(f"Done: {len(rows) - len(failed)} ok/skipped, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
