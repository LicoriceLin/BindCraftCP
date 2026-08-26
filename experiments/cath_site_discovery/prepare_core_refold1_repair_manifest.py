#!/usr/bin/env python
"""Prepare a rerun manifest for incomplete CATH core refold-1 targets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


BASE = Path("input/cath_testset_boltz/refold-1")
SOURCE_MANIFEST = BASE / "manifests" / "core_refold1_manifest.csv"
REPAIR_MANIFEST = BASE / "manifests" / "core_refold1_repair_manifest.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_ids_from_binder_csv(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [row["sample_id"] for row in csv.DictReader(handle)]


def is_complete(output_dir: Path, sample_ids: list[str]) -> bool:
    if not (
        (output_dir / "summary.json").exists()
        and (output_dir / "fold_scores.csv").exists()
        and (output_dir / "aggregate_metrics_analyze.csv").exists()
    ):
        return False
    try:
        fold_rows = read_csv(output_dir / "fold_scores.csv")
        analysis_rows = read_csv(output_dir / "aggregate_metrics_analyze.csv")
    except Exception:
        return False
    fold_ids = {row.get("sample_id") for row in fold_rows}
    analysis_ids = {row.get("id") for row in analysis_rows}
    return set(sample_ids) <= fold_ids and set(sample_ids) <= analysis_ids


def has_all_folds(output_dir: Path, sample_ids: list[str]) -> bool:
    return all(
        (output_dir / "fold_out_npz" / f"{sample_id}.npz").exists()
        and (output_dir / "refold_cif" / f"{sample_id}.cif").exists()
        for sample_id in sample_ids
    )


def main() -> int:
    rows = read_csv(SOURCE_MANIFEST)
    repair_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_ids = sample_ids_from_binder_csv(Path(row["binder_csv"]))
        output_dir = Path(row["output_dir"])
        if is_complete(output_dir, sample_ids):
            continue
        repair_row = dict(row)
        repair_row["repair_shard"] = len(repair_rows) % 4
        repair_row["skip_folding"] = str(has_all_folds(output_dir, sample_ids)).lower()
        repair_row["n_samples"] = len(sample_ids)
        repair_rows.append(repair_row)

    fieldnames = [
        "target",
        "shard",
        "repair_shard",
        "target_structure",
        "target_chains",
        "binding_res_index",
        "cyclic",
        "binder_csv",
        "output_dir",
        "n_passed_designs",
        "n_samples",
        "skip_folding",
    ]
    write_csv(REPAIR_MANIFEST, repair_rows, fieldnames)
    print(f"source_targets={len(rows)}")
    print(f"repair_targets={len(repair_rows)}")
    print(f"skip_folding_targets={sum(row['skip_folding'] == 'true' for row in repair_rows)}")
    print(f"rerun_folding_targets={sum(row['skip_folding'] != 'true' for row in repair_rows)}")
    print(f"manifest={REPAIR_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
