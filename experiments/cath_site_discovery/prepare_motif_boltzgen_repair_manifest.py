#!/usr/bin/env python
"""Prepare a repair manifest for missing CATH motif BoltzGen design outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "input/cath_testset_boltz/motif_boltzgen_design/manifests/"
            "motif_boltzgen_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "input/cath_testset_boltz/motif_boltzgen_design/manifests/"
            "motif_boltzgen_repair_manifest.csv"
        ),
    )
    parser.add_argument("--shard-count", type=int, default=4)
    return parser.parse_args()


def has_complete_metrics(output_dir: Path) -> bool:
    metrics = output_dir / "final_ranked_designs" / "all_designs_metrics.csv"
    return metrics.exists() and metrics.stat().st_size > 0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")

    with args.manifest.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    repair_rows: list[dict[str, Any]] = []
    for row in source_rows:
        if has_complete_metrics(Path(row["output_dir"])):
            continue
        repair_row = dict(row)
        repair_row["repair_shard"] = len(repair_rows) % args.shard_count
        repair_rows.append(repair_row)

    fieldnames = list(source_rows[0]) + ["repair_shard"] if source_rows else []
    write_csv(args.output, repair_rows, fieldnames)

    print(f"source_rows={len(source_rows)}")
    print(f"repair_rows={len(repair_rows)}")
    print(f"manifest={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
