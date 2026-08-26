#!/usr/bin/env python
"""Summarize CATH full and motif refold success rates by target and condition."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


CONDITION_ORDER = ["50", "100", "150", "200", "250", "full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("input/cath_testset_boltz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/cath_testset_boltz/analysis"),
    )
    parser.add_argument("--rmsd-column", default="bb_rmsd_design")
    parser.add_argument("--iptm-column", default="iptm")
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--iptm-threshold", type=float, default=0.8)
    parser.add_argument(
        "--full-source",
        choices=("auto", "core", "constrained", "none"),
        default="auto",
        help=(
            "Source for the full row. auto uses constrained full when available, "
            "otherwise the original core/full refold."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def count_success(
    aggregate_csv: Path,
    rmsd_column: str,
    iptm_column: str,
    rmsd_threshold: float,
    iptm_threshold: float,
) -> tuple[int, int]:
    rows = read_csv(aggregate_csv)
    success = 0
    for row in rows:
        rmsd = numeric(row.get(rmsd_column))
        iptm = numeric(row.get(iptm_column))
        if math.isfinite(rmsd) and math.isfinite(iptm):
            if rmsd < rmsd_threshold and iptm > iptm_threshold:
                success += 1
    return len(rows), success


def add_full_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    constrained_full_manifest = (
        args.base_dir
        / "motif_boltzgen_design"
        / "manifests"
        / "motif_full_boltzgen_manifest.csv"
    )
    if args.full_source == "none":
        return
    if args.full_source == "constrained":
        return
    if args.full_source == "auto" and constrained_full_manifest.exists():
        return

    manifest = args.base_dir / "refold-1" / "manifests" / "core_refold1_manifest.csv"
    for row in read_csv(manifest):
        aggregate = Path(row["output_dir"]) / "aggregate_metrics_analyze.csv"
        refolded, success = count_success(
            aggregate,
            args.rmsd_column,
            args.iptm_column,
            args.rmsd_threshold,
            args.iptm_threshold,
        )
        total_passed = int(row["n_passed_designs"])
        complete = aggregate.exists() and aggregate.stat().st_size > 0 and refolded >= total_passed
        rows.append(
            {
                "target": row["target"],
                "condition": "full",
                "total_passed": total_passed,
                "refolded": refolded,
                "success": success,
                "success_rate": success / total_passed if total_passed else 0.0,
                "status": "complete" if complete else "incomplete",
                "aggregate_csv": str(aggregate),
            }
        )


def add_motif_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    manifest = (
        args.base_dir
        / "motif_boltzgen_design"
        / "refold-1"
        / "manifests"
        / "motif_refold1_manifest.csv"
    )
    for row in read_csv(manifest):
        aggregate = Path(row["output_dir"]) / "aggregate_metrics_analyze.csv"
        refolded, success = count_success(
            aggregate,
            args.rmsd_column,
            args.iptm_column,
            args.rmsd_threshold,
            args.iptm_threshold,
        )
        total_passed = int(row["n_passed_designs"])
        complete = aggregate.exists() and aggregate.stat().st_size > 0 and refolded >= total_passed
        rows.append(
            {
                "target": row["target"],
                "condition": str(row["motif_size"]),
                "total_passed": total_passed,
                "refolded": refolded,
                "success": success,
                "success_rate": success / total_passed if total_passed else 0.0,
                "status": "complete" if complete else "incomplete",
                "aggregate_csv": str(aggregate),
            }
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(args: argparse.Namespace, rows: list[dict[str, Any]]) -> Path:
    targets = [
        row["system"]
        for row in read_csv(args.base_dir / "core_set_manifest.csv")
    ]
    by_key = {(row["target"], row["condition"]): row for row in rows}
    matrix_rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        matrix_row: dict[str, Any] = {"condition": condition}
        for target in targets:
            record = by_key.get((target, condition))
            if record is None or record["status"] != "complete":
                matrix_row[target] = ""
            else:
                matrix_row[target] = f"{float(record['success_rate']):.6f}"
        matrix_rows.append(matrix_row)

    matrix_csv = args.output_dir / "refold_success_rate_matrix.csv"
    write_csv(matrix_csv, matrix_rows, ["condition", *targets])
    return matrix_csv


def write_heatmap(args: argparse.Namespace, matrix_csv: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    matrix = read_csv(matrix_csv)
    targets = [field for field in matrix[0] if field != "condition"] if matrix else []
    values = []
    for row in matrix:
        values.append([
            float(row[target]) if row[target] != "" else math.nan
            for target in targets
        ])
    data = np.array(values, dtype=float)

    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad(color="#c7c7c7")
    width = max(14, len(targets) * 0.18)
    fig, ax = plt.subplots(figsize=(width, 2.7), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([f"{row['condition']} aa" if row["condition"] != "full" else "full" for row in matrix])
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=75, ha="right", fontsize=5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="success rate")

    heatmap = args.output_dir / "refold_success_rate_heatmap.png"
    fig.savefig(heatmap, dpi=300)
    plt.close(fig)
    return heatmap


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    add_motif_rows(args, rows)
    add_full_rows(args, rows)
    rows.sort(
        key=lambda row: (
            CONDITION_ORDER.index(row["condition"])
            if row["condition"] in CONDITION_ORDER
            else len(CONDITION_ORDER),
            row["target"],
        )
    )

    long_csv = args.output_dir / "refold_success_by_condition.csv"
    write_csv(
        long_csv,
        rows,
        [
            "target",
            "condition",
            "total_passed",
            "refolded",
            "success",
            "success_rate",
            "status",
            "aggregate_csv",
        ],
    )
    matrix_csv = write_matrix(args, rows)
    heatmap = write_heatmap(args, matrix_csv)

    complete_rows = sum(1 for row in rows if row["status"] == "complete")
    print(f"rows={len(rows)} complete_rows={complete_rows}")
    print(f"long_csv={long_csv}")
    print(f"matrix_csv={matrix_csv}")
    if heatmap:
        print(f"heatmap={heatmap}")
    else:
        print("heatmap=skipped_matplotlib_not_available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
