#!/usr/bin/env python
"""Summarize CATH BoltzGen pass_filters success rates by target and condition."""

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
    parser.add_argument(
        "--full-source",
        choices=("core", "constrained", "none"),
        default="core",
        help="Use original core/full outputs or constrained motif full outputs for the full row.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def count_passes(metrics_csv: Path) -> tuple[int, int]:
    rows = read_csv(metrics_csv)
    return len(rows), sum(1 for row in rows if truthy(row.get("pass_filters")))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_motif_rows(base_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest = (
        base_dir
        / "motif_boltzgen_design"
        / "manifests"
        / "motif_boltzgen_manifest.csv"
    )
    for row in read_csv(manifest):
        metrics_csv = (
            Path(row["output_dir"])
            / "final_ranked_designs"
            / "all_designs_metrics.csv"
        )
        total, passed = count_passes(metrics_csv)
        rows.append(
            {
                "target": row["target"],
                "condition": str(row["motif_size"]),
                "n_designs": total,
                "n_success": passed,
                "success_rate": passed / total if total else math.nan,
                "status": "complete" if total else "missing_metrics",
                "metrics_csv": str(metrics_csv),
            }
        )


def add_full_rows(base_dir: Path, full_source: str, rows: list[dict[str, Any]]) -> None:
    if full_source == "none":
        return

    for row in read_csv(base_dir / "core_set_manifest.csv"):
        target = row["system"]
        if full_source == "core":
            metrics_csv = (
                base_dir
                / "outputs"
                / target
                / "final_ranked_designs"
                / "all_designs_metrics.csv"
            )
        else:
            metrics_csv = (
                base_dir
                / "motif_boltzgen_design"
                / "outputs"
                / target
                / "full"
                / "final_ranked_designs"
                / "all_designs_metrics.csv"
            )
        total, passed = count_passes(metrics_csv)
        rows.append(
            {
                "target": target,
                "condition": "full",
                "n_designs": total,
                "n_success": passed,
                "success_rate": passed / total if total else math.nan,
                "status": "complete" if total else "missing_metrics",
                "metrics_csv": str(metrics_csv),
            }
        )


def write_matrix(base_dir: Path, output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    targets = [row["system"] for row in read_csv(base_dir / "core_set_manifest.csv")]
    by_key = {(row["target"], row["condition"]): row for row in rows}
    matrix_rows = []
    for condition in CONDITION_ORDER:
        matrix_row: dict[str, Any] = {"condition": condition}
        for target in targets:
            record = by_key.get((target, condition))
            if record is None or record["status"] != "complete":
                matrix_row[target] = ""
            else:
                matrix_row[target] = f"{float(record['success_rate']):.6f}"
        matrix_rows.append(matrix_row)

    matrix_csv = output_dir / "boltzgen_success_rate_matrix.csv"
    write_csv(matrix_csv, matrix_rows, ["condition", *targets])
    return matrix_csv


def write_heatmap(matrix_csv: Path, output_dir: Path) -> Path | None:
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
    labels = [
        f"{row['condition']} aa" if row["condition"] != "full" else "full"
        for row in matrix
    ]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=75, ha="right", fontsize=5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="pass_filters rate")

    heatmap = output_dir / "boltzgen_success_rate_heatmap.png"
    fig.savefig(heatmap, dpi=300)
    plt.close(fig)
    return heatmap


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    add_motif_rows(args.base_dir, rows)
    add_full_rows(args.base_dir, args.full_source, rows)
    rows.sort(
        key=lambda row: (
            CONDITION_ORDER.index(row["condition"]),
            row["target"],
        )
    )

    long_csv = args.output_dir / "boltzgen_success_by_condition.csv"
    write_csv(
        long_csv,
        rows,
        [
            "target",
            "condition",
            "n_designs",
            "n_success",
            "success_rate",
            "status",
            "metrics_csv",
        ],
    )
    matrix_csv = write_matrix(args.base_dir, args.output_dir, rows)
    heatmap = write_heatmap(matrix_csv, args.output_dir)

    complete = sum(1 for row in rows if row["status"] == "complete")
    print(f"rows={len(rows)} complete_rows={complete}")
    print(f"long_csv={long_csv}")
    print(f"matrix_csv={matrix_csv}")
    if heatmap:
        print(f"heatmap={heatmap}")
    else:
        print("heatmap=skipped_matplotlib_not_available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
