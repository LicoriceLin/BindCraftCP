#!/usr/bin/env python
"""Prepare refold-1 validation inputs for CATH motif BoltzGen designs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design-root",
        type=Path,
        default=Path("input/cath_testset_boltz/motif_boltzgen_design"),
    )
    parser.add_argument(
        "--refold-root",
        type=Path,
        default=Path("input/cath_testset_boltz/motif_boltzgen_design/refold-1"),
    )
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def collect_passed_designs(metrics_csv: Path) -> tuple[int, list[dict[str, str]]]:
    total = 0
    passed = []
    with metrics_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            if not truthy(row.get("pass_filters")):
                continue
            sample_id = (row.get("id") or "").strip()
            sequence = (
                row.get("designed_sequence")
                or row.get("designed_chain_sequence")
                or row.get("sequence")
                or ""
            ).strip()
            if not sample_id or not sequence:
                raise ValueError(f"{metrics_csv} has a passed row without id/sequence")
            passed.append({"sample_id": sample_id, "binder_sequence": sequence})
    return total, passed


def load_target_spec(yaml_path: Path) -> tuple[Path, list[str], list[str], bool]:
    data = yaml.safe_load(yaml_path.read_text())
    entities = data.get("entities") or []
    file_entity = next((entry["file"] for entry in entities if "file" in entry), None)
    protein_entity = next((entry["protein"] for entry in entities if "protein" in entry), None)
    if file_entity is None or protein_entity is None:
        raise ValueError(f"{yaml_path} must contain file and protein entities")

    target_path = Path(file_entity["path"])
    if not target_path.is_absolute():
        target_path = (yaml_path.parent / target_path).resolve()

    chains = []
    include = file_entity.get("include") or []
    for item in include:
        chain = item.get("chain")
        if chain and chain.get("id"):
            chains.append(str(chain["id"]))
    if not chains:
        raise ValueError(f"{yaml_path} does not define an included target chain")

    binding_res_index = []
    for item in file_entity.get("binding_types") or []:
        chain = item.get("chain")
        if not chain or not chain.get("id") or not chain.get("binding"):
            continue
        binding_res_index.append(f"{chain['id']}:{chain['binding']}")
    if not binding_res_index:
        raise ValueError(f"{yaml_path} does not define binding residues")

    cyclic = bool(protein_entity.get("cyclic", False))
    return target_path, chains, binding_res_index, cyclic


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

    manifest_rows = []
    manifest_paths = [
        args.design_root / "manifests" / "motif_boltzgen_manifest.csv",
        args.design_root / "manifests" / "motif_full_boltzgen_manifest.csv",
    ]
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        with manifest_path.open(newline="") as handle:
            manifest_rows.extend(csv.DictReader(handle))

    for subdir in ("binder_inputs", "outputs", "logs", "manifests"):
        (args.refold_root / subdir).mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    all_passed_rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        target = row["target"]
        motif_size = str(row["motif_size"])
        condition = f"{target}-{motif_size}"
        metrics_csv = (
            Path(row["output_dir"])
            / "final_ranked_designs"
            / "all_designs_metrics.csv"
        )

        target_structure, target_chains, binding_res_index, cyclic = load_target_spec(
            Path(row["input_yaml"])
        )

        if metrics_csv.exists() and metrics_csv.stat().st_size > 0:
            design_status = "has_metrics"
            n_designs, passed = collect_passed_designs(metrics_csv)
        else:
            design_status = "missing_metrics"
            n_designs = 0
            passed = []

        status_rows.append(
            {
                "target": target,
                "motif_size": motif_size,
                "design_status": design_status,
                "target_chains": ";".join(target_chains),
                "n_designs": n_designs,
                "n_passed_designs": len(passed),
                "n_binding_residue_groups": len(binding_res_index),
            }
        )

        for passed_row in passed:
            all_passed_rows.append(
                {"target": target, "motif_size": motif_size, **passed_row}
            )

        if not passed:
            continue

        binder_csv = args.refold_root / "binder_inputs" / target / f"{motif_size}.csv"
        if args.overwrite or not binder_csv.exists():
            write_csv(binder_csv, passed, ["sample_id", "binder_sequence"])

        output_dir = args.refold_root / "outputs" / target / motif_size
        run_rows.append(
            {
                "target": target,
                "motif_size": motif_size,
                "condition": condition,
                "shard": len(run_rows) % args.shard_count,
                "target_structure": str(target_structure),
                "target_chains": ";".join(target_chains),
                "binding_res_index": ";".join(binding_res_index),
                "cyclic": str(cyclic).lower(),
                "binder_csv": str(binder_csv),
                "output_dir": str(output_dir),
                "n_passed_designs": len(passed),
            }
        )

    write_csv(
        args.refold_root / "manifests" / "motif_refold1_manifest.csv",
        run_rows,
        [
            "target",
            "motif_size",
            "condition",
            "shard",
            "target_structure",
            "target_chains",
            "binding_res_index",
            "cyclic",
            "binder_csv",
            "output_dir",
            "n_passed_designs",
        ],
    )
    write_csv(
        args.refold_root / "manifests" / "motif_refold1_condition_status.csv",
        status_rows,
        [
            "target",
            "motif_size",
            "design_status",
            "target_chains",
            "n_designs",
            "n_passed_designs",
            "n_binding_residue_groups",
        ],
    )
    write_csv(
        args.refold_root / "manifests" / "motif_refold1_passed_designs.csv",
        all_passed_rows,
        ["target", "motif_size", "sample_id", "binder_sequence"],
    )

    print(f"motif_conditions={len(manifest_rows)}")
    print(f"conditions_with_passed_designs={len(run_rows)}")
    print(f"passed_designs={len(all_passed_rows)}")
    print(f"manifest={args.refold_root / 'manifests' / 'motif_refold1_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
