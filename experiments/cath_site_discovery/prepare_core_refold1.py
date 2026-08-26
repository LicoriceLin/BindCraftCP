#!/usr/bin/env python
"""Prepare refold-1 validation inputs for CATH core BoltzGen designs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("input/cath_testset_boltz"),
    )
    parser.add_argument(
        "--refold-root",
        type=Path,
        default=Path("input/cath_testset_boltz/refold-1"),
    )
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_core_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_target_spec(yaml_path: Path) -> tuple[Path, list[str], bool]:
    data = yaml.safe_load(yaml_path.read_text())
    entities = data.get("entities") or []
    file_entity = next((entry["file"] for entry in entities if "file" in entry), None)
    protein_entity = next((entry["protein"] for entry in entities if "protein" in entry), None)
    if file_entity is None or protein_entity is None:
        raise ValueError(f"{yaml_path} must contain file and protein entities")

    target_path = Path(file_entity["path"])
    include = file_entity.get("include") or []
    chains = []
    for item in include:
        chain = item.get("chain")
        if chain and chain.get("id"):
            chains.append(str(chain["id"]))
    if not chains:
        raise ValueError(f"{yaml_path} does not define an included target chain")

    cyclic = bool(protein_entity.get("cyclic", False))
    return target_path, chains, cyclic


def read_chain_ordinals(pdb_path: Path, target_chains: list[str]) -> dict[str, dict[str, int]]:
    ordinals: dict[str, dict[str, int]] = {chain: {} for chain in target_chains}
    seen: dict[str, set[str]] = {chain: set() for chain in target_chains}
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21].strip()
            if chain not in ordinals:
                continue
            resi = line[22:27].strip()
            if resi in seen[chain]:
                continue
            seen[chain].add(resi)
            ordinals[chain][resi] = len(ordinals[chain]) + 1
    return ordinals


def read_binding_res_index(
    hotspot_table: Path,
    target_structure: Path,
    target_chains: list[str],
) -> list[str]:
    ordinals = read_chain_ordinals(target_structure, target_chains)
    by_chain: dict[str, list[str]] = {chain: [] for chain in target_chains}
    seen: set[tuple[str, str]] = set()
    missing: list[str] = []
    with hotspot_table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            chain = str(row["chain"])
            resi = str(row["resi"])
            if chain not in by_chain or (chain, resi) in seen:
                continue
            ordinal = ordinals.get(chain, {}).get(resi)
            if ordinal is None:
                missing.append(f"{chain}:{resi}")
                continue
            by_chain[chain].append(str(ordinal))
            seen.add((chain, resi))
    if missing:
        raise ValueError(
            f"{hotspot_table} residues are absent from {target_structure}: "
            + ",".join(missing[:20])
        )

    values = []
    for chain in target_chains:
        if by_chain[chain]:
            values.append(f"{chain}:{','.join(by_chain[chain])}")
    return values


def collect_passed_designs(metrics_csv: Path) -> list[dict[str, str]]:
    passed = []
    with metrics_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
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
    return passed


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")

    manifest_rows = read_core_rows(args.base_dir / "core_set_manifest.csv")
    args.refold_root.mkdir(parents=True, exist_ok=True)
    for subdir in ("binder_inputs", "outputs", "logs", "manifests"):
        (args.refold_root / subdir).mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    all_passed_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        target = row["system"]
        target_yaml = args.base_dir / "inputs" / f"{target}.yaml"
        metrics_csv = (
            args.base_dir
            / "outputs"
            / target
            / "final_ranked_designs"
            / "all_designs_metrics.csv"
        )
        hotspot_table = (
            args.base_dir
            / "outputs"
            / target
            / "hotspot"
            / f"{target}_hotspot_table.csv"
        )
        target_structure, target_chains, cyclic = load_target_spec(target_yaml)
        passed = collect_passed_designs(metrics_csv)
        binding_res_index = read_binding_res_index(
            hotspot_table,
            target_structure,
            target_chains,
        )

        status_rows.append(
            {
                "target": target,
                "done_experiment_groups": row["done_experiment_groups"],
                "target_chains": ";".join(target_chains),
                "n_passed_designs": len(passed),
                "n_binding_residue_groups": len(binding_res_index),
            }
        )

        for passed_row in passed:
            all_passed_rows.append({"target": target, **passed_row})

        if not passed:
            continue

        binder_csv = args.refold_root / "binder_inputs" / f"{target}.csv"
        if args.overwrite or not binder_csv.exists():
            write_csv(binder_csv, passed, ["sample_id", "binder_sequence"])

        output_dir = args.refold_root / "outputs" / target
        run_rows.append(
            {
                "target": target,
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
        args.refold_root / "manifests" / "core_refold1_manifest.csv",
        run_rows,
        [
            "target",
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
        args.refold_root / "manifests" / "core_refold1_target_status.csv",
        status_rows,
        [
            "target",
            "done_experiment_groups",
            "target_chains",
            "n_passed_designs",
            "n_binding_residue_groups",
        ],
    )
    write_csv(
        args.refold_root / "manifests" / "core_refold1_passed_designs.csv",
        all_passed_rows,
        ["target", "sample_id", "binder_sequence"],
    )

    print(f"core_targets={len(manifest_rows)}")
    print(f"targets_with_passed_designs={len(run_rows)}")
    print(f"passed_designs={len(all_passed_rows)}")
    print(f"manifest={args.refold_root / 'manifests' / 'core_refold1_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
