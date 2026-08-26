#!/usr/bin/env python
"""Prepare CATH full-target BoltzGen design inputs with hotspot binding constraints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("input/cath_testset_boltz/outputs"),
    )
    parser.add_argument(
        "--design-root",
        type=Path,
        default=Path("input/cath_testset_boltz/motif_boltzgen_design"),
    )
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_ca_residues(pdb: Path) -> list[tuple[str, str]]:
    residues = []
    seen = set()
    with pdb.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            residue = (line[21].strip(), line[22:27].strip())
            if residue not in seen:
                seen.add(residue)
                residues.append(residue)
    return residues


def read_hotspot_residues(hotspot_table: Path, freq_threshold: float = 0.5) -> set[tuple[str, str]]:
    residues = set()
    with hotspot_table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["freq"]) > freq_threshold:
                residues.add((str(row["chain"]), str(row["resi"])))
    return residues


def map_to_ordinals(
    residues: list[tuple[str, str]],
    full_residues: list[tuple[str, str]],
    context: str,
) -> list[tuple[str, str]]:
    ordinal_map = {
        residue: str(index)
        for index, residue in enumerate(full_residues, start=1)
    }
    mapped = []
    missing = []
    for residue in residues:
        ordinal = ordinal_map.get(residue)
        if ordinal is None:
            missing.append(f"{residue[0]}:{residue[1]}")
        else:
            mapped.append((residue[0], ordinal))
    if missing:
        preview = ",".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise ValueError(f"Could not map {context} residues: {preview}{suffix}")
    return mapped


def residue_ids(residues: list[tuple[str, str]]) -> str:
    return ",".join(resi for _, resi in residues)


def write_yaml(
    yaml_path: Path,
    full_target_pdb: Path,
    chain_id: str,
    binding_residues: list[tuple[str, str]],
) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                "entities:",
                "- file:",
                f'    path: "{full_target_pdb}"',
                "    include:",
                "    - chain:",
                f"        id: {chain_id}",
                "    structure_groups: all",
                "    binding_types:",
                "    - chain:",
                f"        id: {chain_id}",
                f'        binding: "{residue_ids(binding_residues)}"',
                "- protein:",
                "    id: Z",
                "    sequence: 40..90",
                "    cyclic: false",
                "",
            ]
        )
    )


def main() -> int:
    args = parse_args()
    summary_csv = args.outputs_root / "hotspot_run_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    inputs_dir = args.design_root / "inputs"
    outputs_dir = args.design_root / "outputs"
    manifest_dir = args.design_root / "manifests"
    logs_dir = args.design_root / "logs"
    for directory in (inputs_dir, outputs_dir, manifest_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    with summary_csv.open(newline="") as handle:
        for summary in csv.DictReader(handle):
            if summary["status"] != "ok":
                continue

            target = summary["target"]
            motif_dir = args.outputs_root / target / "hotspot" / f"{target}_motif"
            full_target_pdb = motif_dir / f"{target}-full.pdb"
            if not full_target_pdb.exists():
                full_target_pdb = Path(summary["target_pdb"])
            full_target_for_yaml = full_target_pdb
            if not full_target_for_yaml.is_absolute():
                full_target_for_yaml = Path.cwd() / full_target_for_yaml
            full_target_for_yaml = full_target_for_yaml.absolute()

            full_residues = read_ca_residues(full_target_for_yaml)
            chains = sorted({chain for chain, _ in full_residues})
            if len(chains) != 1:
                raise ValueError(f"{full_target_for_yaml} contains chains {chains}")

            hotspot_residues = read_hotspot_residues(
                args.outputs_root / target / "hotspot" / f"{target}_hotspot_table.csv"
            )
            binding_author_residues = [
                residue for residue in full_residues if residue in hotspot_residues
            ]
            if not binding_author_residues:
                raise ValueError(f"{target} has no hotspot binding residues")
            binding_residues = map_to_ordinals(
                binding_author_residues,
                full_residues,
                f"{target}-full binding",
            )

            yaml_path = inputs_dir / f"{target}-full.yaml"
            if args.overwrite or not yaml_path.exists():
                write_yaml(
                    yaml_path=yaml_path,
                    full_target_pdb=full_target_for_yaml,
                    chain_id=chains[0],
                    binding_residues=binding_residues,
                )

            rows.append(
                {
                    "target": target,
                    "motif_size": "full",
                    "shard": len(rows) % args.shard_count,
                    "input_yaml": str(yaml_path),
                    "output_dir": str(outputs_dir / target / "full"),
                    "full_target_pdb": str(full_target_pdb),
                    "motif_pdb": str(full_target_pdb),
                    "n_res_index": len(full_residues),
                    "n_binding": len(binding_residues),
                }
            )

    manifest = manifest_dir / "motif_full_boltzgen_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote_inputs={len(rows)}")
    print(f"manifest={manifest}")
    print(f"logs_dir={logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
