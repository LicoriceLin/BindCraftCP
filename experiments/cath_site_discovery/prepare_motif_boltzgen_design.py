#!/usr/bin/env python
"""Prepare BoltzGen design inputs from extracted CATH motif PDBs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


MOTIF_SIZES = (50, 100, 150, 200, 250)


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
    parser.add_argument(
        "--submit-template",
        type=Path,
        help="Optional site-specific Slurm script copied beside the manifest.",
    )
    return parser.parse_args()


def residue_sort_key(residue: tuple[str, str]) -> tuple:
    chain, resi = residue
    if resi.startswith("-") and resi[1:].isdigit():
        return chain, 0, int(resi), ""
    if resi[:-1].isdigit() and resi[-1].isalpha():
        return chain, 0, int(resi[:-1]), resi[-1]
    if resi.isdigit():
        return chain, 0, int(resi), ""
    return chain, 1, resi, ""


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


def read_chain_ordinals(pdb: Path, chains: set[str]) -> dict[str, dict[str, int]]:
    """Map PDB author residue IDs to the 1-based ordinal indices BoltzGen expects."""
    residues_by_chain: dict[str, list[str]] = {chain: [] for chain in chains}
    seen = set()
    with pdb.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21].strip()
            if chain not in residues_by_chain:
                continue
            resi = line[22:27].strip()
            residue = (chain, resi)
            if residue not in seen:
                seen.add(residue)
                residues_by_chain[chain].append(resi)

    missing = [chain for chain, residues in residues_by_chain.items() if not residues]
    if missing:
        raise ValueError(f"{pdb} has no CA residues for chain(s): {','.join(missing)}")

    return {
        chain: {resi: idx for idx, resi in enumerate(residues, start=1)}
        for chain, residues in residues_by_chain.items()
    }


def map_to_ordinals(
    residues: list[tuple[str, str]],
    ordinal_maps: dict[str, dict[str, int]],
    context: str,
    target_pdb: Path,
) -> list[tuple[str, str]]:
    mapped = []
    missing = []
    for chain, resi in residues:
        ordinal = ordinal_maps.get(chain, {}).get(resi)
        if ordinal is None:
            missing.append(f"{chain}:{resi}")
        else:
            mapped.append((chain, str(ordinal)))
    if missing:
        preview = ",".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise ValueError(
            f"Could not map {context} residues onto {target_pdb}: {preview}{suffix}"
        )
    return mapped


def read_hotspot_residues(hotspot_table: Path, freq_threshold: float = 0.5) -> set[tuple[str, str]]:
    residues = set()
    with hotspot_table.open() as handle:
        for row in csv.DictReader(handle):
            if float(row["freq"]) > freq_threshold:
                residues.add((str(row["chain"]), str(row["resi"])))
    return residues


def residue_ids(residues: list[tuple[str, str]]) -> str:
    return ",".join(resi for _, resi in residues)


def write_yaml(
    yaml_path: Path,
    full_target_pdb: Path,
    chain_id: str,
    motif_residues: list[tuple[str, str]],
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
                f'        res_index: "{residue_ids(motif_residues)}"',
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
    scripts_dir = args.design_root / "scripts"
    logs_dir = args.design_root / "logs"
    for directory in (inputs_dir, outputs_dir, manifest_dir, scripts_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    with summary_csv.open() as handle:
        for summary in csv.DictReader(handle):
            if summary["status"] != "ok":
                continue
            target = summary["target"]
            motif_dir = args.outputs_root / target / "hotspot" / f"{target}_motif"
            full_target_pdb = motif_dir / f"{target}-full.pdb"
            if not full_target_pdb.exists():
                full_target_pdb = Path(summary["target_pdb"])
            hotspot_residues = read_hotspot_residues(
                args.outputs_root / target / "hotspot" / f"{target}_hotspot_table.csv"
            )

            for motif_size in MOTIF_SIZES:
                motif_pdb = motif_dir / f"{target}-{motif_size}.pdb"
                motif_residues = read_ca_residues(motif_pdb)
                motif_set = set(motif_residues)
                binding_residues = [
                    residue for residue in motif_residues if residue in hotspot_residues
                ]
                if len(motif_residues) != motif_size:
                    raise ValueError(f"{motif_pdb} has {len(motif_residues)} residues")
                if not binding_residues:
                    raise ValueError(f"{motif_pdb} has no hotspot binding residues")

                chains = sorted({chain for chain, _ in motif_residues})
                if len(chains) != 1:
                    raise ValueError(f"{motif_pdb} contains chains {chains}")
                if not set(binding_residues).issubset(motif_set):
                    raise ValueError(f"{motif_pdb} binding residues not in motif")

                full_target_for_yaml = full_target_pdb
                if not full_target_for_yaml.is_absolute():
                    full_target_for_yaml = Path.cwd() / full_target_for_yaml
                full_target_for_yaml = full_target_for_yaml.absolute()
                ordinal_maps = read_chain_ordinals(full_target_for_yaml, set(chains))
                motif_residues_for_yaml = map_to_ordinals(
                    motif_residues,
                    ordinal_maps,
                    f"{target}-{motif_size} motif",
                    full_target_for_yaml,
                )
                binding_residues_for_yaml = map_to_ordinals(
                    binding_residues,
                    ordinal_maps,
                    f"{target}-{motif_size} binding",
                    full_target_for_yaml,
                )

                yaml_name = f"{target}-{motif_size}.yaml"
                yaml_path = inputs_dir / yaml_name
                if args.overwrite or not yaml_path.exists():
                    write_yaml(
                        yaml_path=yaml_path,
                        full_target_pdb=full_target_for_yaml,
                        chain_id=chains[0],
                        motif_residues=motif_residues_for_yaml,
                        binding_residues=binding_residues_for_yaml,
                    )

                output_dir = outputs_dir / target / str(motif_size)
                rows.append(
                    {
                        "target": target,
                        "motif_size": motif_size,
                        "shard": len(rows) % args.shard_count,
                        "input_yaml": str(yaml_path),
                        "output_dir": str(output_dir),
                        "full_target_pdb": str(full_target_pdb),
                        "motif_pdb": str(motif_pdb),
                        "n_res_index": len(motif_residues),
                        "n_binding": len(binding_residues),
                    }
                )

    manifest = manifest_dir / "motif_boltzgen_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    array_script = None
    if args.submit_template is not None:
        array_script = scripts_dir / args.submit_template.name
        shutil.copyfile(args.submit_template, array_script)
        array_script.chmod(0o755)

    print(f"wrote_inputs={len(rows)}")
    print(f"manifest={manifest}")
    if array_script is not None:
        print(f"array_script={array_script}")
    print(f"logs_dir={logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
