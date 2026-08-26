#!/usr/bin/env python
"""Postprocess hsClpP scaffold-full-1500 design contacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, Selection
from scipy.spatial import cKDTree


DEFAULT_BASE_DIR = Path("input/hsClpP/scaffold-full-1500")


@dataclass(frozen=True, order=True)
class ResidueKey:
    chain: str
    seq: int
    icode: str = ""

    def token(self) -> str:
        return f"{self.chain}{self.seq}{self.icode}"

    def swapped_ab(self) -> "ResidueKey":
        if self.chain == "A":
            return ResidueKey("B", self.seq, self.icode)
        if self.chain == "B":
            return ResidueKey("A", self.seq, self.icode)
        raise ValueError(f"Cannot swap non-AB residue {self.token()}")


@dataclass(frozen=True)
class AtomRecord:
    coord: np.ndarray
    residue: ResidueKey


def residue_sort_key(residue: ResidueKey) -> tuple[int, str, int, str]:
    chain_order = {"A": 0, "B": 1, "C": 2}
    return (chain_order.get(residue.chain, 99), residue.chain, residue.seq, residue.icode)


def format_residues(residues: set[ResidueKey]) -> str:
    return ",".join(residue.token() for residue in sorted(residues, key=residue_sort_key))


def is_hydrogen(atom) -> bool:
    element = (getattr(atom, "element", "") or "").strip().upper()
    name = atom.get_name().strip().upper()
    return element in {"H", "D"} or name.startswith(("H", "D"))


def load_atoms_by_chain(cif_path: Path) -> dict[str, list[AtomRecord]]:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    model = structure[0]

    atoms_by_chain: dict[str, list[AtomRecord]] = {}
    for chain in model:
        chain_atoms: list[AtomRecord] = []
        for residue in Selection.unfold_entities(chain, "R"):
            if residue.id[0] != " ":
                continue
            seq = int(residue.id[1])
            icode = "" if residue.id[2] == " " else str(residue.id[2]).strip()
            key = ResidueKey(chain.id, seq, icode)
            for atom in residue.get_atoms():
                if is_hydrogen(atom):
                    continue
                chain_atoms.append(AtomRecord(np.asarray(atom.coord, dtype=float), key))
        atoms_by_chain[chain.id] = chain_atoms
    return atoms_by_chain


def require_chain_atoms(
    atoms_by_chain: dict[str, list[AtomRecord]],
    chains: tuple[str, ...],
    cif_path: Path,
) -> list[AtomRecord]:
    missing = [chain for chain in chains if chain not in atoms_by_chain]
    if missing:
        raise ValueError(f"{cif_path} is missing chain(s): {','.join(missing)}")
    atoms = [atom for chain in chains for atom in atoms_by_chain[chain]]
    if not atoms:
        raise ValueError(f"{cif_path} has no heavy atoms for chain(s): {','.join(chains)}")
    return atoms


def contacting_residues(
    cif_path: Path,
    query_chains: tuple[str, ...],
    target_chains: tuple[str, ...],
    cutoff: float,
) -> tuple[set[ResidueKey], set[ResidueKey]]:
    atoms_by_chain = load_atoms_by_chain(cif_path)
    query_atoms = require_chain_atoms(atoms_by_chain, query_chains, cif_path)
    target_atoms = require_chain_atoms(atoms_by_chain, target_chains, cif_path)

    query_tree = cKDTree(np.asarray([atom.coord for atom in query_atoms], dtype=float))
    target_tree = cKDTree(np.asarray([atom.coord for atom in target_atoms], dtype=float))

    query_contacts: set[ResidueKey] = set()
    target_contacts: set[ResidueKey] = set()
    for query_idx, close_target_indices in enumerate(query_tree.query_ball_tree(target_tree, cutoff)):
        if not close_target_indices:
            continue
        query_contacts.add(query_atoms[query_idx].residue)
        for target_idx in close_target_indices:
            target_contacts.add(target_atoms[target_idx].residue)

    return query_contacts, target_contacts


def index_design_cifs(design_dir: Path) -> dict[str, Path]:
    by_design_name: dict[str, Path] = {}
    for cif_path in sorted(design_dir.glob("*.cif")):
        by_design_name[cif_path.name] = cif_path
        if cif_path.name.startswith("rank") and "_" in cif_path.name:
            design_name = cif_path.name.split("_", 1)[1]
            by_design_name.setdefault(design_name, cif_path)
    return by_design_name


def resolve_design_cif(row: pd.Series, design_cifs: dict[str, Path]) -> Path:
    candidates = [
        str(row.get("file_name", "")),
        f"{row.get('id', '')}.cif",
    ]
    for candidate in candidates:
        if candidate and candidate in design_cifs:
            return design_cifs[candidate]
    raise FileNotFoundError(
        f"Could not find final CIF for id={row.get('id')} file_name={row.get('file_name')}"
    )


def add_pass_contacts(
    pass_df: pd.DataFrame,
    design_cifs: dict[str, Path],
    forbidden: set[ResidueKey],
    cutoff: float,
) -> pd.DataFrame:
    contacts: list[str] = []
    contact_forbidden: list[bool] = []

    for _, row in pass_df.iterrows():
        cif_path = resolve_design_cif(row, design_cifs)
        _, ab_contacts = contacting_residues(cif_path, ("C",), ("A", "B"), cutoff)
        contacts.append(format_residues(ab_contacts))
        contact_forbidden.append(bool(ab_contacts & forbidden))

    result = pass_df.copy()
    result["contact"] = contacts
    result["contact_forbidden"] = contact_forbidden
    return result


def run(base_dir: Path, cutoff: float) -> dict[str, int | Path]:
    scaffold_cif = base_dir / "scaffold-full.cif"
    metrics_csv = base_dir / "final_ranked_designs" / "all_designs_metrics.csv"
    design_dir = base_dir / "final_ranked_designs" / "final_1500_designs"
    ana_dir = base_dir / "ana"
    ana_dir.mkdir(parents=True, exist_ok=True)

    a_contacts, b_contacts = contacting_residues(scaffold_cif, ("A",), ("B",), cutoff)
    scaffold_contacts = a_contacts | b_contacts
    forbidden = {residue.swapped_ab() for residue in scaffold_contacts}

    contact_txt = ana_dir / "contact.txt"
    forbidden_txt = ana_dir / "forbidden.txt"
    contact_txt.write_text(format_residues(scaffold_contacts) + "\n")
    forbidden_txt.write_text(format_residues(forbidden) + "\n")

    metrics = pd.read_csv(metrics_csv)
    pass_mask = (
        pd.to_numeric(metrics["design_to_target_iptm"], errors="coerce").gt(0.35)
        & pd.to_numeric(metrics["filter_rmsd"], errors="coerce").lt(3.5)
    )
    pass_df = metrics.loc[pass_mask].copy()

    design_cifs = index_design_cifs(design_dir)
    pass_df = add_pass_contacts(pass_df, design_cifs, forbidden, cutoff)

    pass_csv = ana_dir / "pass.csv"
    pass_df.to_csv(pass_csv, index=False)

    false_count = int((~pass_df["contact_forbidden"]).sum())
    return {
        "contact_txt": contact_txt,
        "forbidden_txt": forbidden_txt,
        "pass_csv": pass_csv,
        "scaffold_contact_count": len(scaffold_contacts),
        "forbidden_count": len(forbidden),
        "pass_count": len(pass_df),
        "contact_forbidden_false": false_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--cutoff", type=float, default=4.0)
    args = parser.parse_args()

    summary = run(args.base_dir, args.cutoff)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
