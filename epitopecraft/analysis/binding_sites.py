"""Discover recurring canonical target contact patches across designs."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from math import dist
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..core.artifacts import (
    CanonicalResidueId,
    LocalResidueId,
    ReferenceModel,
    SelectionArtifact,
    StructureArtifact,
    TableArtifact,
)
from ..core.design import DesignSet
from ..core.step import PortSpec, Step


@dataclass(frozen=True)
class BindingSiteDiscoveryConfig:
    """Parameters for contact aggregation, patch clustering, and motif expansion."""

    structure_key: str = "design.structure"
    target_entity_ids: tuple[str, ...] = ()
    binder_entity_ids: tuple[str, ...] = ("binder",)
    contact_distance: float = 5.0
    frequency_threshold: float = 0.5
    cluster_distance: float = 12.0
    minimum_patch_size: int = 1
    motif_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Atom:
    residue: LocalResidueId
    name: str
    coordinate: tuple[float, float, float]
    element: str


class BindingSiteDiscovery(Step):
    """Aggregate binder contacts and return canonical target patches.

    Each design must contain a mapped ``StructureArtifact`` under
    ``structure_key``. Backend-local chain and residue ids are translated
    through its ``ResidueMap`` before frequencies or spatial groups are built.
    """

    config_type = BindingSiteDiscoveryConfig
    input_ports = {
        "target": PortSpec(StructureArtifact),
        "designs": PortSpec(DesignSet),
    }
    output_ports = {
        "patches": PortSpec(tuple),
        "motifs": PortSpec(tuple),
        "table": PortSpec(TableArtifact),
    }

    def execute(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        """Aggregate mapped contacts into ranked canonical binding-site patches."""
        target: StructureArtifact = inputs["target"]
        designs: DesignSet = inputs["designs"]
        if not designs:
            raise ValueError("BindingSiteDiscovery requires at least one design")

        structures: list[StructureArtifact] = []
        for design in designs:
            artifact = design.artifacts.get(self.config.structure_key)
            if not isinstance(artifact, StructureArtifact):
                raise KeyError(
                    f"{design.id} lacks StructureArtifact {self.config.structure_key!r}"
                )
            if artifact.reference is None:
                raise ValueError(f"{artifact.id} has no canonical ReferenceModel")
            structures.append(artifact)
        reference = structures[0].reference
        target_entities = set(self.config.target_entity_ids) or {
            entity_id
            for entity_id, entity in reference.entities.items()
            if entity.kind == "target"
        }
        binder_entities = set(self.config.binder_entity_ids)

        counts: Counter[CanonicalResidueId] = Counter()
        for artifact in structures:
            contacts = _contact_target_residues(
                artifact,
                target_entities=target_entities,
                binder_entities=binder_entities,
                cutoff=self.config.contact_distance,
            )
            counts.update(contacts)

        frequencies = {
            residue: count / len(structures) for residue, count in counts.items()
        }
        selected = {
            residue
            for residue, frequency in frequencies.items()
            if frequency >= self.config.frequency_threshold
        }
        coordinates = _reference_ca_coordinates(target, reference, target_entities)
        groups = _connected_groups(
            selected,
            coordinates,
            distance_cutoff=self.config.cluster_distance,
        )
        groups = [
            group for group in groups if len(group) >= self.config.minimum_patch_size
        ]
        groups.sort(
            key=lambda group: (
                -max(frequencies[residue] for residue in group),
                min((residue.entity_id, residue.position) for residue in group),
            )
        )
        patches = tuple(
            SelectionArtifact(
                id=f"{self.id}:patch:{index}",
                residues=tuple(sorted(group)),
                label=f"binding-site-{index}",
                provenance={
                    "step": self.id,
                    "design_count": len(structures),
                    "frequency_threshold": self.config.frequency_threshold,
                },
            )
            for index, group in enumerate(groups, start=1)
        )
        motifs = _build_motifs(
            self.id,
            selected,
            frequencies,
            coordinates,
            self.config.motif_sizes,
        )
        patch_ids = {
            residue: index
            for index, group in enumerate(groups, start=1)
            for residue in group
        }
        rows = tuple(
            {
                "canonical_entity": residue.entity_id,
                "canonical_position": residue.position,
                "author_residue": reference.author_residue(residue).label(),
                "count": counts[residue],
                "frequency": frequencies[residue],
                "patch_id": patch_ids.get(residue),
            }
            for residue in sorted(frequencies)
        )
        table_path = None
        step_dir = context.step_dir(self.id)
        if step_dir is not None:
            table_path = step_dir / "binding_sites.csv"
            _write_rows(table_path, rows)
        table = TableArtifact(
            id=f"{self.id}:table",
            rows=rows,
            path=table_path,
            provenance={"step": self.id, "design_count": len(structures)},
        )
        return {"patches": patches, "motifs": motifs, "table": table}


def _atoms(artifact: StructureArtifact) -> tuple[_Atom, ...]:
    """Read atoms from the Artifact's selected live Gemmi model."""

    structure = artifact.load_structure()
    if artifact.model_index < 0 or artifact.model_index >= len(structure):
        raise ValueError(
            f"Model index {artifact.model_index} is outside {artifact.id}"
        )
    atoms: list[_Atom] = []
    for chain in structure[artifact.model_index]:
        for residue in chain:
            local = LocalResidueId(
                chain.name or "_",
                residue.seqid.num,
                residue.seqid.icode.strip(),
            )
            for atom in residue:
                atoms.append(
                    _Atom(
                        residue=local,
                        name=atom.name,
                        coordinate=(atom.pos.x, atom.pos.y, atom.pos.z),
                        element=atom.element.name.upper(),
                    )
                )
    return tuple(atoms)


def _contact_target_residues(
    artifact: StructureArtifact,
    *,
    target_entities: set[str],
    binder_entities: set[str],
    cutoff: float,
) -> set[CanonicalResidueId]:
    target_atoms: list[tuple[CanonicalResidueId, _Atom]] = []
    binder_atoms: list[_Atom] = []
    for atom in _atoms(artifact):
        if atom.element == "H":
            continue
        try:
            canonical = artifact.residue_map.to_canonical(atom.residue)
        except KeyError:
            continue
        if canonical is None:
            continue
        if canonical.entity_id in target_entities:
            target_atoms.append((canonical, atom))
        elif canonical.entity_id in binder_entities:
            binder_atoms.append(atom)
    contacts: set[CanonicalResidueId] = set()
    for canonical, target_atom in target_atoms:
        if any(dist(target_atom.coordinate, binder.coordinate) <= cutoff for binder in binder_atoms):
            contacts.add(canonical)
    return contacts


def _reference_ca_coordinates(
    target: StructureArtifact,
    reference: ReferenceModel,
    target_entities: set[str],
) -> dict[CanonicalResidueId, tuple[float, float, float]]:
    author_to_canonical = {
        residue.author_id: residue.canonical_id
        for entity_id, entity in reference.entities.items()
        if entity_id in target_entities
        for residue in entity.residues
    }
    coordinates: dict[CanonicalResidueId, tuple[float, float, float]] = {}
    fallback: dict[CanonicalResidueId, tuple[float, float, float]] = {}
    for atom in _atoms(target):
        canonical = author_to_canonical.get(atom.residue)
        if canonical is None:
            continue
        fallback.setdefault(canonical, atom.coordinate)
        if atom.name == "CA":
            coordinates[canonical] = atom.coordinate
    for canonical, coordinate in fallback.items():
        coordinates.setdefault(canonical, coordinate)
    return coordinates


def _connected_groups(
    residues: Iterable[CanonicalResidueId],
    coordinates: dict[CanonicalResidueId, tuple[float, float, float]],
    *,
    distance_cutoff: float,
) -> list[set[CanonicalResidueId]]:
    remaining = {residue for residue in residues if residue in coordinates}
    groups: list[set[CanonicalResidueId]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = {
                candidate
                for candidate in remaining
                if dist(coordinates[current], coordinates[candidate]) <= distance_cutoff
            }
            remaining.difference_update(neighbors)
            group.update(neighbors)
            frontier.extend(neighbors)
        groups.append(group)
    return groups


def _build_motifs(
    step_id: str,
    selected: set[CanonicalResidueId],
    frequencies: dict[CanonicalResidueId, float],
    coordinates: dict[CanonicalResidueId, tuple[float, float, float]],
    motif_sizes: tuple[int, ...],
) -> tuple[SelectionArtifact, ...]:
    if not selected:
        return ()
    ranked = sorted(selected, key=lambda residue: (-frequencies[residue], residue))
    motifs = []
    all_residues = sorted(coordinates)
    for requested_size in motif_sizes:
        size = min(int(requested_size), len(all_residues))
        if size <= 0:
            continue
        chosen = list(ranked[:size])
        if len(chosen) < size:
            seeds = [coordinates[residue] for residue in chosen]
            remaining = [residue for residue in all_residues if residue not in chosen]
            remaining.sort(
                key=lambda residue: (
                    min(dist(coordinates[residue], seed) for seed in seeds),
                    residue,
                )
            )
            chosen.extend(remaining[: size - len(chosen)])
        motifs.append(
            SelectionArtifact(
                id=f"{step_id}:motif:{requested_size}",
                residues=tuple(chosen),
                label=f"motif-{requested_size}",
                provenance={"step": step_id, "requested_size": requested_size},
            )
        )
    return tuple(motifs)


def _write_rows(path: Path, rows: tuple[dict, ...]) -> None:
    fieldnames = [
        "canonical_entity",
        "canonical_position",
        "author_residue",
        "count",
        "frequency",
        "patch_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
