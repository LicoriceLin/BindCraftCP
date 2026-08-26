"""Structure artifacts and explicit local-to-reference residue mappings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Iterable, Iterator, Mapping

import yaml


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


class ArtifactError(ValueError):
    """Base error for malformed or unsupported artifact data."""


class Artifact:
    """Common semantic interface for data exchanged between Steps."""

    id: str
    provenance: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a compact representation suitable for debugging."""

        raise NotImplementedError


@dataclass(frozen=True)
class SequenceArtifact(Artifact):
    """A sequence tied to a canonical entity."""

    id: str
    sequence: str
    entity_id: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Describe the sequence without embedding its full contents."""
        return {"id": self.id, "entity_id": self.entity_id, "length": len(self.sequence)}


@dataclass(frozen=True)
class LigandArtifact(Artifact):
    """A small molecule and optional structure file produced by a backend."""

    id: str
    smiles: str
    path: Path | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Describe the ligand and its optional materialized path."""
        return {"id": self.id, "smiles": self.smiles, "path": str(self.path) if self.path else None}


@dataclass(frozen=True)
class TrajectoryArtifact(Artifact):
    """An MD trajectory with the topology required by downstream scorers."""

    id: str
    path: Path
    topology: StructureArtifact | Path
    frame_count: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Describe the trajectory, topology, and known frame count."""
        return {
            "id": self.id,
            "path": str(self.path),
            "frames": self.frame_count,
            "topology": getattr(self.topology, "id", str(self.topology)),
        }


@dataclass(frozen=True)
class TableArtifact(Artifact):
    """A tabular result stored as rows or a portable file."""

    id: str
    rows: tuple[Mapping[str, Any], ...] = ()
    path: Path | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Describe the table's storage location and in-memory row count."""
        return {"id": self.id, "rows": len(self.rows), "path": str(self.path) if self.path else None}


@dataclass(frozen=True, order=True)
class LocalResidueId:
    """A residue identifier exactly as emitted by one backend artifact."""

    chain_id: str
    sequence_id: int
    insertion_code: str = ""

    def label(self) -> str:
        """Render a compact backend-local residue label."""
        return f"{self.chain_id}:{self.sequence_id}{self.insertion_code}"


@dataclass(frozen=True, order=True)
class CanonicalResidueId:
    """A stable entity-local position shared by every artifact in a design."""

    entity_id: str
    position: int

    def label(self) -> str:
        """Render a compact canonical entity-position label."""
        return f"{self.entity_id}:{self.position}"


@dataclass(frozen=True)
class SelectionArtifact(Artifact):
    """A named canonical residue selection independent of backend numbering."""

    id: str
    residues: tuple[CanonicalResidueId, ...]
    label: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Describe selection size and participating canonical entities."""
        entities = sorted({residue.entity_id for residue in self.residues})
        return {
            "id": self.id,
            "label": self.label,
            "entities": entities,
            "residues": len(self.residues),
        }


@dataclass(frozen=True)
class ParsedResidue:
    """A unique polymer residue parsed from a PDB artifact."""

    local_id: LocalResidueId
    residue_name: str
    one_letter: str


@dataclass(frozen=True)
class ReferenceResidue:
    """Canonical position linked to its original author numbering."""

    canonical_id: CanonicalResidueId
    author_id: LocalResidueId
    residue_name: str
    one_letter: str


@dataclass(frozen=True)
class ReferenceEntity:
    """One target, binder, or ligand entity in the design reference."""

    id: str
    kind: str
    chain_id: str | None
    residues: tuple[ReferenceResidue, ...] = ()
    smiles: str | None = None

    @property
    def sequence(self) -> str:
        """Return the canonical one-letter polymer sequence."""
        return "".join(residue.one_letter for residue in self.residues)


@dataclass(frozen=True)
class ResidueMapEntry:
    """One local residue correspondence and the reason for its status."""

    local_id: LocalResidueId
    canonical_id: CanonicalResidueId | None
    status: str
    local_residue_name: str
    reference_residue_name: str | None = None


class ResidueMap:
    """Bidirectional residue correspondence for one StructureArtifact."""

    def __init__(self, entries: Iterable[ResidueMapEntry] = ()):
        self.entries = tuple(entries)
        self._by_local = {entry.local_id: entry for entry in self.entries}
        if len(self._by_local) != len(self.entries):
            raise ArtifactError("ResidueMap contains duplicate local residue ids")
        inverse: dict[CanonicalResidueId, list[LocalResidueId]] = {}
        for entry in self.entries:
            if entry.canonical_id is not None:
                inverse.setdefault(entry.canonical_id, []).append(entry.local_id)
        self._by_canonical = {key: tuple(value) for key, value in inverse.items()}

    def to_canonical(self, local_id: LocalResidueId) -> CanonicalResidueId | None:
        """Translate a backend-local residue into the design reference."""

        return self._by_local[local_id].canonical_id

    def from_canonical(self, canonical_id: CanonicalResidueId) -> tuple[LocalResidueId, ...]:
        """Return every local residue associated with a canonical position."""

        return self._by_canonical.get(canonical_id, ())

    def validate(self) -> list[str]:
        """Return human-readable issues without raising during inspection."""

        issues: list[str] = []
        for canonical, locals_ in self._by_canonical.items():
            if len(locals_) > 1:
                labels = ", ".join(item.label() for item in locals_)
                issues.append(f"Multiple local residues map to {canonical.label()}: {labels}")
        return issues

    def summary(self) -> dict[str, int]:
        """Count exact, substituted, and unmapped residues."""

        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return counts


@dataclass(frozen=True)
class DebugBundle:
    """Paths emitted by ``StructureArtifact.write_debug_bundle``."""

    raw: Path
    canonicalized: Path
    residue_map: Path
    entities: Path
    validation: Path
    provenance: Path


@dataclass(frozen=True)
class StructureArtifact(Artifact):
    """A structure plus its coordinate mapping and transformation provenance."""

    id: str
    structure_format: str
    text: str | None = field(default=None, repr=False)
    path: Path | None = None
    residue_map: ResidueMap = field(default_factory=ResidueMap, repr=False)
    reference: "ReferenceModel | None" = field(default=None, repr=False)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        artifact_id: str,
        text: str,
        *,
        structure_format: str = "pdb",
        provenance: Mapping[str, Any] | None = None,
    ) -> "StructureArtifact":
        """Create an in-memory structure artifact."""

        return cls(
            id=artifact_id,
            structure_format=structure_format.lower(),
            text=text,
            provenance=dict(provenance or {}),
        )

    @classmethod
    def from_file(
        cls,
        artifact_id: str,
        path: str | Path,
        *,
        structure_format: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "StructureArtifact":
        """Create a lazily loaded artifact from a PDB or CIF path."""

        resolved = Path(path)
        return cls(
            id=artifact_id,
            structure_format=(structure_format or resolved.suffix.lstrip(".")).lower(),
            path=resolved,
            provenance=dict(provenance or {}),
        )

    def read_text(self) -> str:
        """Return structure text from memory or its source path."""

        if self.text is not None:
            return self.text
        if self.path is None:
            raise ArtifactError(f"Artifact {self.id} has neither text nor path")
        return self.path.read_text()

    def residues(self) -> tuple[ParsedResidue, ...]:
        """Parse unique polymer residues in file order."""

        if self.structure_format in {"pdb", "ent"}:
            return tuple(_parse_pdb_residues(self.read_text()))
        if self.structure_format in {"cif", "mmcif"}:
            return tuple(_parse_mmcif_residues(self))
        raise ArtifactError(f"Unsupported structure format: {self.structure_format!r}")

    def summary(self) -> dict[str, Any]:
        """Return a compact printable description for interactive debugging."""

        residues = self.residues()
        return {
            "id": self.id,
            "format": self.structure_format,
            "path": str(self.path) if self.path else None,
            "chains": sorted({item.local_id.chain_id for item in residues}),
            "residues": len(residues),
            "mapping": self.residue_map.summary(),
            "provenance": dict(self.provenance),
        }

    def validate(self) -> list[str]:
        """Return mapping and structure issues suitable for a debug report."""

        issues = self.residue_map.validate()
        if not self.read_text().strip():
            issues.append("Structure text is empty")
        return issues

    def export(self, path: str | Path, *, numbering: str = "raw") -> Path:
        """Write raw, author-reference, or contiguous canonical PDB numbering."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if numbering == "raw":
            output = self.read_text()
        elif numbering in {"reference", "canonical"}:
            if self.reference is None:
                raise ArtifactError("Reference-numbered export requires a ReferenceModel")
            if self.structure_format not in {"pdb", "ent"}:
                raise ArtifactError("Reference-numbered export currently requires PDB input")
            output = _rewrite_pdb_numbering(
                self.read_text(),
                self.residue_map,
                self.reference,
                canonical=numbering == "canonical",
            )
        else:
            raise ValueError("numbering must be 'raw', 'reference', or 'canonical'")
        destination.write_text(output)
        return destination

    def write_debug_bundle(self, directory: str | Path) -> DebugBundle:
        """Write raw structure, mapping tables, validation, and provenance."""

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        suffix = "pdb" if self.structure_format == "ent" else self.structure_format
        raw = self.export(target / f"raw.{suffix}")
        canonical = target / f"canonicalized.{suffix}"
        if self.reference is not None and self.structure_format in {"pdb", "ent"}:
            self.export(canonical, numbering="reference")
        else:
            canonical.write_text(self.read_text())

        residue_map_path = target / "residue_map.tsv"
        rows = [
            "local_chain\tlocal_residue\tinsertion_code\tcanonical_entity\t"
            "canonical_position\tstatus\tlocal_name\treference_name"
        ]
        for entry in self.residue_map.entries:
            canonical_id = entry.canonical_id
            rows.append(
                "\t".join(
                    [
                        entry.local_id.chain_id,
                        str(entry.local_id.sequence_id),
                        entry.local_id.insertion_code,
                        canonical_id.entity_id if canonical_id else "",
                        str(canonical_id.position) if canonical_id else "",
                        entry.status,
                        entry.local_residue_name,
                        entry.reference_residue_name or "",
                    ]
                )
            )
        residue_map_path.write_text("\n".join(rows) + "\n")

        entities_path = target / "entities.yaml"
        entities = self.reference.to_dict() if self.reference else {"entities": {}}
        entities_path.write_text(yaml.safe_dump(entities, sort_keys=False))
        validation_path = target / "validation.json"
        validation_path.write_text(json.dumps({"issues": self.validate()}, indent=2))
        provenance_path = target / "provenance.json"
        provenance_path.write_text(json.dumps(dict(self.provenance), indent=2, default=str))
        return DebugBundle(
            raw=raw,
            canonicalized=canonical,
            residue_map=residue_map_path,
            entities=entities_path,
            validation=validation_path,
            provenance=provenance_path,
        )


class ReferenceModel:
    """Canonical entities and author residue ids for one target/design lineage."""

    def __init__(
        self,
        entities: Mapping[str, ReferenceEntity] | None = None,
        *,
        source_artifact_id: str | None = None,
        source_preference: str | None = None,
    ):
        self.entities = dict(entities or {})
        self.source_artifact_id = source_artifact_id
        self.source_preference = source_preference

    def copy(self) -> "ReferenceModel":
        """Copy the entity registry for one design-specific binder lineage."""

        return ReferenceModel(
            self.entities,
            source_artifact_id=self.source_artifact_id,
            source_preference=self.source_preference,
        )

    @classmethod
    def from_target(
        cls,
        *,
        full_structure: StructureArtifact | None = None,
        local_structure: StructureArtifact | None = None,
        target_chains: Iterable[str] | None = None,
    ) -> "ReferenceModel":
        """Build a target reference, preferring the complete structure when present."""

        source = full_structure or local_structure
        if source is None:
            raise ArtifactError("A full or local target structure is required")
        preference = "full" if full_structure is not None else "local"
        parsed = source.residues()
        available_chains = tuple(dict.fromkeys(item.local_id.chain_id for item in parsed))
        chains = tuple(target_chains or available_chains)
        missing = set(chains) - set(available_chains)
        if missing:
            raise ArtifactError(
                f"Target chains missing from {source.id}: {', '.join(sorted(missing))}"
            )
        entities: dict[str, ReferenceEntity] = {}
        for chain_id in chains:
            chain_residues = [item for item in parsed if item.local_id.chain_id == chain_id]
            entity_id = f"target:{chain_id}"
            reference_residues = tuple(
                ReferenceResidue(
                    canonical_id=CanonicalResidueId(entity_id, position),
                    author_id=item.local_id,
                    residue_name=item.residue_name,
                    one_letter=item.one_letter,
                )
                for position, item in enumerate(chain_residues, start=1)
            )
            entities[entity_id] = ReferenceEntity(
                id=entity_id,
                kind="target",
                chain_id=chain_id,
                residues=reference_residues,
            )
        return cls(
            entities,
            source_artifact_id=source.id,
            source_preference=preference,
        )

    def add_protein_binder(
        self,
        sequence: str,
        *,
        preferred_chain: str = "B",
        entity_id: str = "binder",
    ) -> ReferenceEntity:
        """Add a de novo binder with a non-conflicting canonical chain."""

        if entity_id in self.entities:
            existing = self.entities[entity_id]
            if existing.sequence != sequence:
                raise ArtifactError(
                    f"Binder entity {entity_id!r} already exists with another sequence"
                )
            return existing
        used = {entity.chain_id for entity in self.entities.values() if entity.chain_id}
        chain_id = preferred_chain if preferred_chain not in used else next(
            (candidate for candidate in ascii_uppercase if candidate not in used),
            None,
        )
        if chain_id is None:
            raise ArtifactError("No single-letter chain id remains for the binder")
        residues = tuple(
            ReferenceResidue(
                canonical_id=CanonicalResidueId(entity_id, position),
                author_id=LocalResidueId(chain_id, position),
                residue_name=_one_to_three(one_letter),
                one_letter=one_letter,
            )
            for position, one_letter in enumerate(sequence, start=1)
        )
        entity = ReferenceEntity(
            id=entity_id,
            kind="protein_binder",
            chain_id=chain_id,
            residues=residues,
        )
        self.entities[entity_id] = entity
        return entity

    def add_small_molecule(
        self,
        smiles: str,
        *,
        entity_id: str = "ligand",
    ) -> ReferenceEntity:
        """Add a ligand entity without inventing protein residues."""

        if entity_id in self.entities:
            raise ArtifactError(f"Entity already exists: {entity_id}")
        entity = ReferenceEntity(
            id=entity_id,
            kind="small_molecule",
            chain_id=None,
            smiles=smiles,
        )
        self.entities[entity_id] = entity
        return entity

    def author_residue(self, canonical_id: CanonicalResidueId) -> LocalResidueId:
        """Resolve a canonical position back to its reference author id."""

        entity = self.entities[canonical_id.entity_id]
        if canonical_id.position < 1 or canonical_id.position > len(entity.residues):
            raise KeyError(canonical_id)
        return entity.residues[canonical_id.position - 1].author_id

    def map_structure(
        self,
        artifact: StructureArtifact,
        chain_entities: Mapping[str, str],
    ) -> StructureArtifact:
        """Align each local chain to a declared canonical reference entity."""

        parsed = artifact.residues()
        entries: list[ResidueMapEntry] = []
        for chain_id in dict.fromkeys(item.local_id.chain_id for item in parsed):
            local_residues = [item for item in parsed if item.local_id.chain_id == chain_id]
            entity_id = chain_entities.get(chain_id)
            if entity_id is None:
                entries.extend(
                    ResidueMapEntry(
                        local_id=item.local_id,
                        canonical_id=None,
                        status="unmapped_chain",
                        local_residue_name=item.residue_name,
                    )
                    for item in local_residues
                )
                continue
            if entity_id not in self.entities:
                raise ArtifactError(f"Unknown reference entity: {entity_id}")
            entity = self.entities[entity_id]
            if not entity.residues:
                raise ArtifactError(f"Entity {entity_id} has no polymer residues")
            local_sequence = "".join(item.one_letter for item in local_residues)
            alignment = _align_local_to_reference(local_sequence, entity.sequence)
            for local_index, item in enumerate(local_residues):
                reference_index = alignment.get(local_index)
                if reference_index is None:
                    entries.append(
                        ResidueMapEntry(
                            local_id=item.local_id,
                            canonical_id=None,
                            status="unmapped_residue",
                            local_residue_name=item.residue_name,
                        )
                    )
                    continue
                reference_residue = entity.residues[reference_index]
                status = (
                    "exact"
                    if item.one_letter == reference_residue.one_letter
                    else "substitution"
                )
                entries.append(
                    ResidueMapEntry(
                        local_id=item.local_id,
                        canonical_id=reference_residue.canonical_id,
                        status=status,
                        local_residue_name=item.residue_name,
                        reference_residue_name=reference_residue.residue_name,
                    )
                )
        provenance = {
            **dict(artifact.provenance),
            "reference_source": self.source_artifact_id,
            "chain_entities": dict(chain_entities),
        }
        return replace(
            artifact,
            residue_map=ResidueMap(entries),
            reference=self,
            provenance=provenance,
        )

    def infer_chain_entities(
        self,
        artifact: StructureArtifact,
        *,
        entity_ids: Iterable[str] | None = None,
        minimum_identity: float = 0.5,
    ) -> dict[str, str]:
        """Match renamed backend chains to reference entities by sequence.

        Matching is one-to-one and prioritizes identity, local-chain coverage,
        and the number of exact residues.  It intentionally does not trust
        backend chain labels, which commonly change between preparation and
        prediction.
        """

        parsed = artifact.residues()
        chains = {
            chain_id: tuple(item for item in parsed if item.local_id.chain_id == chain_id)
            for chain_id in dict.fromkeys(item.local_id.chain_id for item in parsed)
        }
        allowed = set(entity_ids) if entity_ids is not None else set(self.entities)
        entities = {
            entity_id: entity
            for entity_id, entity in self.entities.items()
            if entity_id in allowed and entity.residues
        }
        candidates: list[tuple[tuple[float, float, int, int], str, str]] = []
        for chain_id, local_residues in chains.items():
            local_sequence = "".join(item.one_letter for item in local_residues)
            for entity_id, entity in entities.items():
                alignment = _align_local_to_reference(local_sequence, entity.sequence)
                pairs = tuple(alignment.items())
                if not pairs:
                    continue
                exact = sum(
                    local_sequence[local_index] == entity.sequence[reference_index]
                    for local_index, reference_index in pairs
                )
                identity = exact / len(pairs)
                local_coverage = len(pairs) / len(local_sequence)
                if identity < minimum_identity:
                    continue
                length_delta = -abs(len(local_sequence) - len(entity.sequence))
                candidates.append(
                    ((identity, local_coverage, exact, length_delta), chain_id, entity_id)
                )
        assigned_chains: set[str] = set()
        assigned_entities: set[str] = set()
        result: dict[str, str] = {}
        for _, chain_id, entity_id in sorted(candidates, reverse=True):
            if chain_id in assigned_chains or entity_id in assigned_entities:
                continue
            result[chain_id] = entity_id
            assigned_chains.add(chain_id)
            assigned_entities.add(entity_id)
        return result

    def map_structure_auto(
        self,
        artifact: StructureArtifact,
        *,
        entity_ids: Iterable[str] | None = None,
        minimum_identity: float = 0.5,
    ) -> StructureArtifact:
        """Infer renamed chain identities, then create the residue map."""

        chain_entities = self.infer_chain_entities(
            artifact,
            entity_ids=entity_ids,
            minimum_identity=minimum_identity,
        )
        if not chain_entities:
            raise ArtifactError(f"No chains in {artifact.id} match the reference entities")
        return self.map_structure(artifact, chain_entities)

    def to_dict(self) -> dict[str, Any]:
        """Serialize entity definitions for manifests and debug bundles."""

        return {
            "source_artifact_id": self.source_artifact_id,
            "source_preference": self.source_preference,
            "entities": {
                entity_id: {
                    "kind": entity.kind,
                    "chain_id": entity.chain_id,
                    "sequence": entity.sequence or None,
                    "smiles": entity.smiles,
                    "residues": [
                        {
                            "position": residue.canonical_id.position,
                            "author_chain": residue.author_id.chain_id,
                            "author_sequence_id": residue.author_id.sequence_id,
                            "insertion_code": residue.author_id.insertion_code,
                            "residue_name": residue.residue_name,
                        }
                        for residue in entity.residues
                    ],
                }
                for entity_id, entity in self.entities.items()
            },
        }


def _parse_pdb_residues(text: str) -> Iterator[ParsedResidue]:
    seen: set[LocalResidueId] = set()
    for line in text.splitlines():
        if line[:6].strip() not in {"ATOM", "HETATM"} or len(line) < 27:
            continue
        chain_id = line[21].strip() or "_"
        try:
            sequence_id = int(line[22:26])
        except ValueError as error:
            raise ArtifactError(f"Invalid PDB residue number in line: {line}") from error
        local_id = LocalResidueId(chain_id, sequence_id, line[26].strip())
        if local_id in seen:
            continue
        seen.add(local_id)
        residue_name = line[17:20].strip().upper()
        yield ParsedResidue(
            local_id=local_id,
            residue_name=residue_name,
            one_letter=AA3_TO_1.get(residue_name, "X"),
        )


def _parse_mmcif_residues(artifact: StructureArtifact) -> Iterator[ParsedResidue]:
    """Parse author residue ids from mmCIF through the optional BioPython adapter."""

    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as error:
        raise ArtifactError("BioPython is required to parse mmCIF artifacts") from error

    if artifact.path is not None:
        data = MMCIF2Dict(str(artifact.path))
    else:
        from io import StringIO

        data = MMCIF2Dict(StringIO(artifact.read_text()))

    def values(*keys: str) -> list[str]:
        """Return the first present mmCIF field as a normalized list."""
        for key in keys:
            if key in data:
                value = data[key]
                return list(value) if isinstance(value, list) else [value]
        raise ArtifactError(f"mmCIF atom_site field is missing: {' or '.join(keys)}")

    groups = values("_atom_site.group_PDB")
    residue_names = values("_atom_site.auth_comp_id", "_atom_site.label_comp_id")
    chains = values("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
    sequence_ids = values("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
    insertions = values("_atom_site.pdbx_PDB_ins_code")
    seen: set[LocalResidueId] = set()
    for group, residue_name, chain, sequence_id, insertion in zip(
        groups, residue_names, chains, sequence_ids, insertions
    ):
        if group not in {"ATOM", "HETATM"}:
            continue
        try:
            numeric_id = int(float(sequence_id))
        except ValueError as error:
            raise ArtifactError(f"Invalid mmCIF author residue id: {sequence_id}") from error
        local_id = LocalResidueId(
            chain if chain not in {".", "?"} else "_",
            numeric_id,
            "" if insertion in {".", "?"} else insertion,
        )
        if local_id in seen:
            continue
        seen.add(local_id)
        normalized_name = residue_name.upper()
        yield ParsedResidue(
            local_id=local_id,
            residue_name=normalized_name,
            one_letter=AA3_TO_1.get(normalized_name, "X"),
        )


def _align_local_to_reference(local: str, reference: str) -> dict[int, int]:
    """Globally align sequences and return local-index to reference-index pairs."""

    rows = len(local) + 1
    columns = len(reference) + 1
    gap = -2
    score = [[0] * columns for _ in range(rows)]
    trace = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        score[row][0] = row * gap
        trace[row][0] = "up"
    for column in range(1, columns):
        score[0][column] = column * gap
        trace[0][column] = "left"
    for row in range(1, rows):
        for column in range(1, columns):
            diagonal = score[row - 1][column - 1] + (
                2 if local[row - 1] == reference[column - 1] else -1
            )
            up = score[row - 1][column] + gap
            left = score[row][column - 1] + gap
            best = max(diagonal, up, left)
            score[row][column] = best
            trace[row][column] = "diag" if best == diagonal else ("up" if best == up else "left")
    row = len(local)
    column = len(reference)
    result: dict[int, int] = {}
    while row or column:
        direction = trace[row][column]
        if direction == "diag":
            result[row - 1] = column - 1
            row -= 1
            column -= 1
        elif direction == "up":
            row -= 1
        elif direction == "left":
            column -= 1
        else:
            break
    return result


def _rewrite_pdb_numbering(
    text: str,
    residue_map: ResidueMap,
    reference: ReferenceModel,
    *,
    canonical: bool = False,
) -> str:
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        if line[:6].strip() not in {"ATOM", "HETATM"} or len(line) < 27:
            output.append(line)
            continue
        try:
            local_id = LocalResidueId(
                line[21].strip() or "_",
                int(line[22:26]),
                line[26].strip(),
            )
            canonical_id = residue_map.to_canonical(local_id)
        except (KeyError, ValueError):
            canonical_id = None
        if canonical_id is None:
            output.append(line)
            continue
        if canonical:
            entity = reference.entities[canonical_id.entity_id]
            author = LocalResidueId(
                entity.chain_id or canonical_id.entity_id[:1],
                canonical_id.position,
            )
        else:
            author = reference.author_residue(canonical_id)
        padded = line.rstrip("\n").ljust(27)
        rewritten = (
            padded[:21]
            + author.chain_id[:1]
            + f"{author.sequence_id:4d}"
            + (author.insertion_code[:1] or " ")
            + padded[27:]
        )
        output.append(rewritten + ("\n" if line.endswith("\n") else ""))
    return "".join(output)


def _one_to_three(one_letter: str) -> str:
    inverse = {value: key for key, value in AA3_TO_1.items() if key != "MSE"}
    return inverse.get(one_letter.upper(), "UNK")
