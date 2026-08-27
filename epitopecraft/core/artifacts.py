"""Structure artifacts and explicit local-to-reference residue mappings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Iterable, Iterator, Mapping

import gemmi
import yaml


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
    """A unique polymer residue read from one coordinate model."""

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

    def __init__(self, entries: Iterable[ResidueMapEntry] = ()) -> None:
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


@dataclass
class StructureArtifact(Artifact):
    """A mutable Gemmi structure plus mapping and transformation provenance."""

    id: str
    structure_format: str
    structure: gemmi.Structure = field(repr=False)
    path: Path | None = None
    model_index: int = 0
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
        model_index: int = 0,
        provenance: Mapping[str, Any] | None = None,
    ) -> "StructureArtifact":
        """Create an in-memory structure artifact."""

        normalized_format = _normalize_structure_format(structure_format)
        return cls(
            id=artifact_id,
            structure_format=normalized_format,
            structure=_parse_structure_text(artifact_id, text, normalized_format),
            model_index=model_index,
            provenance=dict(provenance or {}),
        )

    @classmethod
    def from_file(
        cls,
        artifact_id: str,
        path: str | Path,
        *,
        structure_format: str | None = None,
        model_index: int = 0,
        provenance: Mapping[str, Any] | None = None,
    ) -> "StructureArtifact":
        """Parse a PDB or mmCIF path into an in-memory Gemmi structure."""

        resolved = Path(path)
        normalized_format = _normalize_structure_format(
            structure_format or _format_from_path(resolved)
        )
        return cls(
            id=artifact_id,
            structure_format=normalized_format,
            structure=_parse_structure_path(artifact_id, resolved, normalized_format),
            path=resolved,
            model_index=model_index,
            provenance=dict(provenance or {}),
        )

    @classmethod
    def from_structure(
        cls,
        artifact_id: str,
        structure: gemmi.Structure,
        *,
        structure_format: str = "cif",
        path: str | Path | None = None,
        model_index: int = 0,
        provenance: Mapping[str, Any] | None = None,
    ) -> "StructureArtifact":
        """Wrap a live Gemmi object without copying it."""

        return cls(
            id=artifact_id,
            structure_format=_normalize_structure_format(structure_format),
            structure=_prepare_structure(artifact_id, structure),
            path=Path(path) if path is not None else None,
            model_index=model_index,
            provenance=dict(provenance or {}),
        )

    def read_text(self) -> str:
        """Serialize the current mutable structure in its declared format."""

        return _serialize_structure(self.structure, self.structure_format)

    def load_structure(self) -> gemmi.Structure:
        """Return the live mutable Gemmi object held by this artifact."""

        return self.structure

    def residues(self) -> tuple[ParsedResidue, ...]:
        """Parse unique polymer residues in file order."""

        structure = self.load_structure()
        return tuple(_iter_polymer_residues(structure, self.model_index))

    def summary(self) -> dict[str, Any]:
        """Return a compact printable description for interactive debugging."""

        structure = self.load_structure()
        residues = tuple(_iter_polymer_residues(structure, self.model_index))
        return {
            "id": self.id,
            "format": self.structure_format,
            "path": str(self.path) if self.path else None,
            "chains": sorted({item.local_id.chain_id for item in residues}),
            "residues": len(residues),
            "models": len(structure),
            "model_index": self.model_index,
            "mapping": self.residue_map.summary(),
            "provenance": dict(self.provenance),
        }

    def validate(self) -> list[str]:
        """Return mapping and structure issues suitable for a debug report."""

        issues = self.residue_map.validate()
        if not self.read_text().strip():
            issues.append("Structure text is empty")
            return issues
        try:
            structure = self.load_structure()
            _model_at(structure, self.model_index)
        except ArtifactError as error:
            issues.append(str(error))
        return issues

    def export(
        self,
        path: str | Path,
        *,
        numbering: str = "raw",
        structure_format: str | None = None,
    ) -> Path:
        """Write current, author-reference, or canonical coordinates."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_format = _normalize_structure_format(
            structure_format or _format_from_path(destination)
        )
        if numbering == "raw":
            output = _serialize_structure(self.structure, output_format)
        elif numbering in {"reference", "canonical"}:
            if self.reference is None:
                raise ArtifactError("Reference-numbered export requires a ReferenceModel")
            output = _rewrite_structure_numbering(
                self.load_structure(),
                self.residue_map,
                self.reference,
                structure_format=output_format,
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
        if self.reference is not None:
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
    ) -> None:
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


def _normalize_structure_format(structure_format: str) -> str:
    """Normalize supported coordinate format aliases."""

    normalized = structure_format.lower().lstrip(".")
    normalized = {"ent": "pdb", "mmcif": "cif"}.get(normalized, normalized)
    if normalized not in {"pdb", "cif"}:
        raise ArtifactError(f"Unsupported structure format: {structure_format!r}")
    return normalized


def _format_from_path(path: Path) -> str:
    """Infer a coordinate format, including paths ending in ``.gz``."""

    suffixes = [suffix.lower().lstrip(".") for suffix in path.suffixes]
    if suffixes and suffixes[-1] == "gz":
        suffixes.pop()
    if not suffixes:
        raise ArtifactError(f"Cannot infer structure format from path: {path}")
    return suffixes[-1]


def _gemmi_coordinate_format(structure_format: str) -> gemmi.CoorFormat:
    """Translate the public format name to Gemmi's coordinate enum."""

    normalized = _normalize_structure_format(structure_format)
    return gemmi.CoorFormat.Pdb if normalized == "pdb" else gemmi.CoorFormat.Mmcif


def _prepare_structure(
    artifact_id: str,
    structure: gemmi.Structure,
) -> gemmi.Structure:
    """Populate Gemmi entity annotations and reject empty coordinate models."""

    if len(structure) == 0:
        raise ArtifactError(f"Structure artifact {artifact_id} contains no models")
    structure.setup_entities()
    return structure


def _parse_structure_path(
    artifact_id: str,
    path: Path,
    structure_format: str,
) -> gemmi.Structure:
    """Parse a coordinate path with Gemmi."""

    try:
        structure = gemmi.read_structure(
            str(path),
            format=_gemmi_coordinate_format(structure_format),
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactError(f"Cannot parse structure artifact {artifact_id}: {error}") from error
    return _prepare_structure(artifact_id, structure)


def _parse_structure_text(
    artifact_id: str,
    text: str,
    structure_format: str,
) -> gemmi.Structure:
    """Parse in-memory PDB or mmCIF coordinates with Gemmi."""

    try:
        if structure_format == "pdb":
            structure = gemmi.read_pdb_string(text)
        else:
            document = gemmi.cif.read_string(text)
            if len(document) == 0:
                raise ArtifactError(f"mmCIF artifact {artifact_id} contains no data block")
            structure = gemmi.make_structure_from_block(document[0])
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactError(f"Cannot parse structure artifact {artifact_id}: {error}") from error
    return _prepare_structure(artifact_id, structure)


def _serialize_structure(
    structure: gemmi.Structure,
    structure_format: str,
) -> str:
    """Serialize the current Gemmi object in PDB or mmCIF format."""

    normalized_format = _normalize_structure_format(structure_format)
    if normalized_format == "pdb":
        invalid_chains = sorted(
            {
                chain.name
                for model in structure
                for chain in model
                if len(chain.name) > 1
            }
        )
        if invalid_chains:
            raise ArtifactError(
                "PDB export requires one-character chain ids; use mmCIF for "
                f"{invalid_chains}"
            )
        return structure.make_pdb_string()
    return structure.make_mmcif_document().as_string()


def _model_at(structure: gemmi.Structure, model_index: int) -> gemmi.Model:
    """Return one explicit model instead of silently merging ensembles."""

    if model_index < 0 or model_index >= len(structure):
        raise ArtifactError(
            f"Model index {model_index} is outside a {len(structure)}-model structure"
        )
    return structure[model_index]


def _local_residue_id(chain: gemmi.Chain, residue: gemmi.Residue) -> LocalResidueId:
    """Convert a Gemmi residue identifier into the artifact-local key."""

    return LocalResidueId(
        chain.name or "_",
        residue.seqid.num,
        residue.seqid.icode.strip(),
    )


def _iter_polymer_residues(
    structure: gemmi.Structure,
    model_index: int,
) -> Iterator[ParsedResidue]:
    """Yield polymer residues from one Gemmi model in coordinate order."""

    seen: set[LocalResidueId] = set()
    for chain in _model_at(structure, model_index):
        for residue in chain:
            if residue.entity_type != gemmi.EntityType.Polymer:
                continue
            local_id = _local_residue_id(chain, residue)
            if local_id in seen:
                raise ArtifactError(
                    f"Duplicate polymer residue id in model {model_index}: "
                    f"{local_id.label()}"
                )
            seen.add(local_id)
            residue_name = residue.name.upper()
            code = gemmi.find_tabulated_residue(residue_name).one_letter_code
            yield ParsedResidue(
                local_id=local_id,
                residue_name=residue_name,
                one_letter=code.upper() if code.strip() else "X",
            )


def _align_local_to_reference(local: str, reference: str) -> dict[int, int]:
    """Globally align sequences and return local-index to reference-index pairs."""

    alignment = gemmi.align_string_sequences(tuple(local), tuple(reference), ())
    aligned_local = alignment.add_gaps(local, 1)
    aligned_reference = alignment.add_gaps(reference, 2)
    local_index = 0
    reference_index = 0
    pairs: dict[int, int] = {}
    for local_residue, reference_residue in zip(aligned_local, aligned_reference):
        if local_residue != "-" and reference_residue != "-":
            pairs[local_index] = reference_index
        if local_residue != "-":
            local_index += 1
        if reference_residue != "-":
            reference_index += 1
    return pairs


def _rewrite_structure_numbering(
    structure: gemmi.Structure,
    residue_map: ResidueMap,
    reference: ReferenceModel,
    *,
    structure_format: str,
    canonical: bool = False,
) -> str:
    """Clone and renumber polymer residues through the canonical map."""

    rewritten = structure.clone()
    for model in rewritten:
        for chain in model:
            output_chain_ids: set[str] = set()
            for residue in chain:
                if residue.entity_type != gemmi.EntityType.Polymer:
                    continue
                try:
                    canonical_id = residue_map.to_canonical(
                        _local_residue_id(chain, residue)
                    )
                except KeyError:
                    canonical_id = None
                if canonical_id is None:
                    continue
                if canonical:
                    entity = reference.entities[canonical_id.entity_id]
                    author = LocalResidueId(
                        entity.chain_id or canonical_id.entity_id[:1],
                        canonical_id.position,
                    )
                else:
                    author = reference.author_residue(canonical_id)
                output_chain_ids.add(author.chain_id)
                residue.seqid = gemmi.SeqId(
                    author.sequence_id,
                    author.insertion_code or " ",
                )
            if len(output_chain_ids) > 1:
                raise ArtifactError(
                    f"One local chain maps to multiple output chains: "
                    f"{sorted(output_chain_ids)}"
                )
            if output_chain_ids:
                chain.name = next(iter(output_chain_ids))

    return _serialize_structure(rewritten, structure_format)


def _one_to_three(one_letter: str) -> str:
    """Expand a protein one-letter code with Gemmi's residue table."""

    return gemmi.expand_one_letter(one_letter.upper(), gemmi.ResidueKind.AA)
