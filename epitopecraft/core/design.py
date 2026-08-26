"""Backend-neutral candidates, designs, and filtered collections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping


@dataclass(frozen=True)
class ProteinCandidate:
    """A protein or peptide candidate represented by its sequence."""

    sequence: str
    cyclic: bool = False

    def __post_init__(self):
        if not self.sequence:
            raise ValueError("Protein candidate sequence cannot be empty")


@dataclass(frozen=True)
class SmallMoleculeCandidate:
    """A small-molecule candidate represented by a SMILES string."""

    smiles: str

    def __post_init__(self):
        if not self.smiles:
            raise ValueError("Small-molecule SMILES cannot be empty")


Candidate = ProteinCandidate | SmallMoleculeCandidate


@dataclass
class Design:
    """One candidate and all artifacts, metrics, tracks, and lineage attached to it."""

    id: str
    candidate: Candidate
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    tracks: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def child(self, design_id: str, *, candidate: Candidate) -> "Design":
        """Create a derived design without sharing mutable metric containers."""

        return Design(
            id=design_id,
            candidate=candidate,
            artifacts=dict(self.artifacts),
            metrics=dict(self.metrics),
            tracks=dict(self.tracks),
            parent_id=self.id,
            provenance=[*self.provenance, {"operation": "derive", "parent": self.id}],
        )


class DesignSet:
    """An ordered, id-addressable collection passed between workflow Steps."""

    def __init__(self, designs: Mapping[str, Design] | None = None):
        self._designs = dict(designs or {})

    @classmethod
    def from_designs(cls, designs: Iterable[Design]) -> "DesignSet":
        """Build a collection and reject duplicate design ids."""

        result: dict[str, Design] = {}
        for design in designs:
            if design.id in result:
                raise ValueError(f"Duplicate design id: {design.id}")
            result[design.id] = design
        return cls(result)

    @property
    def ids(self) -> tuple[str, ...]:
        """Return design ids in collection order."""
        return tuple(self._designs)

    def partition(self, predicate: Callable[[Design], bool]) -> dict[str, "DesignSet"]:
        """Split records without copying the underlying Design objects."""

        passed: dict[str, Design] = {}
        rejected: dict[str, Design] = {}
        for design_id, design in self._designs.items():
            (passed if predicate(design) else rejected)[design_id] = design
        return {"passed": DesignSet(passed), "rejected": DesignSet(rejected)}

    def __getitem__(self, design_id: str) -> Design:
        return self._designs[design_id]

    def __iter__(self) -> Iterator[Design]:
        return iter(self._designs.values())

    def __len__(self) -> int:
        return len(self._designs)
