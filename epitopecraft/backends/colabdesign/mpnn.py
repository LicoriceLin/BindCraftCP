"""ProteinMPNN redesign over canonical StructureArtifacts."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

from ...core.artifacts import StructureArtifact
from ...core.design import Design, DesignSet, ProteinCandidate
from ...core.recipes import PenaltyRecipeBook
from ...core.step import PortSpec, Step


@dataclass(frozen=True)
class ProteinMPNNConfig:
    """Parameters consumed by one ProteinMPNN redesign instance."""

    structure_key: str = "design.structure"
    recipe_path: str = "epitopecraft/pipelines/config/mpnn-default-recipe.json"
    n_samples: int = 20
    max_sequences: int = 2
    sampling_temperature: float = 0.1
    batch_size: int = 16
    backbone_noise: float = 0.0
    model_name: str = "v_48_020"
    weights: str = "soluble"
    seed: int = 42
    binder_entity_id: str = "binder"
    target_entity_ids: tuple[str, ...] = ()
    include_input: bool = True
    selection: str = "diverse"


class ProteinMPNN(Step):
    """Redesign protein candidates with safe, freely composable bias recipes.

    Consumes mapped template structures already attached to each design. It
    identifies target and binder chains through the artifact reference, runs
    ProteinMPNN, and emits child designs with explicit lineage.
    """

    config_type = ProteinMPNNConfig
    input_ports = {"designs": PortSpec(DesignSet)}
    output_ports = {"designs": PortSpec(DesignSet)}

    @property
    def model(self):
        """Lazily construct the heavy ColabDesign model in the backend environment."""

        if getattr(self, "_model", None) is None:
            from colabdesign.mpnn import mk_mpnn_model

            self._model = mk_mpnn_model(
                backbone_noise=self.config.backbone_noise,
                model_name=self.config.model_name,
                weights=self.config.weights,
                seed=self.config.seed,
            )
        return self._model

    def execute(self, inputs, context):
        """Sample recipe-constrained sequences and return derived designs."""
        designs: DesignSet = inputs["designs"]
        recipes = PenaltyRecipeBook.from_file(self.config.recipe_path)
        output: list[Design] = []
        for design in designs:
            if not isinstance(design.candidate, ProteinCandidate):
                raise TypeError(f"ProteinMPNN requires a protein candidate: {design.id}")
            artifact = design.artifacts.get(self.config.structure_key)
            if not isinstance(artifact, StructureArtifact):
                raise KeyError(
                    f"{design.id} lacks StructureArtifact {self.config.structure_key!r}"
                )
            if artifact.reference is None:
                raise ValueError(f"{artifact.id} has no canonical ReferenceModel")
            rows = self._track_rows(design)
            plan = recipes.apply(rows)
            structure_path = self._pdb_path(artifact, design.id, context)
            target_chains, binder_chain = self._local_chains(artifact)
            fixed_positions = [*target_chains]
            fixed_positions.extend(
                f"{binder_chain}{position}"
                for position, mutable in enumerate(plan.mutable, start=1)
                if not mutable
            )
            chains = ",".join([*target_chains, binder_chain])
            self.model.prep_inputs(
                str(structure_path),
                chains,
                fix_pos=",".join(fixed_positions),
            )
            original_score = self.model.score(
                temperature=self.config.sampling_temperature
            )["score"]
            design.metrics[self.id] = {"score": float(original_score), "seqid": 1.0}
            self._install_bias(plan.bias, len(design.candidate.sequence))
            sampled = self.model.sample(
                temperature=self.config.sampling_temperature,
                num=self.config.n_samples,
                batch=self.config.batch_size,
            )
            candidates = self._deduplicate(sampled, len(design.candidate.sequence))
            selected = self._select(candidates)
            if self.config.include_input:
                output.append(design)
            for index, candidate in enumerate(selected, start=1):
                child = design.child(
                    f"{design.id}-{self.id}{index}",
                    candidate=ProteinCandidate(
                        sequence=candidate["sequence"],
                        cyclic=design.candidate.cyclic,
                    ),
                )
                child.metrics[self.id] = {
                    "score": float(candidate["score"]),
                    "seqid": float(candidate["seqid"]),
                }
                output.append(child)
        return {"designs": DesignSet.from_designs(output)}

    def _track_rows(self, design: Design) -> tuple[dict[str, Any], ...]:
        sequence = design.candidate.sequence
        rows = []
        for position, residue in enumerate(sequence):
            row = {"seq": residue}
            for name, track in design.tracks.items():
                if len(track) != len(sequence):
                    raise ValueError(
                        f"Track {name!r} on {design.id} has {len(track)} values; "
                        f"expected {len(sequence)}"
                    )
                row[name] = track[position]
            rows.append(row)
        return tuple(rows)

    def _local_chains(self, artifact: StructureArtifact) -> tuple[tuple[str, ...], str]:
        reference = artifact.reference
        mapping = reference.infer_chain_entities(artifact)
        by_entity = {entity_id: chain for chain, entity_id in mapping.items()}
        target_entities = self.config.target_entity_ids or tuple(
            entity_id
            for entity_id, entity in reference.entities.items()
            if entity.kind == "target"
        )
        missing = [entity_id for entity_id in target_entities if entity_id not in by_entity]
        if missing:
            raise ValueError(f"Target entities missing from {artifact.id}: {missing}")
        if self.config.binder_entity_id not in by_entity:
            raise ValueError(
                f"Binder entity {self.config.binder_entity_id!r} missing from {artifact.id}"
            )
        return (
            tuple(by_entity[entity_id] for entity_id in target_entities),
            by_entity[self.config.binder_entity_id],
        )

    def _pdb_path(self, artifact: StructureArtifact, design_id: str, context) -> Path:
        if artifact.structure_format in {"pdb", "ent"}:
            if artifact.path is not None and artifact.path.exists():
                return artifact.path.resolve()
            step_dir = context.step_dir(self.id)
            if step_dir is None:
                raise RuntimeError("In-memory structures require PipelineRunner(run_dir=...)")
            return artifact.export(step_dir / f"{design_id}.pdb")
        if artifact.structure_format not in {"cif", "mmcif"}:
            raise ValueError(f"ProteinMPNN cannot read {artifact.structure_format!r}")
        step_dir = context.step_dir(self.id)
        if step_dir is None:
            raise RuntimeError("CIF conversion requires PipelineRunner(run_dir=...)")
        try:
            from Bio.PDB import MMCIFParser, PDBIO
        except ImportError as error:
            raise RuntimeError("BioPython is required to convert mmCIF for ProteinMPNN") from error
        parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
        source = str(artifact.path) if artifact.path is not None else StringIO(artifact.read_text())
        structure = parser.get_structure(artifact.id, source)
        destination = step_dir / f"{design_id}.pdb"
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(destination))
        return destination

    def _install_bias(self, bias: tuple[tuple[float, ...], ...], length: int) -> None:
        import numpy as np

        array = np.asarray(bias, dtype=float)
        model_bias = self.model._inputs["bias"]
        if model_bias.shape[-1] != array.shape[-1]:
            raise ValueError(
                f"ProteinMPNN bias width {model_bias.shape[-1]} != recipe width {array.shape[-1]}"
            )
        model_bias[-length:] += array

    @staticmethod
    def _deduplicate(sampled: Mapping[str, Any], length: int) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}
        for sequence, score, seqid in zip(
            sampled["seq"], sampled["score"], sampled["seqid"]
        ):
            binder_sequence = str(sequence)[-length:]
            candidate = {
                "sequence": binder_sequence,
                "score": float(score),
                "seqid": float(seqid),
            }
            previous = unique.get(binder_sequence)
            if previous is None or candidate["score"] > previous["score"]:
                unique[binder_sequence] = candidate
        return sorted(unique.values(), key=lambda item: item["score"], reverse=True)

    def _select(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = self.config.max_sequences
        if limit < 1:
            return []
        if len(candidates) <= limit or self.config.selection == "score":
            return candidates[:limit]
        if self.config.selection != "diverse":
            raise ValueError(f"Unknown ProteinMPNN selection: {self.config.selection}")
        selected = [candidates[0]]
        remaining = candidates[1:]
        while remaining and len(selected) < limit:
            chosen = max(
                remaining,
                key=lambda candidate: (
                    min(
                        _sequence_distance(candidate["sequence"], item["sequence"])
                        for item in selected
                    ),
                    candidate["score"],
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return selected


def _sequence_distance(left: str, right: str) -> float:
    if len(left) != len(right):
        raise ValueError("ProteinMPNN candidate sequences must have equal lengths")
    return sum(a != b for a, b in zip(left, right)) / len(left)
