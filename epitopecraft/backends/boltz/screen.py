"""Known-site protein and small-molecule screening with Boltz-2."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import yaml

from ...core.artifacts import (
    LigandArtifact,
    ReferenceModel,
    SelectionArtifact,
    StructureArtifact,
)
from ...core.design import (
    Design,
    DesignSet,
    ProteinCandidate,
    SmallMoleculeCandidate,
)
from ...core.step import PortSpec, Step


@dataclass(frozen=True)
class Boltz2ScreenConfig:
    """Parameters consumed by one known-site Boltz-2 screening instance."""

    target_msa: str | None = "empty"
    target_msa_by_entity: tuple[str, ...] = ()
    use_target_template: bool = False
    binder_id: str = "B"
    include_affinity: bool = True
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    sampling_steps_affinity: int = 200
    diffusion_samples_affinity: int = 3
    accelerator: str = "gpu"
    devices: int = 1
    num_workers: int = 2
    preprocessing_threads: int = 1
    max_msa_seqs: int = 8192
    output_format: str = "mmcif"
    seed: int | None = None
    override: bool = False
    no_kernels: bool = False
    command_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "-n",
        "binding_affinity",
        "boltz",
    )


def _msa_by_entity(config: Boltz2ScreenConfig) -> dict[str, str]:
    result: dict[str, str] = {}
    for specification in config.target_msa_by_entity:
        if "=" not in specification:
            raise ValueError(
                "target_msa_by_entity entries must use canonical_entity=path"
            )
        entity_id, path = specification.split("=", 1)
        result[entity_id.strip()] = path.strip()
    return result


def _binder_id(reference: ReferenceModel, preferred: str) -> str:
    used = {entity.chain_id for entity in reference.entities.values() if entity.chain_id}
    if preferred not in used:
        return preferred
    candidate = next((chain for chain in ascii_uppercase if chain not in used), None)
    if candidate is None:
        raise ValueError("No free single-letter Boltz entity id remains")
    return candidate


def build_boltz2_input(
    reference: ReferenceModel,
    site: SelectionArtifact,
    candidate: ProteinCandidate | SmallMoleculeCandidate,
    config: Boltz2ScreenConfig,
    *,
    target_path: str | Path,
) -> dict[str, Any]:
    """Build one Boltz-2 YAML payload from canonical target positions."""

    msa_overrides = _msa_by_entity(config)
    target_entities = [
        entity for entity in reference.entities.values() if entity.kind == "target"
    ]
    if not target_entities:
        raise ValueError("Reference contains no target entities")
    sequences: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    for entity in target_entities:
        protein: dict[str, Any] = {"id": entity.chain_id, "sequence": entity.sequence}
        msa = msa_overrides.get(entity.id, config.target_msa)
        if msa is not None:
            protein["msa"] = msa
        sequences.append({"protein": protein})
        if config.use_target_template:
            templates.append(
                {
                    "pdb": str(target_path),
                    "chain_id": entity.chain_id,
                    "template_id": entity.chain_id,
                }
            )

    binder_id = _binder_id(reference, config.binder_id)
    if isinstance(candidate, ProteinCandidate):
        binder: dict[str, Any] = {
            "id": binder_id,
            "sequence": candidate.sequence,
            "msa": "empty",
        }
        if candidate.cyclic:
            binder["cyclic"] = True
        sequences.append({"protein": binder})
    elif isinstance(candidate, SmallMoleculeCandidate):
        sequences.append({"ligand": {"id": binder_id, "smiles": candidate.smiles}})
    else:
        raise TypeError(f"Unsupported screening candidate: {type(candidate)}")

    contacts = []
    for residue in site.residues:
        if residue.entity_id not in reference.entities:
            raise ValueError(f"Site uses unknown entity: {residue.entity_id}")
        entity = reference.entities[residue.entity_id]
        if entity.kind != "target":
            raise ValueError(f"Site residue is not on a target entity: {residue.label()}")
        contacts.append([entity.chain_id, residue.position])
    payload: dict[str, Any] = {
        "version": 1,
        "sequences": sequences,
        "constraints": [{"pocket": {"binder": binder_id, "contacts": contacts}}],
    }
    if templates:
        payload["templates"] = templates
    if isinstance(candidate, SmallMoleculeCandidate) and config.include_affinity:
        payload["properties"] = [{"affinity": {"binder": binder_id}}]
    return payload


def build_boltz2_command(
    config: Boltz2ScreenConfig,
    input_dir: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Build the Boltz-2 batch prediction command."""

    command = [
        *config.command_prefix,
        "predict",
        str(input_dir),
        "--out_dir",
        str(output_dir),
        "--model",
        "boltz2",
        "--accelerator",
        config.accelerator,
        "--devices",
        str(config.devices),
        "--recycling_steps",
        str(config.recycling_steps),
        "--sampling_steps",
        str(config.sampling_steps),
        "--diffusion_samples",
        str(config.diffusion_samples),
        "--sampling_steps_affinity",
        str(config.sampling_steps_affinity),
        "--diffusion_samples_affinity",
        str(config.diffusion_samples_affinity),
        "--num_workers",
        str(config.num_workers),
        "--preprocessing-threads",
        str(config.preprocessing_threads),
        "--max_msa_seqs",
        str(config.max_msa_seqs),
        "--output_format",
        config.output_format,
    ]
    if config.seed is not None:
        command.extend(["--seed", str(config.seed)])
    if config.override:
        command.append("--override")
    if config.no_kernels:
        command.append("--no_kernels")
    return command


class Boltz2Screen(Step):
    """Co-fold protein or SMILES candidates at one canonical target site."""

    config_type = Boltz2ScreenConfig
    input_ports = {
        "target": PortSpec(StructureArtifact),
        "site": PortSpec(SelectionArtifact),
        "candidates": PortSpec(tuple),
    }
    output_ports = {"designs": PortSpec(DesignSet)}

    def execute(self, inputs, context):
        """Run one Boltz-2 batch and attach mapped structures and scores."""
        target: StructureArtifact = inputs["target"]
        site: SelectionArtifact = inputs["site"]
        candidates: tuple = inputs["candidates"]
        step_dir = context.step_dir(self.id)
        if step_dir is None:
            raise RuntimeError("Boltz2Screen requires PipelineRunner(run_dir=...)")
        reference = target.reference or ReferenceModel.from_target(full_structure=target)
        target_path = self._materialize_target(target, reference, step_dir)
        input_root = step_dir / "inputs"
        input_root.mkdir(parents=True, exist_ok=True)
        cases: dict[str, tuple[Any, dict[str, Any]]] = {}
        groups: dict[str, list[str]] = {"structure": [], "affinity": []}
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, (ProteinCandidate, SmallMoleculeCandidate)):
                raise TypeError(f"Unsupported screening candidate: {type(candidate)}")
            case_id = f"candidate_{index:05d}"
            payload = build_boltz2_input(
                reference,
                site,
                candidate,
                self.config,
                target_path=target_path,
            )
            group = "affinity" if payload.get("properties") else "structure"
            input_dir = input_root / group
            input_dir.mkdir(parents=True, exist_ok=True)
            yaml_path = input_dir / f"{case_id}.yaml"
            yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
            cases[case_id] = (candidate, payload)
            groups[group].append(case_id)

        prediction_roots: dict[str, Path] = {}
        for group, case_ids in groups.items():
            if not case_ids:
                continue
            input_dir = input_root / group
            run_root = step_dir / "run" / group
            command = build_boltz2_command(self.config, input_dir, run_root)
            subprocess.run(command, check=True)
            predictions = (
                run_root / f"boltz_results_{input_dir.stem}" / "predictions"
            )
            prediction_roots.update({case_id: predictions for case_id in case_ids})
        designs = [
            self._read_case(
                case_id,
                candidate,
                payload,
                prediction_roots[case_id],
                reference,
            )
            for case_id, (candidate, payload) in cases.items()
        ]
        return {"designs": DesignSet.from_designs(designs)}

    def _materialize_target(
        self,
        target: StructureArtifact,
        reference: ReferenceModel,
        step_dir: Path,
    ) -> Path:
        if self.config.use_target_template and target.structure_format in {"pdb", "ent"}:
            chain_entities = {
                entity.chain_id: entity_id
                for entity_id, entity in reference.entities.items()
                if entity.kind == "target" and entity.chain_id
            }
            mapped_target = reference.map_structure(target, chain_entities)
            return mapped_target.export(
                step_dir / "target.canonical.pdb",
                numbering="canonical",
            ).resolve()
        if target.path is not None and target.path.exists():
            return target.path.resolve()
        suffix = "pdb" if target.structure_format == "ent" else target.structure_format
        return target.export(step_dir / f"target.{suffix}").resolve()

    def _read_case(
        self,
        case_id: str,
        candidate: ProteinCandidate | SmallMoleculeCandidate,
        payload: dict[str, Any],
        predictions: Path,
        target_reference: ReferenceModel,
    ) -> Design:
        case_dir = predictions / case_id
        confidence_path = case_dir / f"confidence_{case_id}_model_0.json"
        structure_suffix = "pdb" if self.config.output_format == "pdb" else "cif"
        structure_path = case_dir / f"{case_id}_model_0.{structure_suffix}"
        if not confidence_path.exists() or not structure_path.exists():
            raise FileNotFoundError(f"Incomplete Boltz-2 output for {case_id}: {case_dir}")
        confidence = json.loads(confidence_path.read_text())
        metrics: dict[str, Any] = {self.id: {"confidence": confidence}}
        affinity_path = case_dir / f"affinity_{case_id}.json"
        if affinity_path.exists():
            metrics[self.id]["affinity"] = json.loads(affinity_path.read_text())
        artifact = StructureArtifact.from_file(
            f"{case_id}:{self.id}",
            structure_path,
            structure_format=structure_suffix,
            provenance={"step": self.id, "backend": "boltz2", "input": payload},
        )
        reference = target_reference.copy()
        binder_id = payload["constraints"][0]["pocket"]["binder"]
        if isinstance(candidate, ProteinCandidate):
            reference.add_protein_binder(
                candidate.sequence,
                preferred_chain=binder_id,
            )
        else:
            reference.add_small_molecule(candidate.smiles)
        mapped = reference.map_structure_auto(artifact)
        artifacts: dict[str, Any] = {f"{self.id}.structure": mapped}
        if isinstance(candidate, SmallMoleculeCandidate):
            artifacts[f"{self.id}.ligand"] = LigandArtifact(
                id=f"{case_id}:ligand",
                smiles=candidate.smiles,
                provenance={"step": self.id, "yaml_id": binder_id},
            )
        return Design(
            id=case_id,
            candidate=candidate,
            artifacts=artifacts,
            metrics=metrics,
            provenance=[{"operation": "screen", "step": self.id}],
        )
