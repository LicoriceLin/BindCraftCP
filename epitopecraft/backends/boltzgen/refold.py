"""Pipeline Step and command adapter for BoltzGen/Boltz2 binder co-folding."""

from __future__ import annotations

import csv
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ...core.artifacts import ReferenceModel, StructureArtifact
from ...core.design import Design, DesignSet, ProteinCandidate
from ...core.step import PortSpec, Step


@dataclass(frozen=True)
class BoltzGenRefoldConfig:
    """Parameters consumed by one BoltzGen protein-binder refold instance."""

    target_chains: tuple[str, ...] = field(
        default=(), metadata={"description": "Target chains retained from the input."}
    )
    target_res_index: str | None = None
    binding_res_index: tuple[str, ...] = ()
    target_msa: tuple[str, ...] = ()
    target_msa_res_index: tuple[str, ...] = ()
    max_msa_seqs: int | None = None
    structure_groups: str = "all"
    structure_group: tuple[str, ...] = ()
    binder_chain: str = "Z"
    cyclic: bool = False
    prepare_only: bool = False
    skip_folding: bool = False
    reuse_existing_folds: bool = False
    run_analysis: bool = False
    debug: bool = False
    accelerator: str = "gpu"
    devices: str = "1"
    precision: str = "bf16-mixed"
    num_workers: int = 1
    analysis_processes: int = 4
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 5
    fold_checkpoint: str | None = None
    moldir: str | None = None
    liability_peptide_type: str = "linear"
    no_delta_sasa_refolded: bool = False
    no_noncovalents_refolded: bool = False
    command_prefix: tuple[str, ...] = (
        "conda",
        "run",
        "-n",
        "boltzgen",
        "python",
    )


def _append_value(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def _append_many(command: list[str], flag: str, values: tuple[str, ...]) -> None:
    for value in values:
        _append_value(command, flag, value)


def build_refold_command(
    config: BoltzGenRefoldConfig,
    *,
    target_structure: str | Path,
    binder_csv: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Build the complete repository-owned co-fold CLI invocation."""

    command = [
        *config.command_prefix,
        "-m",
        "epitopecraft.backends.boltzgen.peptide_refold_cli",
        "--target-structure",
        str(target_structure),
        "--binder-csv",
        str(binder_csv),
        "--output-dir",
        str(output_dir),
    ]
    _append_many(command, "--target-chain", config.target_chains)
    _append_value(command, "--target-res-index", config.target_res_index)
    _append_many(command, "--binding-res-index", config.binding_res_index)
    _append_many(command, "--target-msa", config.target_msa)
    _append_many(command, "--target-msa-res-index", config.target_msa_res_index)
    _append_value(command, "--max-msa-seqs", config.max_msa_seqs)
    _append_value(command, "--structure-groups", config.structure_groups)
    _append_many(command, "--structure-group", config.structure_group)
    _append_value(command, "--binder-chain", config.binder_chain)
    _append_value(command, "--accelerator", config.accelerator)
    _append_value(command, "--devices", config.devices)
    _append_value(command, "--precision", config.precision)
    _append_value(command, "--num-workers", config.num_workers)
    _append_value(command, "--analysis-processes", config.analysis_processes)
    _append_value(command, "--recycling-steps", config.recycling_steps)
    _append_value(command, "--sampling-steps", config.sampling_steps)
    _append_value(command, "--diffusion-samples", config.diffusion_samples)
    _append_value(command, "--fold-checkpoint", config.fold_checkpoint)
    _append_value(command, "--moldir", config.moldir)
    _append_value(command, "--liability-peptide-type", config.liability_peptide_type)
    boolean_flags = {
        "--cyclic": config.cyclic,
        "--prepare-only": config.prepare_only,
        "--skip-folding": config.skip_folding,
        "--reuse-existing-folds": config.reuse_existing_folds,
        "--run-analysis": config.run_analysis,
        "--debug": config.debug,
        "--no-delta-sasa-refolded": config.no_delta_sasa_refolded,
        "--no-noncovalents-refolded": config.no_noncovalents_refolded,
    }
    command.extend(flag for flag, enabled in boolean_flags.items() if enabled)
    return command


class BoltzGenRefold(Step):
    """Co-fold protein candidates against a target with BoltzGen/Boltz2.

    Consumes a target structure and protein ``DesignSet``.  Produces the same
    set with namespaced folding metrics and mapped refold CIF artifacts.  The
    CLI subprocess owns heavy BoltzGen imports; this module stays lightweight.
    """

    config_type = BoltzGenRefoldConfig
    input_ports = {
        "target": PortSpec(StructureArtifact, description="Canonical target structure."),
        "designs": PortSpec(DesignSet, description="Protein candidates to co-fold."),
    }
    output_ports = {"designs": PortSpec(DesignSet)}

    def execute(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        """Run co-folding and attach namespaced metrics and mapped CIF artifacts."""
        target: StructureArtifact = inputs["target"]
        designs: DesignSet = inputs["designs"]
        output_dir = context.step_dir(self.id)
        if output_dir is None:
            raise RuntimeError("BoltzGenRefold requires PipelineRunner(run_dir=...)")
        target_path = self._materialize_target(target, output_dir)
        binder_csv, sample_to_design = self._write_binder_csv(designs, output_dir)
        command = build_refold_command(
            self.config,
            target_structure=target_path,
            binder_csv=binder_csv,
            output_dir=output_dir,
        )
        (output_dir / "command.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n"
        )
        subprocess.run(command, check=True)
        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"BoltzGen refold did not produce {summary_path}")
        summary = json.loads(summary_path.read_text())
        self._attach_results(target, summary, sample_to_design)
        return {"designs": designs}

    def _materialize_target(self, target: StructureArtifact, output_dir: Path) -> Path:
        """Export live target coordinates outside BoltzGen's design scan root."""

        input_dir = output_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        return target.export(
            input_dir / f"target.{target.structure_format}",
        ).resolve()

    def _write_binder_csv(
        self,
        designs: DesignSet,
        output_dir: Path,
    ) -> tuple[Path, dict[str, Design]]:
        path = output_dir / "binder_inputs.csv"
        sample_to_design: dict[str, Design] = {}
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "binder_sequence", "binder_chain", "cyclic"],
            )
            writer.writeheader()
            for index, design in enumerate(designs):
                if not isinstance(design.candidate, ProteinCandidate):
                    raise TypeError(
                        f"BoltzGenRefold only accepts protein candidates: {design.id}"
                    )
                sample_id = self._sample_id(design.id, index, sample_to_design)
                sample_to_design[sample_id] = design
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "binder_sequence": design.candidate.sequence,
                        "binder_chain": self.config.binder_chain,
                        "cyclic": self.config.cyclic or design.candidate.cyclic,
                    }
                )
        return path, sample_to_design

    @staticmethod
    def _sample_id(
        design_id: str,
        index: int,
        existing: dict[str, Design],
    ) -> str:
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", design_id).strip("._-")
        base = base or f"sample_{index:04d}"
        candidate = base
        suffix = 1
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _attach_results(
        self,
        target: StructureArtifact,
        summary: dict[str, Any],
        sample_to_design: dict[str, Design],
    ) -> None:
        for sample in summary.get("samples", []):
            sample_id = sample.get("sample_id")
            if sample_id not in sample_to_design:
                continue
            design = sample_to_design[sample_id]
            design.metrics[self.id] = {
                key: sample[key]
                for key in ("prepared", "prepared_input", "folding", "analysis")
                if key in sample
            }
            folding = sample.get("folding") or {}
            refold_path = folding.get("refold_cif")
            if not refold_path:
                continue
            artifact = StructureArtifact.from_file(
                f"{design.id}:{self.id}",
                refold_path,
                structure_format="cif",
                provenance={
                    "step": self.id,
                    "backend": "boltzgen/boltz2",
                    "sample_id": sample_id,
                },
            )
            reference = self._reference_for(target, design)
            design.artifacts[f"{self.id}.structure"] = reference.map_structure_auto(
                artifact
            )

    def _reference_for(
        self,
        target: StructureArtifact,
        design: Design,
    ) -> ReferenceModel:
        if target.reference is not None:
            reference = target.reference.copy()
        else:
            reference = ReferenceModel.from_target(
                full_structure=target,
                target_chains=self.config.target_chains or None,
            )
        reference.add_protein_binder(
            design.candidate.sequence,
            preferred_chain=self.config.binder_chain,
        )
        return reference
