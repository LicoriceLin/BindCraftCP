from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from pymol import cmd
from tqdm import tqdm

from .basestep import *


FOLD_SUMMARY_KEYS = (
    "iptm",
    "ptm",
    "protein_iptm",
    "design_iptm",
    "design_iiptm",
    "design_to_target_iptm",
    "target_ptm",
    "design_ptm",
    "ligand_iptm",
    "interaction_pae",
    "min_interaction_pae",
    "min_design_to_target_pae",
    "boltz_confidence",
    "best_sample_idx",
)

REQUIRED_FOLD_SUMMARY_KEYS = (
    "iptm",
    "ptm",
    "min_interaction_pae",
    "min_design_to_target_pae",
)

ANALYSIS_SUMMARY_KEYS = (
    "bb_rmsd",
    "bb_rmsd_design",
    "bb_rmsd_target",
    "bb_rmsd_design_target",
    "bb_target_aligned_rmsd_design",
    "min_interaction_pae",
    "min_design_to_target_pae",
    "iptm",
    "ptm",
    "protein_iptm",
    "design_iptm",
    "design_iiptm",
    "design_to_target_iptm",
    "target_ptm",
    "design_ptm",
    "delta_sasa_refolded",
    "design_sasa_unbound_refolded",
    "design_sasa_bound_refolded",
    "plip_hbonds_refolded",
    "plip_saltbridge_refolded",
    "liability_score",
    "liability_num_violations",
)


class BoltzRefold(BaseStep):
    """BoltzGen partial-target protein binder co-fold step.

    The heavy lifting is delegated to boltzgen_peptide_refold_score.py so this
    Step uses the same target residue, binding residue, structure group, and MSA
    preparation logic as the standalone BoltzGen refold workflow.
    """

    @property
    def name(self) -> str:
        return "boltzrefold"

    @property
    def _default_pdb_input_key(self) -> str:
        return ""

    @property
    def pdb_to_add(self) -> Tuple[str, ...]:
        return (f"{self.metrics_prefix}best",)

    @property
    def metrics_to_add(self) -> Tuple[str, ...]:
        best_key = f"{self.metrics_prefix}best"
        return tuple(
            f"{best_key}{NEST_SEP}{key}" for key in REQUIRED_FOLD_SUMMARY_KEYS
        )

    @property
    def params_to_take(self) -> Tuple[str, ...]:
        keys = [
            f"{self.name}-prefix",
            f"{self.name}-pdb-input",
            f"{self.name}-script",
            f"{self.name}-env",
            f"{self.name}-python",
            f"{self.name}-command-prefix",
            f"{self.name}-output-stem",
            f"{self.name}-target-structure",
            f"{self.name}-target-chain",
            f"{self.name}-target-res-index",
            f"{self.name}-binding-res-index",
            f"{self.name}-target-msa",
            f"{self.name}-target-msa-res-index",
            f"{self.name}-max-msa-seqs",
            f"{self.name}-structure-groups",
            f"{self.name}-structure-group",
            f"{self.name}-binder-chain",
            f"{self.name}-cyclic",
            f"{self.name}-run-analysis",
            f"{self.name}-debug",
            f"{self.name}-num-workers",
            f"{self.name}-analysis-processes",
            f"{self.name}-accelerator",
            f"{self.name}-devices",
            f"{self.name}-precision",
            f"{self.name}-recycling-steps",
            f"{self.name}-sampling-steps",
            f"{self.name}-diffusion-samples",
            f"{self.name}-fold-checkpoint",
            f"{self.name}-moldir",
            f"{self.name}-liability-peptide-type",
            "cyclize_peptide",
        ]
        return tuple(keys)

    def __init__(self, settings: GlobalSettings):
        super().__init__(settings)
        adv = self.settings.adv
        adv.setdefault(
            f"{self.name}-script",
            "/hpf/projects/mkoziarski/zdeng/boltzgen/boltzgen_peptide_refold_score.py",
        )
        adv.setdefault(f"{self.name}-env", "boltzgen")
        adv.setdefault(f"{self.name}-output-stem", "boltzrefold")
        adv.setdefault(f"{self.name}-binder-chain", self._default_binder_chain())
        adv.setdefault(f"{self.name}-cyclic", bool(adv.get("cyclize_peptide", False)))
        adv.setdefault(f"{self.name}-run-analysis", False)
        adv.setdefault(f"{self.name}-debug", False)
        adv.setdefault(f"{self.name}-structure-groups", "all")
        adv.setdefault(f"{self.name}-num-workers", 1)
        adv.setdefault(f"{self.name}-analysis-processes", 4)
        adv.setdefault(f"{self.name}-accelerator", "gpu")
        adv.setdefault(f"{self.name}-devices", "1")
        adv.setdefault(f"{self.name}-precision", "bf16-mixed")
        adv.setdefault(f"{self.name}-recycling-steps", 3)
        adv.setdefault(f"{self.name}-sampling-steps", 200)
        adv.setdefault(f"{self.name}-diffusion-samples", 5)

    def process_record(self, input: DesignRecord | None = None) -> DesignRecord | None:
        if input is None:
            return None
        self._run_records([input])
        self.purge_record(input)
        return input

    def process_batch(
        self,
        input: DesignBatch | None = None,
        pdb_purge_stem: Optional[str] = None,
        pdb_to_take: str | None = None,
        metrics_prefix: str | None = None,
    ) -> DesignBatch | None:
        if input is None:
            return None
        if pdb_purge_stem is not None:
            self.config_pdb_purge(pdb_purge_stem)
        if metrics_prefix is not None:
            self.config_metrics_prefix(metrics_prefix)
        if pdb_to_take is not None:
            self.config_pdb_input_key(pdb_to_take)

        records = [
            record
            for record in input.records.values()
            if input.overwrite or not self.check_processed(record)
        ]
        if not records:
            return input

        self._run_records(records)
        for record in tqdm(records, desc=f"{self.name}:records"):
            self.purge_record(record)
            input.save_record(record.id)
        return input

    def purge_record(self, record: DesignRecord):
        if self.pdb_purge_dir is None:
            return
        key = f"{self.metrics_prefix}best"
        out_path = self.pdb_purge_dir / f"{record.id}.cif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if key in record.pdb_strs:
            record.purge_pdb(key, out_path)
        elif key in record.pdb_files:
            shutil.copyfile(record.pdb_files[key], out_path)
            record.pdb_files[key] = str(out_path)

    def _run_records(self, records: list[DesignRecord]) -> None:
        sample_to_record = self._write_binder_csv(records)
        output_dir = self._output_dir()
        cmd_list = self._build_command(output_dir / "binder_inputs.csv", output_dir)
        self._write_command(output_dir / "cmd.sh", cmd_list)

        subprocess.run(cmd_list, cwd=str(self._script_path().parent), check=True)
        summary = self._read_summary(output_dir)
        self._attach_results(summary, sample_to_record)

    def _attach_results(
        self,
        summary: dict[str, Any],
        sample_to_record: dict[str, DesignRecord],
    ) -> None:
        best_key = f"{self.metrics_prefix}best"
        analysis_key = f"{self.metrics_prefix}analysis"
        samples = summary.get("samples", [])
        for sample in samples:
            sample_id = sample.get("sample_id")
            if sample_id not in sample_to_record:
                continue
            record = sample_to_record[sample_id]
            folding = sample.get("folding") or {}
            refold_cif = folding.get("refold_cif")
            if refold_cif:
                refold_path = Path(refold_cif)
                if refold_path.exists():
                    record.pdb_files[best_key] = str(refold_path)
            for key in FOLD_SUMMARY_KEYS:
                if key in folding:
                    record.set_metrics(f"{best_key}{NEST_SEP}{key}", folding[key])
            for key in ("sample_id", "binder_chain", "cyclic"):
                if key in sample:
                    record.set_metrics(f"{best_key}{NEST_SEP}{key}", sample[key])
            prepared = sample.get("prepared") or {}
            for key in (
                "binder_chain_parsed",
                "num_tokens",
                "num_design_tokens",
                "num_binding_tokens",
            ):
                if key in prepared:
                    record.set_metrics(f"{best_key}{NEST_SEP}{key}", prepared[key])
            analysis = sample.get("analysis") or {}
            for key in ANALYSIS_SUMMARY_KEYS:
                if key in analysis:
                    record.set_metrics(f"{analysis_key}{NEST_SEP}{key}", analysis[key])

    def _build_command(self, binder_csv: Path, output_dir: Path) -> list[str]:
        adv = self.settings.adv
        cmd_list = self._python_command_prefix() + [str(self._script_path())]

        self._extend_arg(cmd_list, "--target-structure", self._target_structure())
        for chain in self._target_chains():
            self._extend_arg(cmd_list, "--target-chain", chain)
        self._extend_arg(
            cmd_list,
            "--target-res-index",
            self._format_spec(adv.get(f"{self.name}-target-res-index")),
        )
        for value in self._binding_residue_specs():
            self._extend_arg(cmd_list, "--binding-res-index", value)
        for value in self._target_msa_specs():
            self._extend_arg(cmd_list, "--target-msa", value)
        for value in self._target_msa_res_index_specs():
            self._extend_arg(cmd_list, "--target-msa-res-index", value)
        for value in self._structure_group_specs():
            self._extend_arg(cmd_list, "--structure-group", value)

        self._extend_arg(
            cmd_list,
            "--structure-groups",
            adv.get(f"{self.name}-structure-groups"),
        )
        self._extend_arg(cmd_list, "--binder-csv", binder_csv)
        self._extend_arg(cmd_list, "--binder-chain", adv.get(f"{self.name}-binder-chain"))
        self._extend_arg(cmd_list, "--output-dir", output_dir)
        self._extend_arg(cmd_list, "--accelerator", adv.get(f"{self.name}-accelerator"))
        self._extend_arg(cmd_list, "--devices", adv.get(f"{self.name}-devices"))
        self._extend_arg(cmd_list, "--precision", adv.get(f"{self.name}-precision"))
        self._extend_arg(cmd_list, "--num-workers", adv.get(f"{self.name}-num-workers"))
        self._extend_arg(
            cmd_list,
            "--analysis-processes",
            adv.get(f"{self.name}-analysis-processes"),
        )
        self._extend_arg(
            cmd_list,
            "--recycling-steps",
            adv.get(f"{self.name}-recycling-steps"),
        )
        self._extend_arg(
            cmd_list,
            "--sampling-steps",
            adv.get(f"{self.name}-sampling-steps"),
        )
        self._extend_arg(
            cmd_list,
            "--diffusion-samples",
            adv.get(f"{self.name}-diffusion-samples"),
        )
        self._extend_arg(cmd_list, "--max-msa-seqs", adv.get(f"{self.name}-max-msa-seqs"))
        self._extend_arg(
            cmd_list,
            "--fold-checkpoint",
            adv.get(f"{self.name}-fold-checkpoint"),
        )
        self._extend_arg(cmd_list, "--moldir", adv.get(f"{self.name}-moldir"))
        self._extend_arg(
            cmd_list,
            "--liability-peptide-type",
            adv.get(f"{self.name}-liability-peptide-type"),
        )

        if adv.get(f"{self.name}-cyclic"):
            cmd_list.append("--cyclic")
        if adv.get(f"{self.name}-run-analysis"):
            cmd_list.append("--run-analysis")
        if adv.get(f"{self.name}-debug"):
            cmd_list.append("--debug")
        return cmd_list

    def _write_binder_csv(self, records: list[DesignRecord]) -> dict[str, DesignRecord]:
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_to_record: dict[str, DesignRecord] = {}
        with (output_dir / "binder_inputs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "binder_sequence", "binder_chain", "cyclic"],
            )
            writer.writeheader()
            for sample_id, record in self._sample_ids(records).items():
                sample_to_record[sample_id] = record
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "binder_sequence": record.sequence,
                        "binder_chain": self.settings.adv.get(
                            f"{self.name}-binder-chain"
                        ),
                        "cyclic": str(bool(self.settings.adv.get(f"{self.name}-cyclic"))),
                    }
                )
        return sample_to_record

    def _sample_ids(self, records: list[DesignRecord]) -> dict[str, DesignRecord]:
        used: set[str] = set()
        sample_to_record: dict[str, DesignRecord] = {}
        for idx, record in enumerate(records):
            base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record.id)).strip("._-")
            if not base:
                base = f"sample_{idx:04d}"
            sample_id = base
            suffix = 1
            while sample_id in used:
                sample_id = f"{base}_{suffix}"
                suffix += 1
            used.add(sample_id)
            sample_to_record[sample_id] = record
        return sample_to_record

    def _read_summary(self, output_dir: Path) -> dict[str, Any]:
        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"BoltzRefold did not produce {summary_path}")
        return json.loads(summary_path.read_text())

    def _write_command(self, path: Path, cmd_list: list[str]) -> None:
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"cd {shlex.quote(str(self._script_path().parent))}\n"
            f"{shlex.join(cmd_list)}\n"
        )
        path.chmod(0o755)

    def _python_command_prefix(self) -> list[str]:
        adv = self.settings.adv
        command_prefix = adv.get(f"{self.name}-command-prefix")
        if command_prefix:
            return self._split_command_prefix(command_prefix)

        python_executable = adv.get(f"{self.name}-python")
        if python_executable:
            return self._split_command_prefix(python_executable)

        env_name = adv.get(f"{self.name}-env")
        if env_name:
            return ["conda", "run", "-n", str(env_name), "python"]
        return [sys.executable]

    def _split_command_prefix(self, value: Any) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item)]
        return shlex.split(str(value))

    def _script_path(self) -> Path:
        return self._resolve_path(self.settings.adv[f"{self.name}-script"], must_exist=True)

    def _output_dir(self) -> Path:
        stem = self.settings.adv.get(f"{self.name}-output-stem", "boltzrefold")
        path = Path(stem).expanduser()
        if not path.is_absolute():
            path = Path(self.settings.binder_settings.design_path) / path
        return path.resolve()

    def _target_structure(self) -> Path:
        adv = self.settings.adv
        value = adv.get(f"{self.name}-target-structure")
        if value is None:
            target = self.settings.target_settings
            value = getattr(target, "full_target_pdb", None) or getattr(
                target, "starting_pdb", None
            )
        if value is None:
            raise ValueError(
                f"{self.name}-target-structure is required when target settings "
                "do not define full_target_pdb or starting_pdb."
            )
        return self._resolve_path(value, must_exist=True)

    def _target_chains(self) -> list[str]:
        adv = self.settings.adv
        value = adv.get(f"{self.name}-target-chain")
        if value is None:
            target = self.settings.target_settings
            value = getattr(target, "full_target_chain", None) or getattr(
                target, "chains", None
            )
        return self._normalize_list(value)

    def _default_binder_chain(self) -> str:
        target = self.settings.target_settings
        return (
            getattr(target, "new_binder_chain", None)
            or getattr(target, "full_binder_chain", None)
            or "Z"
        )

    def _binding_residue_specs(self) -> list[str]:
        value = self.settings.adv.get(f"{self.name}-binding-res-index")
        if value:
            return self._normalize_chain_specs(value)
        hotspots = getattr(self.settings.target_settings, "target_hotspot_residues", None)
        if not hotspots:
            return []
        return self._hotspot_binding_specs(hotspots)

    def _hotspot_binding_specs(self, hotspots: Iterable[Any] | str) -> list[str]:
        chains = self._target_chains()
        chain_maps = self._target_chain_residue_maps(chains)
        grouped: dict[str, list[str]] = {}
        for item in self._normalize_list(hotspots):
            chain, residue = self._parse_hotspot(item, chains)
            if chain not in chain_maps:
                continue
            mapped = chain_maps[chain].get(residue)
            if mapped is not None:
                grouped.setdefault(chain, []).append(mapped)
        if not grouped:
            return []
        if len(chains) <= 1:
            return [",".join(grouped[next(iter(grouped))])]
        return [f"{chain}:{','.join(values)}" for chain, values in grouped.items()]

    def _parse_hotspot(self, value: Any, chains: list[str]) -> tuple[str, str]:
        text = str(value).strip()
        if ":" in text:
            chain, residue = text.split(":", 1)
        elif text and text[0].isalpha() and text[1:].strip():
            chain, residue = text[0], text[1:]
        elif len(chains) == 1:
            chain, residue = chains[0], text
        else:
            raise ValueError(
                f"Ambiguous hotspot {value!r}; use CHAIN:RESI for multi-chain targets."
            )
        return chain.strip(), residue.strip()

    def _target_chain_residue_maps(self, chains: list[str]) -> dict[str, dict[str, str]]:
        if not chains:
            raise ValueError("Cannot map hotspots without an explicit target chain.")
        obj = f"_boltzrefold_target_{id(self)}"
        structure = self._target_structure()
        cmd.delete(obj)
        cmd.load(str(structure), obj)
        cmd.remove(f'{obj} and not (alt "" or alt A)')
        try:
            maps: dict[str, dict[str, str]] = {}
            for chain in chains:
                residues: list[str] = []
                space = {"residues": residues}
                cmd.iterate(
                    f"{obj} and chain {chain} and polymer.protein and name CA",
                    "residues.append(resi)",
                    space=space,
                )
                maps[chain] = {
                    str(residue): str(idx)
                    for idx, residue in enumerate(residues, start=1)
                }
            return maps
        finally:
            cmd.delete(obj)

    def _target_msa_specs(self) -> list[str]:
        return self._normalize_path_specs(self.settings.adv.get(f"{self.name}-target-msa"))

    def _target_msa_res_index_specs(self) -> list[str]:
        value = self.settings.adv.get(f"{self.name}-target-msa-res-index")
        return self._normalize_chain_specs(value)

    def _structure_group_specs(self) -> list[str]:
        return self._normalize_chain_specs(
            self.settings.adv.get(f"{self.name}-structure-group")
        )

    def _normalize_path_specs(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, dict):
            return [
                f"{chain}:{self._resolve_path(path, must_exist=True)}"
                for chain, path in value.items()
            ]
        specs = []
        for item in self._normalize_spec_list(value):
            text = str(item).strip()
            if not text:
                continue
            if ":" in text and not Path(text).exists():
                chain, path = text.split(":", 1)
                specs.append(f"{chain}:{self._resolve_path(path, must_exist=True)}")
            else:
                specs.append(str(self._resolve_path(text, must_exist=True)))
        return specs

    def _normalize_chain_specs(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, dict):
            return [f"{chain}:{self._format_spec(spec)}" for chain, spec in value.items()]
        return [
            str(item).strip()
            for item in self._normalize_spec_list(value)
            if str(item).strip()
        ]

    def _format_spec(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            return ",".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    def _normalize_spec_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(";") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    def _normalize_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            sep = ";" if ";" in value else ","
            if sep == "," and ".." in value:
                return [value.strip()]
            return [item.strip() for item in value.split(sep) if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    def _resolve_path(self, value: Any, must_exist: bool = False) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path

    def _extend_arg(self, cmd_list: list[str], flag: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            if value:
                cmd_list.append(flag)
            return
        text = str(value)
        if text == "":
            return
        cmd_list.extend([flag, text])
