#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from boltzgen.data import const
from boltzgen.data.data import MSA, MSADeletion, MSAResidue, MSASequence
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.write.mmcif import to_mmcif
import boltzgen.resources.main as boltzgen_resource_main


DEFAULT_FOLD_CKPT = "huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt"
DEFAULT_MOLDIR = "huggingface:boltzgen/inference-data:mols.zip"

FOLD_SUMMARY_KEYS = [
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
]

ANALYSIS_SUMMARY_KEYS = [
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
]


@dataclass(frozen=True)
class BinderSpec:
    """One binder sequence and its requested BoltzGen chain behavior."""

    sample_id: str
    sequence: str
    chain: str
    cyclic: bool


@dataclass(frozen=True)
class TargetMSASpec:
    """An MSA source and optional residue slice for one target chain."""

    chain: str
    path: Path
    res_index: str | None


TOKEN_ID_TO_NAME = {token_id: token_name for token_name, token_id in const.token_ids.items()}
TOKEN_NAME_TO_PROTEIN_LETTER = {
    token_name: letter for letter, token_name in const.prot_letter_to_token.items()
}


def token_id_to_protein_letter(token_id: int) -> str:
    """Translate one BoltzGen residue token into a protein letter."""
    token_name = TOKEN_ID_TO_NAME.get(int(token_id), "UNK")
    return TOKEN_NAME_TO_PROTEIN_LETTER.get(
        token_name, "X" if token_name == "UNK" else "-"
    )


def protein_residue_sequence(residues: np.ndarray) -> str:
    """Recover a one-letter sequence from BoltzGen residue records."""
    return "".join(token_id_to_protein_letter(residue["res_type"]) for residue in residues)


def parse_args() -> argparse.Namespace:
    """Parse standalone co-fold preparation, execution, and analysis options."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and optionally run the BoltzGen protein/peptide-binder refold "
            "step as a standalone scoring job. The prepared input matches the "
            "folding step used after inverse folding: target residues are used as "
            "templates and the binder chain is marked as the design/refold chain."
        )
    )
    parser.add_argument("--target-structure", required=True, type=Path)
    parser.add_argument(
        "--target-chain",
        action="append",
        default=None,
        help=(
            "Target chain id to include. Can be repeated, e.g. "
            "--target-chain A --target-chain B. Default keeps all chains."
        ),
    )
    parser.add_argument(
        "--target-res-index",
        default=None,
        help=(
            "Optional BoltzGen res_index range/list for --target-chain, e.g. "
            "'15,16,20..25'. Indices are 1-based in the chain, matching BoltzGen YAML."
        ),
    )
    parser.add_argument(
        "--binding-res-index",
        action="append",
        default=None,
        help=(
            "Optional BoltzGen binding_types.chain.binding range/list for "
            "--target-chain. For multiple target chains, repeat as "
            "'CHAIN:residue-list', e.g. --binding-res-index A:1,2,3."
        ),
    )
    parser.add_argument(
        "--target-msa",
        action="append",
        default=None,
        help=(
            "Local target-chain MSA to use during folding. Can be passed as PATH "
            "when there is exactly one --target-chain, or as CHAIN:PATH / "
            "CHAIN=PATH. Repeat for multi-chain targets. CSV files with "
            "key,sequence columns and A3M/A3M.GZ files are supported."
        ),
    )
    parser.add_argument(
        "--target-msa-res-index",
        action="append",
        default=None,
        help=(
            "Optional full-MSA column selection as CHAIN:residue-list or "
            "CHAIN=residue-list. If omitted for a single target chain, "
            "--target-res-index is reused, so full-chain MSAs can be supplied "
            "for segment refolds."
        ),
    )
    parser.add_argument(
        "--max-msa-seqs",
        type=int,
        default=None,
        help=(
            "Maximum MSA sequences passed to BoltzGen featurization. Defaults to "
            "8192 when --target-msa is provided, otherwise leaves the BoltzGen "
            "single-sequence default unchanged."
        ),
    )
    parser.add_argument(
        "--structure-groups",
        default="all",
        help=(
            "Target structure grouping passed to the BoltzGen file schema. "
            "Use 'all' to keep all target residues in one fixed structure group "
            "(default), or 'none' to set all target residues to visibility 0."
        ),
    )
    parser.add_argument(
        "--structure-group",
        action="append",
        default=None,
        help=(
            "Fine-grained structure group entry as CHAIN:VISIBILITY[:RES_INDEX]. "
            "Can be repeated. Example: --structure-group A:1 --structure-group B:1."
        ),
    )
    parser.add_argument(
        "--binder-sequence",
        action="append",
        default=None,
        help=(
            "Binder protein sequence. Can be repeated for batch mode. "
            "If repeated, provide matching repeated --sample-id values or let the script generate ids."
        ),
    )
    parser.add_argument(
        "--binder-csv",
        type=Path,
        default=None,
        help=(
            "CSV/TSV with sample_id,binder_sequence columns. Optional columns: "
            "binder_chain,cyclic."
        ),
    )
    parser.add_argument("--binder-chain", default="Z")
    parser.add_argument(
        "--cyclic",
        action="store_true",
        help="Add BoltzGen cyclic polymer connectivity for the binder chain.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Sample id. Can be repeated with repeated --binder-sequence.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the design_dir input and summary; do not run inference.",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run BoltzGen analysis.yaml after folding and include aggregate metrics.",
    )
    parser.add_argument(
        "--skip-folding",
        action="store_true",
        help="Reuse existing fold_out_npz/refold_cif outputs under --output-dir.",
    )
    parser.add_argument(
        "--reuse-existing-folds",
        action="store_true",
        help="Skip individual samples whose fold_out_npz/refold_cif outputs already exist.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep prepared inputs, fold NPZs, analysis scratch files, schemas, and logs.",
    )
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "DataLoader workers for folding/analysis. Default matches the BoltzGen "
            "refold configs; molecule zip inputs are automatically capped at 1."
        ),
    )
    parser.add_argument("--analysis-processes", type=int, default=4)
    parser.add_argument(
        "--no-delta-sasa-refolded",
        action="store_true",
        help=(
            "Disable delta_sasa_refolded in analysis.yaml. This keeps backbone "
            "RMSD and iPTM metrics but avoids known Biotite atom-mask failures "
            "on some refolded CATH structures."
        ),
    )
    parser.add_argument(
        "--no-noncovalents-refolded",
        action="store_true",
        help=(
            "Disable noncovalents_refolded in analysis.yaml. This keeps backbone "
            "RMSD and iPTM metrics but avoids atom-mask failures in PLIP-style "
            "interaction analysis on some refolded CATH structures."
        ),
    )
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--diffusion-samples", type=int, default=5)
    parser.add_argument("--fold-checkpoint", default=DEFAULT_FOLD_CKPT)
    parser.add_argument("--moldir", default=DEFAULT_MOLDIR)
    parser.add_argument(
        "--liability-peptide-type",
        choices=["linear", "cyclic"],
        default="linear",
        help="Passed to analysis.yaml when --run-analysis is set. Default matches generated peptide-anything configs.",
    )
    return parser.parse_args()


def resolve_artifact(spec: str, repo_type: str) -> Path:
    """Resolve a local path or download a Hugging Face artifact."""
    if not spec.startswith("huggingface:"):
        return Path(spec).expanduser().resolve()

    _, repo_id, filename = spec.split(":", 2)
    if repo_type == "dataset":
        snapshot_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                allow_patterns=filename,
                library_name="boltzgen",
            )
        )
        return snapshot_dir / filename

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            library_name="boltzgen",
        )
    ).resolve()


def parse_bool(value: Any) -> bool:
    """Parse common CSV and command-line boolean spellings strictly."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def read_binder_csv(path: Path, args: argparse.Namespace) -> list[BinderSpec]:
    """Read binder specifications from a CSV or TSV manifest."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    specs: list[BinderSpec] = []
    with path.expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        for row_idx, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or row.get("id") or "").strip()
            sequence = (
                row.get("binder_sequence") or row.get("sequence") or ""
            ).strip()
            if not sample_id or not sequence:
                raise ValueError(
                    f"{path}:{row_idx} must define sample_id and binder_sequence."
                )
            chain = (row.get("binder_chain") or args.binder_chain).strip()
            cyclic_raw = row.get("cyclic")
            cyclic = args.cyclic if cyclic_raw is None else parse_bool(cyclic_raw)
            specs.append(
                BinderSpec(
                    sample_id=sample_id,
                    sequence=sequence,
                    chain=chain,
                    cyclic=cyclic,
                )
            )
    return specs


def collect_binder_specs(args: argparse.Namespace) -> list[BinderSpec]:
    """Collect and validate binder specifications from all CLI sources."""
    specs: list[BinderSpec] = []
    if args.binder_csv is not None:
        specs.extend(read_binder_csv(args.binder_csv, args))

    if args.binder_sequence:
        sequences = [seq.strip() for seq in args.binder_sequence]
        sample_ids = args.sample_id or []
        if sample_ids and len(sample_ids) != len(sequences):
            raise ValueError(
                "When using repeated --binder-sequence, repeated --sample-id must "
                "have the same count."
            )
        if not sample_ids:
            sample_ids = (
                ["refold_input"]
                if len(sequences) == 1
                else [f"sample{i:04d}" for i in range(len(sequences))]
            )
        for sample_id, sequence in zip(sample_ids, sequences):
            specs.append(
                BinderSpec(
                    sample_id=sample_id,
                    sequence=sequence,
                    chain=args.binder_chain,
                    cyclic=args.cyclic,
                )
            )

    if not specs:
        raise ValueError("Provide --binder-sequence or --binder-csv.")

    sample_ids = [spec.sample_id for spec in specs]
    duplicates = sorted(
        sample_id for sample_id in set(sample_ids) if sample_ids.count(sample_id) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate sample ids: {', '.join(duplicates)}")
    return specs


def normalize_target_chains(args: argparse.Namespace) -> list[str] | None:
    """Normalize repeated or comma-separated target-chain arguments."""
    values = args.target_chain
    if values is None:
        return None
    if not isinstance(values, list):
        values = [values]

    chains: list[str] = []
    for value in values:
        chains.extend(
            chain.strip() for chain in str(value).split(",") if chain.strip()
        )
    if not chains:
        return None

    seen: set[str] = set()
    duplicates = []
    for chain in chains:
        if chain in seen:
            duplicates.append(chain)
        seen.add(chain)
    if duplicates:
        raise ValueError(f"Duplicate target chains: {', '.join(sorted(duplicates))}")
    return chains


def normalize_binding_types(
    args: argparse.Namespace, target_chains: list[str] | None
) -> list[dict[str, dict[str, str]]]:
    """Convert binding-residue arguments into BoltzGen schema blocks."""
    values = args.binding_res_index
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]

    binding_types = []
    for value in values:
        raw = str(value).strip()
        if not raw:
            continue
        if ":" in raw:
            chain_id, binding = raw.split(":", 1)
            chain_id = chain_id.strip()
            binding = binding.strip()
        else:
            if target_chains is None:
                raise ValueError("--binding-res-index requires --target-chain")
            if len(target_chains) != 1:
                raise ValueError(
                    "Chainless --binding-res-index is only valid with one "
                    "--target-chain. Use CHAIN:residue-list for multi-chain targets."
                )
            chain_id = target_chains[0]
            binding = raw
        if not chain_id or not binding:
            raise ValueError(f"Invalid --binding-res-index value: {value}")
        binding_types.append({"chain": {"id": chain_id, "binding": binding}})

    return binding_types


def normalize_structure_groups(args: argparse.Namespace) -> str | list[dict[str, Any]]:
    """Convert visibility arguments into BoltzGen structure groups."""
    if args.structure_group:
        groups: list[dict[str, Any]] = []
        for value in args.structure_group:
            parts = str(value).split(":", 2)
            if len(parts) < 2:
                raise ValueError(
                    "--structure-group must be CHAIN:VISIBILITY[:RES_INDEX]"
                )
            chain_id = parts[0].strip()
            visibility_raw = parts[1].strip()
            res_index = parts[2].strip() if len(parts) == 3 else None
            if not chain_id or not visibility_raw:
                raise ValueError(f"Invalid --structure-group value: {value}")
            group: dict[str, Any] = {
                "id": chain_id,
                "visibility": int(visibility_raw),
            }
            if res_index:
                group["res_index"] = res_index
            groups.append({"group": group})
        return groups

    normalized = str(args.structure_groups).strip().lower()
    if normalized == "all":
        return "all"
    if normalized in {"none", "0"}:
        return [{"group": {"id": "all", "visibility": 0}}]
    raise ValueError(
        "--structure-groups must be 'all' or 'none'. For custom groups, use "
        "repeated --structure-group CHAIN:VISIBILITY[:RES_INDEX]."
    )


def split_chain_value_spec(
    value: str,
    option_name: str,
    target_chains: list[str] | None,
) -> tuple[str, str]:
    """Split a chain-qualified CLI value and infer the sole target when safe."""
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"Empty {option_name} value.")

    if "=" in raw:
        chain_id, payload = raw.split("=", 1)
    elif ":" in raw:
        chain_id, payload = raw.split(":", 1)
    else:
        if target_chains is None or len(target_chains) != 1:
            raise ValueError(
                f"Unqualified {option_name} is only valid with exactly one "
                "--target-chain. Use CHAIN:VALUE for multi-chain targets."
            )
        chain_id, payload = target_chains[0], raw

    chain_id = chain_id.strip()
    payload = payload.strip()
    if not chain_id or not payload:
        raise ValueError(f"Invalid {option_name} value: {value}")
    return chain_id, payload


def parse_res_index_positions(res_index: str, sequence_length: int | None = None) -> list[int]:
    """Expand BoltzGen one-based residue ranges into zero-based positions."""
    positions: list[int] = []
    for part in str(res_index).split(","):
        spec = part.strip()
        if not spec:
            continue
        if ".." in spec:
            start_raw, end_raw = spec.split("..", 1)
            start = int(start_raw) if start_raw else 1
            if end_raw:
                end = int(end_raw)
            else:
                if sequence_length is None:
                    raise ValueError(
                        f"Open-ended residue range '{spec}' requires sequence length."
                    )
                end = sequence_length
            if start < 1 or end < start:
                raise ValueError(f"Invalid residue range: {spec}")
            positions.extend(range(start - 1, end))
        else:
            idx = int(spec)
            if idx < 1:
                raise ValueError("Residue indices are 1-based; found 0.")
            positions.append(idx - 1)
    return positions


def normalize_target_msa_specs(
    args: argparse.Namespace,
    target_chains: list[str] | None,
) -> dict[str, TargetMSASpec]:
    """Resolve chain-specific target MSA inputs and residue selections."""
    if not args.target_msa:
        return {}

    res_index_by_chain: dict[str, str] = {}
    for value in args.target_msa_res_index or []:
        chain_id, res_index = split_chain_value_spec(
            value, "--target-msa-res-index", target_chains
        )
        res_index_by_chain[chain_id] = res_index

    if (
        args.target_res_index is not None
        and target_chains is not None
        and len(target_chains) == 1
    ):
        res_index_by_chain.setdefault(target_chains[0], args.target_res_index)

    specs: dict[str, TargetMSASpec] = {}
    for value in args.target_msa:
        chain_id, path_value = split_chain_value_spec(
            value, "--target-msa", target_chains
        )
        if chain_id in specs:
            raise ValueError(f"Duplicate --target-msa for chain {chain_id}.")
        specs[chain_id] = TargetMSASpec(
            chain=chain_id,
            path=Path(path_value).expanduser().resolve(),
            res_index=res_index_by_chain.get(chain_id),
        )
    return specs


def iter_msa_sequences(path: Path) -> list[tuple[int, str]]:
    """Read taxonomy and aligned sequences from supported MSA formats."""
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows: list[tuple[int, str]] = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None or "sequence" not in reader.fieldnames:
                raise ValueError(f"{path} must contain a sequence column.")
            for row in reader:
                sequence = (row.get("sequence") or "").strip()
                if not sequence:
                    continue
                key = (row.get("key") or row.get("taxonomy") or "-1").strip()
                try:
                    taxonomy = int(key)
                except ValueError:
                    taxonomy = -1
                rows.append((taxonomy, sequence))
        return rows

    if ".a3m" in suffixes:
        rows = []
        taxonomy = -1
        opener = __import__("gzip").open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(">"):
                    taxonomy = -1
                    continue
                rows.append((taxonomy, line))
        return rows

    raise ValueError(f"Unsupported MSA format for {path}; use CSV/TSV/A3M/A3M.GZ.")


def select_msa_columns(sequence: str, positions: list[int] | None) -> str:
    """Select residue columns while retaining A3M insertion semantics."""
    if positions is None:
        return sequence
    aligned = [char for char in sequence if not char.islower()]
    if positions and max(positions) >= len(aligned):
        raise ValueError(
            f"MSA sequence length {len(aligned)} is shorter than requested "
            f"residue index {max(positions) + 1}."
        )
    return "".join(aligned[idx] for idx in positions)


def hamming_mismatches(left: str, right: str) -> int:
    """Count positional mismatches, using the longer length on mismatch."""
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(a != b for a, b in zip(left, right))


def select_aligned_sequence(aligned: list[str], positions: list[int]) -> str:
    """Extract uppercase residues at selected aligned positions."""
    return "".join(aligned[idx].upper() for idx in positions)


def greedy_subsequence_positions(aligned: list[str], expected: str) -> list[int] | None:
    """Locate an expected sequence as an ordered aligned subsequence."""
    positions: list[int] = []
    cursor = 0
    expected = expected.upper()
    for idx, char in enumerate(aligned):
        if cursor >= len(expected):
            break
        if char.upper() == expected[cursor]:
            positions.append(idx)
            cursor += 1
    if cursor == len(expected):
        return positions
    return None


def resolve_msa_positions(
    rows: list[tuple[int, str]],
    res_index: str | None,
    expected_sequence: str | None,
) -> tuple[list[int] | None, dict[str, Any]]:
    """Choose MSA columns and report how they were matched to the target."""
    aligned = [char for char in rows[0][1] if not char.islower()]
    info: dict[str, Any] = {
        "first_msa_aligned_length": len(aligned),
        "position_strategy": "none",
    }
    positions = None
    if res_index:
        positions = parse_res_index_positions(res_index, len(aligned))
        info["position_strategy"] = "res_index"
        info["res_index_mismatches"] = None

    if expected_sequence is None:
        return positions, info

    expected = expected_sequence.upper()
    info["expected_sequence_length"] = len(expected)
    if positions is not None:
        selected = select_aligned_sequence(aligned, positions)
        mismatches = hamming_mismatches(selected, expected)
        info["res_index_mismatches"] = mismatches
        if mismatches == 0:
            return positions, info

        best_positions = positions
        best_offset = 0
        best_mismatches = mismatches
        for offset in range(-200, 201):
            shifted = [idx + offset for idx in positions]
            if min(shifted) < 0 or max(shifted) >= len(aligned):
                continue
            shifted_selected = select_aligned_sequence(aligned, shifted)
            shifted_mismatches = hamming_mismatches(shifted_selected, expected)
            if shifted_mismatches < best_mismatches:
                best_mismatches = shifted_mismatches
                best_offset = offset
                best_positions = shifted
                if best_mismatches == 0:
                    break
        if best_mismatches < mismatches:
            info["position_strategy"] = "res_index_offset"
            info["position_offset"] = best_offset
            info["first_sequence_mismatches"] = best_mismatches
            return best_positions, info

    greedy_positions = greedy_subsequence_positions(aligned, expected)
    if greedy_positions is not None:
        info["position_strategy"] = "first_sequence_subsequence"
        info["first_sequence_mismatches"] = 0
        return greedy_positions, info

    info["first_sequence_mismatches"] = (
        hamming_mismatches(select_aligned_sequence(aligned, positions), expected)
        if positions is not None
        else None
    )
    return positions, info


def build_msa(
    path: Path,
    res_index: str | None,
    max_seqs: int | None,
    expected_sequence: str | None = None,
) -> tuple[MSA, int, dict[str, Any]]:
    """Build a deduplicated BoltzGen MSA with optional target-column slicing."""
    rows = iter_msa_sequences(path)
    if not rows:
        raise ValueError(f"No MSA sequences found in {path}.")
    positions, position_info = resolve_msa_positions(rows, res_index, expected_sequence)

    visited: set[str] = set()
    sequences = []
    deletions = []
    residues = []
    seq_idx = 0
    for taxonomy, raw_sequence in rows:
        sequence = select_msa_columns(raw_sequence, positions)
        dedupe_key = "".join(
            char for char in sequence.upper() if char != "-" and not char.islower()
        )
        if dedupe_key in visited:
            continue
        visited.add(dedupe_key)

        residue = []
        deletion = []
        insertion_count = 0
        res_idx = 0
        for char in sequence:
            if char != "-" and char.islower():
                insertion_count += 1
                continue
            token_name = const.prot_letter_to_token.get(char.upper(), "UNK")
            residue.append(const.token_ids[token_name])
            if insertion_count > 0:
                deletion.append((res_idx, insertion_count))
                insertion_count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)
        del_start = len(deletions)
        del_end = del_start + len(deletion)
        sequences.append((seq_idx, taxonomy, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)
        seq_idx += 1
        if max_seqs is not None and seq_idx >= max_seqs:
            break

    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa, len(rows), position_info


def effective_max_msa_seqs(args: argparse.Namespace) -> int | None:
    """Resolve the explicit or target-MSA-specific sequence cap."""
    if args.max_msa_seqs is not None:
        if args.max_msa_seqs < 1:
            raise ValueError("--max-msa-seqs must be positive.")
        return args.max_msa_seqs
    if args.target_msa:
        return 8192
    return None


def write_target_msa_sidecars(
    args: argparse.Namespace,
    design_dir: Path,
    target_msa_specs: dict[str, TargetMSASpec],
    target_chain_sequences: dict[str, str],
) -> dict[str, Any]:
    """Materialize per-chain MSA files and their runtime configuration."""
    if not target_msa_specs:
        return {}

    max_seqs = effective_max_msa_seqs(args)
    msa_dir = design_dir / "target_msas"
    msa_dir.mkdir(parents=True, exist_ok=True)
    chains: dict[str, dict[str, Any]] = {}
    for chain_id, spec in target_msa_specs.items():
        msa, num_input_sequences, position_info = build_msa(
            path=spec.path,
            res_index=spec.res_index,
            max_seqs=max_seqs,
            expected_sequence=target_chain_sequences.get(chain_id),
        )
        out_path = msa_dir / f"{chain_id}.npz"
        msa.dump(out_path)
        chains[chain_id] = {
            "source": str(spec.path),
            "path": str(out_path),
            "res_index": spec.res_index,
            "num_input_sequences": num_input_sequences,
            "num_used_sequences": int(len(msa.sequences)),
            "first_sequence_length": int(
                msa.sequences[0]["res_end"] - msa.sequences[0]["res_start"]
            ),
            **position_info,
        }

    config = {
        "max_msa_seqs": max_seqs,
        "chains": chains,
    }
    (design_dir / "target_msa_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    return config


def install_generated_msa_patch(msa_config_path: Path) -> None:
    """Install the runtime hook that injects supplied MSAs into BoltzGen."""
    import boltzgen.data.feature.featurizer as featurizer_module

    if getattr(featurizer_module, "_peptide_refold_msa_patch", False):
        featurizer_module._peptide_refold_msa_config_path = msa_config_path
        return

    original_construct_paired_msa = featurizer_module.construct_paired_msa
    msa_cache: dict[str, MSA] = {}

    def load_config() -> dict[str, Any]:
        """Load the active generated-MSA configuration when present."""
        config_path = getattr(
            featurizer_module,
            "_peptide_refold_msa_config_path",
            msa_config_path,
        )
        if not Path(config_path).exists():
            return {}
        return json.loads(Path(config_path).read_text())

    def compatible_msa(msa: MSA, residues: np.ndarray) -> MSA | None:
        """Return an MSA compatible with parsed residues, patching MET/UNK."""
        first = msa.sequences[0]
        first_residues = msa.residues[first["res_start"] : first["res_end"]]
        if len(residues) != len(first_residues):
            return None
        mismatches = residues["res_type"] != first_residues["res_type"]
        if mismatches.sum().item() == 0:
            return msa
        idx = np.where(mismatches)[0]
        is_met = residues["res_type"][idx] == const.token_ids["MET"]
        is_unk = residues["res_type"][idx] == const.token_ids["UNK"]
        is_msa_unk = first_residues["res_type"][idx] == const.token_ids["UNK"]
        if (np.all(is_met) and np.all(is_msa_unk)) or np.all(is_unk):
            patched = replace(
                msa,
                residues=msa.residues.copy(),
                deletions=msa.deletions.copy(),
                sequences=msa.sequences.copy(),
            )
            patched.residues[first["res_start"] : first["res_end"]]["res_type"] = (
                residues["res_type"]
            )
            return patched
        return None

    def construct_paired_msa_with_targets(data, random, max_seqs, *args, **kwargs):
        """Inject matching target-chain MSAs before standard paired featurization."""
        config = load_config()
        if not config or data.msa:
            return original_construct_paired_msa(
                data, random, max_seqs, *args, **kwargs
            )

        chain_configs = config.get("chains", {})
        msa_by_asym: dict[int, MSA] = {}
        msa_chain_asym_ids: set[int] = set()
        for chain in data.structure.chains:
            chain_name = str(chain["name"])
            chain_cfg = chain_configs.get(chain_name)
            if chain_cfg is None:
                continue
            msa_path = chain_cfg["path"]
            if msa_path not in msa_cache:
                msa_cache[msa_path] = MSA.load(Path(msa_path))
            start = int(chain["res_idx"])
            end = start + int(chain["res_num"])
            msa = compatible_msa(msa_cache[msa_path], data.structure.residues[start:end])
            if msa is None:
                print(
                    "Warning: supplied target MSA for chain "
                    f"{chain_name} does not match parsed structure; using dummy MSA."
                )
                continue
            asym_id = int(chain["asym_id"])
            msa_by_asym[asym_id] = msa
            msa_chain_asym_ids.add(asym_id)

        if not msa_by_asym:
            return original_construct_paired_msa(
                data, random, max_seqs, *args, **kwargs
            )
        patched_tokens = data.tokens.copy()
        for asym_id in msa_chain_asym_ids:
            mask = patched_tokens["asym_id"] == asym_id
            original_res_idx = patched_tokens["res_idx"][mask]
            ordered_res_idx = list(dict.fromkeys(original_res_idx.tolist()))
            idx_map = {res_idx: idx for idx, res_idx in enumerate(ordered_res_idx)}
            patched_tokens["res_idx"][mask] = np.array(
                [idx_map[res_idx] for res_idx in original_res_idx],
                dtype=patched_tokens["res_idx"].dtype,
            )

        return original_construct_paired_msa(
            replace(data, tokens=patched_tokens, msa=msa_by_asym),
            random,
            max_seqs,
            *args,
            **kwargs,
        )

    featurizer_module.construct_paired_msa = construct_paired_msa_with_targets
    featurizer_module._peptide_refold_msa_config_path = msa_config_path
    featurizer_module._peptide_refold_msa_patch = True


def build_schema(
    args: argparse.Namespace,
    target_structure: Path,
    binder: BinderSpec,
) -> dict[str, Any]:
    """Build one BoltzGen target-plus-binder input schema."""
    file_block: dict[str, Any] = {"path": str(target_structure)}
    target_chains = normalize_target_chains(args)
    target_msa_specs = normalize_target_msa_specs(args, target_chains)

    if target_chains is not None:
        if args.target_res_index is not None and len(target_chains) != 1:
            raise ValueError("--target-res-index currently supports one target chain")
        include = []
        for target_chain in target_chains:
            chain_block: dict[str, Any] = {"id": target_chain}
            if args.target_res_index is not None:
                chain_block["res_index"] = args.target_res_index
            if target_chain in target_msa_specs:
                chain_block["msa"] = str(target_msa_specs[target_chain].path)
            include.append({"chain": chain_block})
        file_block["include"] = include
    elif args.target_res_index is not None:
        raise ValueError("--target-res-index requires --target-chain")

    file_block["structure_groups"] = normalize_structure_groups(args)

    binding_types = normalize_binding_types(args, target_chains)
    if binding_types:
        file_block["binding_types"] = binding_types

    binder_block: dict[str, Any] = {
        "id": binder.chain,
        "sequence": binder.sequence,
    }
    if binder.cyclic:
        binder_block["cyclic"] = True

    return {
        "entities": [
            {"file": file_block},
            {"protein": binder_block},
        ]
    }


def get_chain_residue_mask(structure, chain_name: str) -> np.ndarray:
    """Return the residue mask for a parsed chain name."""
    hits = np.where(structure.chains["name"] == chain_name)[0]
    if len(hits) == 0:
        raise ValueError(
            f"Chain '{chain_name}' was not found after BoltzGen parsing. "
            f"Available chains: {structure.chains['name'].tolist()}"
        )

    residue_mask = np.zeros(len(structure.residues), dtype=bool)
    for chain_idx in hits:
        chain = structure.chains[chain_idx]
        start = int(chain["res_idx"])
        end = start + int(chain["res_num"])
        residue_mask[start:end] = True
    return residue_mask


def infer_binder_chain_name(structure, requested: str) -> str:
    """Recover a renamed binder chain, falling back to the final chain."""
    names = [str(name) for name in structure.chains["name"].tolist()]
    if requested in names:
        return requested
    if not names:
        raise ValueError("Parsed structure contains no chains.")
    return names[-1]


def prepare_design_dir(
    args: argparse.Namespace,
    target_structure: Path,
    moldir: Path,
    parser: YamlDesignParser,
    tokenizer: Tokenizer,
    binder: BinderSpec,
) -> dict[str, Any]:
    """Parse one design and write BoltzGen CIF and token metadata inputs."""
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mol_out_dir = output_dir / const.molecules_dirname
    mol_out_dir.mkdir(parents=True, exist_ok=True)

    schema = build_schema(args, target_structure, binder)
    if args.debug:
        (output_dir / f"{binder.sample_id}_input_schema.json").write_text(
            json.dumps(schema, indent=2) + "\n"
        )

    target = parser.parse_boltzgen_schema(
        name=binder.sample_id,
        schema=schema,
        mols={},
        mol_dir=moldir,
        base_file_path=Path.cwd(),
    )

    for name, mol in (target.extra_mols or {}).items():
        with (mol_out_dir / f"{name}.pkl").open("wb") as handle:
            pickle.dump(mol, handle)

    design_cif = output_dir / f"{binder.sample_id}.cif"
    design_cif.write_text(to_mmcif(target.structure))

    tokenized = tokenizer.tokenize(target.structure)
    binder_chain = infer_binder_chain_name(target.structure, binder.chain)
    binder_residue_mask = get_chain_residue_mask(target.structure, binder_chain)
    token_to_res = tokenized.token_to_res
    design_mask = binder_residue_mask[token_to_res].astype(np.float32)

    if design_mask.sum() == 0:
        raise ValueError("No binder tokens were marked for refolding.")

    design_info = target.design_info
    metadata: dict[str, Any] = {"design_mask": design_mask}
    if design_info is not None:
        metadata["structure_group"] = design_info.res_structure_groups[
            token_to_res
        ].astype(np.int64)
        metadata["binding_type"] = design_info.res_binding_type[token_to_res].astype(
            np.float32
        )
        metadata["ss_type"] = design_info.res_ss_types[token_to_res].astype(np.int64)
    metadata["mol_type"] = tokenized.tokens["mol_type"]
    metadata["token_resolved_mask"] = np.ones(len(tokenized.tokens), dtype=np.float32)
    metadata["inverse_fold_design_mask"] = None
    np.savez_compressed(output_dir / f"{binder.sample_id}.npz", **metadata)

    chain_summary = []
    chain_sequences: dict[str, str] = {}
    for chain in target.structure.chains:
        start = int(chain["res_idx"])
        end = start + int(chain["res_num"])
        chain_name = str(chain["name"])
        if int(chain["mol_type"]) == const.chain_type_ids["PROTEIN"]:
            chain_sequences[chain_name] = protein_residue_sequence(
                target.structure.residues[start:end]
            )
        chain_summary.append(
            {
                "name": chain_name,
                "mol_type": int(chain["mol_type"]),
                "num_residues": int(chain["res_num"]),
                "design_residues": int(binder_residue_mask[start:end].sum()),
            }
        )

    return {
        "design_dir": str(output_dir),
        "design_cif": str(design_cif),
        "metadata_npz": str((output_dir / f"{binder.sample_id}.npz")),
        "binder_chain_requested": binder.chain,
        "binder_chain_parsed": binder_chain,
        "binder_sequence": binder.sequence,
        "cyclic": binder.cyclic,
        "num_tokens": int(len(tokenized.tokens)),
        "num_design_tokens": int(design_mask.sum()),
        "num_binding_tokens": int(metadata.get("binding_type", np.array([])).sum()),
        "structure_group_counts": {
            str(group): int((metadata["structure_group"] == group).sum())
            for group in np.unique(metadata.get("structure_group", np.array([])))
        },
        "chain_sequences": chain_sequences,
        "chains": chain_summary,
    }


def get_resource_config(name: str) -> Path:
    """Locate a configuration shipped with the installed BoltzGen runtime."""
    return Path(boltzgen_resource_main.__file__).resolve().parent / "config" / name


def run_folding_step(
    args: argparse.Namespace,
    design_dir: Path,
    checkpoint: Path,
    moldir: Path,
) -> None:
    """Execute BoltzGen folding with CLI-derived Hydra overrides."""
    msa_config_path = design_dir / "target_msa_config.json"
    if args.target_msa:
        install_generated_msa_patch(msa_config_path)

    overrides = [
        f"output={design_dir}",
        f"data.design_dir={design_dir}",
        f"checkpoint={checkpoint}",
        f"data.cfg.moldir={moldir}",
        f"trainer.accelerator={args.accelerator}",
        f"trainer.devices={args.devices}",
        f"trainer.precision={args.precision}",
        f"data.cfg.num_workers={args.num_workers}",
        f"data.skip_existing={str(args.reuse_existing_folds).lower()}",
        f"recycling_steps={args.recycling_steps}",
        f"sampling_steps={args.sampling_steps}",
        f"diffusion_samples={args.diffusion_samples}",
    ]
    max_msa_seqs = effective_max_msa_seqs(args)
    if max_msa_seqs is not None:
        overrides.append(f"data.cfg.max_seqs={max_msa_seqs}")
    boltzgen_resource_main.main(str(get_resource_config("fold.yaml")), overrides)


def run_analysis_step(args: argparse.Namespace, design_dir: Path, moldir: Path) -> None:
    """Execute the BoltzGen analysis recipe for the folded batch."""
    overrides = [
        f"design_dir={design_dir}",
        f"data.design_dir={design_dir}",
        f"data.cfg.moldir={moldir}",
        f"data.cfg.num_workers={args.num_workers}",
        f"num_processes={args.analysis_processes}",
        "noncovalents_original=false",
        f"noncovalents_refolded={str(not args.no_noncovalents_refolded).lower()}",
        "delta_sasa_original=false",
        f"delta_sasa_refolded={str(not args.no_delta_sasa_refolded).lower()}",
        "largest_hydrophobic=false",
        "largest_hydrophobic_refolded=false",
        "backbone_fold_metrics=true",
        "designfolding_metrics=false",
        "allatom_fold_metrics=false",
        "liability_analysis=true",
        "liability_modality=peptide",
        f"liability_peptide_type={args.liability_peptide_type}",
    ]
    boltzgen_resource_main.main(str(get_resource_config("analysis.yaml")), overrides)


def summarize_npz(npz_path: Path, keys: list[str]) -> dict[str, Any]:
    """Extract best-sample folding metrics from a Boltz result archive."""
    arr = np.load(npz_path)
    summary: dict[str, Any] = {"path": str(npz_path)}

    best_idx = None
    if "iptm" in arr.files and "ptm" in arr.files:
        confidence = 0.8 * arr["iptm"] + 0.2 * arr["ptm"]
        best_idx = int(np.argmax(confidence))
        summary["best_sample_idx"] = best_idx
        summary["boltz_confidence"] = float(confidence[best_idx])

    for key in keys:
        if key not in arr.files:
            continue
        value = arr[key]
        if np.ndim(value) == 0:
            summary[key] = value.item()
        elif best_idx is not None and len(value) > best_idx:
            summary[key] = float(value[best_idx])
            summary[f"{key}_all"] = np.asarray(value).tolist()
        else:
            summary[key] = np.asarray(value).tolist()

    return summary


def summarize_analysis_row(csv_path: Path, row: dict[str, str]) -> dict[str, Any]:
    """Normalize selected analysis CSV columns for the JSON summary."""
    summary: dict[str, Any] = {"path": str(csv_path)}
    for key in ["id", "target_id", "designed_sequence", "designed_chain_sequence"]:
        if key in row:
            summary[key] = row[key]
    for key in ANALYSIS_SUMMARY_KEYS:
        if key not in row or row[key] == "":
            continue
        try:
            summary[key] = float(row[key])
        except ValueError:
            summary[key] = row[key]
    return summary


def summarize_analysis_csv(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Index normalized analysis summaries by sample id."""
    summaries: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row.get("id")
            if sample_id:
                summaries[sample_id] = summarize_analysis_row(csv_path, row)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous summary rows with JSON-encoded nested values."""
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                normalized[key] = value
            writer.writerow(normalized)


def cleanup_intermediates(design_dir: Path, sample_ids: list[str], debug: bool) -> None:
    """Remove regenerable batch intermediates unless debugging is enabled."""
    if debug:
        return

    for sample_id in sample_ids:
        for suffix in [".cif", ".npz", "_input_schema.json", "_summary.json"]:
            (design_dir / f"{sample_id}{suffix}").unlink(missing_ok=True)

    for filename in ["ca_coords_sequences.pkl.gz"]:
        (design_dir / filename).unlink(missing_ok=True)
    (design_dir / "target_msa_config.json").unlink(missing_ok=True)

    for dirname in [
        const.folding_dirname,
        const.molecules_dirname,
        const.metrics_dirname,
        "target_msas",
        "des_pdbs",
        "des_refold_pdbs",
        "lightning_logs",
    ]:
        path = design_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def remove_stale_prepared_inputs(design_dir: Path, sample_ids: list[str]) -> None:
    """Remove prepared inputs that no longer belong to the active manifest."""
    if not design_dir.exists():
        return
    current = set(sample_ids)
    for path in design_dir.iterdir():
        if path.is_dir():
            continue
        if path.suffix in {".cif", ".npz"} and path.stem not in current:
            path.unlink()
            continue
        if path.name.endswith("_input_schema.json"):
            sample_id = path.name.removesuffix("_input_schema.json")
            if sample_id not in current:
                path.unlink()


def cap_zip_moldir_workers(args: argparse.Namespace, moldir: Path) -> None:
    """Avoid unsafe forked reads when the molecule directory is a zip file."""
    if args.num_workers <= 1:
        return
    if not moldir.is_file() or moldir.suffix != ".zip":
        return
    print(
        "Warning: reducing --num-workers from "
        f"{args.num_workers} to 1 because BoltzGen caches an open molecule zip "
        "file, which can fail under forked DataLoader workers. Use an extracted "
        "--moldir directory to safely use more workers."
    )
    args.num_workers = 1


def main() -> None:
    """Prepare binders, run requested stages, and write portable summaries."""
    args = parse_args()
    target_structure = args.target_structure.expanduser().resolve()
    design_dir = args.output_dir.expanduser().resolve()
    design_dir.mkdir(parents=True, exist_ok=True)
    moldir = resolve_artifact(args.moldir, repo_type="dataset")
    cap_zip_moldir_workers(args, moldir)
    binders = collect_binder_specs(args)
    sample_ids = [binder.sample_id for binder in binders]
    target_chains = normalize_target_chains(args)
    target_msa_specs = normalize_target_msa_specs(args, target_chains)
    remove_stale_prepared_inputs(design_dir, sample_ids)

    parser = YamlDesignParser(mol_dir=moldir)
    tokenizer = Tokenizer(atomize_modified_residues=False)
    prepared_by_id = {
        binder.sample_id: prepare_design_dir(
            args=args,
            target_structure=target_structure,
            moldir=moldir,
            parser=parser,
            tokenizer=tokenizer,
            binder=binder,
        )
        for binder in binders
    }
    first_prepared = next(iter(prepared_by_id.values()))
    target_msa_config = write_target_msa_sidecars(
        args,
        design_dir,
        target_msa_specs,
        first_prepared.get("chain_sequences", {}),
    )

    summary: dict[str, Any] = {
        "target_structure": str(target_structure),
        "target_chain": target_chains,
        "target_res_index": args.target_res_index,
        "binding_res_index": args.binding_res_index,
        "structure_groups": normalize_structure_groups(args),
        "target_msa": target_msa_config,
        "moldir": str(moldir),
        "output_dir": str(design_dir),
        "num_samples": len(binders),
        "samples": [],
    }

    if args.prepare_only and args.skip_folding:
        raise ValueError("--prepare-only and --skip-folding are mutually exclusive")

    analysis_by_id: dict[str, dict[str, Any]] = {}
    if not args.prepare_only:
        if not args.skip_folding:
            fold_checkpoint = resolve_artifact(args.fold_checkpoint, repo_type="model")
            run_folding_step(args, design_dir, fold_checkpoint, moldir)

        if args.run_analysis:
            missing_outputs = [
                sample_id
                for sample_id in sample_ids
                if not (
                    design_dir / const.folding_dirname / f"{sample_id}.npz"
                ).exists()
                or not (
                    design_dir / const.refold_cif_dirname / f"{sample_id}.cif"
                ).exists()
            ]
            if missing_outputs:
                raise FileNotFoundError(
                    "Analysis requires existing folding outputs for every sample. "
                    f"Missing: {', '.join(missing_outputs)}"
                )
            run_analysis_step(args, design_dir, moldir)
            csv_path = design_dir / "aggregate_metrics_analyze.csv"
            if csv_path.exists():
                analysis_by_id = summarize_analysis_csv(csv_path)

    fold_rows: list[dict[str, Any]] = []
    for binder in binders:
        sample_summary: dict[str, Any] = {
            "sample_id": binder.sample_id,
            "binder_sequence": binder.sequence,
            "binder_chain": binder.chain,
            "cyclic": binder.cyclic,
        }
        prepared = prepared_by_id[binder.sample_id]
        if args.debug or args.prepare_only:
            sample_summary["prepared_input"] = prepared
        else:
            sample_summary["prepared"] = {
                "binder_chain_parsed": prepared["binder_chain_parsed"],
                "num_tokens": prepared["num_tokens"],
                "num_design_tokens": prepared["num_design_tokens"],
                "num_binding_tokens": prepared["num_binding_tokens"],
                "chains": prepared["chains"],
            }
        if not args.prepare_only:
            fold_npz = design_dir / const.folding_dirname / f"{binder.sample_id}.npz"
            refold_cif = design_dir / const.refold_cif_dirname / f"{binder.sample_id}.cif"
            if fold_npz.exists():
                fold_summary = summarize_npz(fold_npz, FOLD_SUMMARY_KEYS)
                if not args.debug:
                    fold_summary.pop("path", None)
                if refold_cif.exists():
                    fold_summary["refold_cif"] = str(refold_cif)
                sample_summary["folding"] = fold_summary
                fold_rows.append(
                    {
                        "sample_id": binder.sample_id,
                        "binder_sequence": binder.sequence,
                        "binder_chain": binder.chain,
                        "cyclic": binder.cyclic,
                        **{
                            key: value
                            for key, value in fold_summary.items()
                            if not key.endswith("_all")
                        },
                    }
                )
            if binder.sample_id in analysis_by_id:
                sample_summary["analysis"] = analysis_by_id[binder.sample_id]
        summary["samples"].append(sample_summary)

    write_csv(design_dir / "fold_scores.csv", fold_rows)
    summary_path = design_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if not args.prepare_only:
        cleanup_intermediates(design_dir, sample_ids, args.debug)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
