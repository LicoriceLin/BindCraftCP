#!/usr/bin/env python
"""Run graft + templated Refold for cached designs that are missing refold output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ADVANCED = Path("epitopecraft/pipelines/config/base_advanced_settings.yaml")
DEFAULT_TEMPLATE_PATCH = Path("epitopecraft/pipelines/config/patch_templated.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design-path",
        action="append",
        type=Path,
        dest="design_paths",
        required=True,
        help="Design output directory containing settings.json and metrics/*.json.",
    )
    parser.add_argument("--advanced-settings", type=Path, default=DEFAULT_ADVANCED)
    parser.add_argument("--template-patch", type=Path, default=DEFAULT_TEMPLATE_PATCH)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--run-monomer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run monomer refold models in addition to multimer validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force rerun selected entries even when refold outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selection counts and selected ids; do not import AF2 steps.",
    )
    parser.add_argument(
        "--selection-log",
        type=Path,
        help=(
            "Reuse selected_ids from a previous stdout log. This avoids shard drift "
            "when rerunning one cancelled array task after other shards have completed."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Static JSON manifest of record ids to process, written by --write-manifest.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="Write a static JSON manifest of currently incomplete records and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def record_json_paths(metrics_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in metrics_dir.glob("*.json")
        if path.stem != "metrics"
        and "-mpnn" not in path.stem
        and "seg_con" not in path.stem
    )


def nested_has(mapping: dict[str, Any], key: str) -> bool:
    current: Any = mapping
    for part in key.split(":"):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def has_complete_refold(record: dict[str, Any], run_monomer: bool = True) -> bool:
    metrics = record.get("metrics", {})
    pdbs = record.get("pdb_files", {}) | record.get("pdb_strs", {})
    required_metrics = []
    required_pdbs = ["refold:best"]

    for model_num in (1, 2):
        key = f"refold:multimer-{model_num}"
        required_pdbs.append(key)
        required_metrics.extend(
            f"{key}:{metric}"
            for metric in ("pLDDT", "pTM", "i-pTM", "pAE", "i-pAE")
        )
        if run_monomer:
            key = f"refold:monomer-{model_num}"
            required_pdbs.append(key)
            required_metrics.extend(
                f"{key}:{metric}" for metric in ("pLDDT", "pTM", "pAE")
            )

    required_metrics.append("refold:best:i-pAE")
    return all(key in pdbs for key in required_pdbs) and all(
        nested_has(metrics, key) for key in required_metrics
    )


def missing_refold_paths(metrics_dir: Path, overwrite: bool, run_monomer: bool) -> list[Path]:
    selected = []
    for path in record_json_paths(metrics_dir):
        if overwrite:
            selected.append(path)
            continue
        if not has_complete_refold(load_json(path), run_monomer=run_monomer):
            selected.append(path)
    return selected


def shard(items: Iterable[str], shard_index: int, shard_count: int) -> list[str]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < count")
    ordered = sorted(items)
    return [item for offset, item in enumerate(ordered) if offset % shard_count == shard_index]


def print_selection(design_path: Path, total: int, selected_ids: list[str]) -> None:
    print(
        "selection",
        f"design_path={design_path}",
        f"missing_or_overwrite={total}",
        f"shard_selected={len(selected_ids)}",
    )
    if selected_ids:
        print("selected_ids=" + ",".join(selected_ids))


def selected_ids_from_log(log_path: Path) -> dict[str, list[str]]:
    ret: dict[str, list[str]] = {}
    current: str | None = None
    design_re = re.compile(r"selection design_path=(\S+)")
    with log_path.open() as handle:
        for line in handle:
            design_match = design_re.search(line)
            if design_match:
                current = str(Path(design_match.group(1)).resolve())
                continue
            if current is not None and line.startswith("selected_ids="):
                ids = line.removeprefix("selected_ids=").strip()
                ret[current] = [item for item in ids.split(",") if item]
                current = None
    return ret


def selected_ids_from_manifest(manifest_path: Path) -> dict[str, list[str]]:
    manifest = load_json(manifest_path)
    ret: dict[str, list[str]] = {}
    for item in manifest["design_paths"]:
        ret[str(Path(item["design_path"]).resolve())] = list(item["selected_ids"])
    return ret


def write_manifest(
    manifest_path: Path,
    design_paths: list[Path],
    overwrite: bool,
    run_monomer: bool,
) -> dict[str, list[str]]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selections: dict[str, list[str]] = {}
    payload = {"design_paths": []}
    for design_path in design_paths:
        missing = missing_refold_paths(
            design_path / "metrics",
            overwrite=overwrite,
            run_monomer=run_monomer,
        )
        selected_ids = [path.stem for path in missing]
        selections[str(design_path)] = selected_ids
        payload["design_paths"].append(
            {
                "design_path": str(design_path),
                "selected_ids": selected_ids,
                "selected_count": len(selected_ids),
            }
        )
        print_selection(design_path, len(selected_ids), selected_ids)

    with manifest_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"manifest={manifest_path}")
    return selections


def build_settings(
    design_path: Path,
    advanced_settings: Path,
    template_patch: Path,
    run_monomer: bool,
):
    from epitopecraft.utils.settings import (
        AdvancedSettings,
        BinderSettings,
        FilterSettings,
        GlobalSettings,
        TargetSettings,
    )

    saved = load_json(design_path / "settings.json")
    target_settings = TargetSettings.from_dict(saved)
    binder_settings = BinderSettings.from_dict(saved)
    binder_settings.design_path = str(design_path)

    extra_patch = {
        "templated": True,
        "refold-run-monomer": run_monomer,
        "cyclize_peptide": bool(saved.get("cyclize_peptide", False)),
    }
    for key in (
        "use_multimer_design",
        "num_recycles_validation",
        "global_seed",
        "af_params_dir",
    ):
        if key in saved and saved[key] not in (None, ""):
            extra_patch[key] = saved[key]

    return GlobalSettings(
        target_settings=target_settings,
        binder_settings=binder_settings,
        advanced_settings=AdvancedSettings(
            advanced_paths=[str(advanced_settings), str(template_patch)],
            extra_patch=extra_patch,
        ),
        filter_settings=FilterSettings(filters_path=None),
    )


def run_design_path(
    design_path: Path,
    selected_ids: list[str],
    args: argparse.Namespace,
) -> None:
    if not selected_ids:
        return

    from epitopecraft.steps.refold import Graft, Refold
    from epitopecraft.utils.design_record import DesignBatch

    settings = build_settings(
        design_path,
        args.advanced_settings,
        args.template_patch,
        run_monomer=args.run_monomer,
    )
    batch = DesignBatch.from_cache(design_path / "metrics")[selected_ids]
    batch.parent.set_overwrite(args.overwrite)

    graft = Graft(settings)
    refold = Refold(settings)

    graft.process_batch(batch, pdb_purge_stem="graft")
    refold.process_batch(batch, pdb_purge_stem="refold", pdb_to_take=graft.pdb_to_add[0])


def main() -> None:
    args = parse_args()
    design_paths = list(args.design_paths)
    design_paths = [path.resolve() for path in design_paths]
    args.advanced_settings = args.advanced_settings.resolve()
    args.template_patch = args.template_patch.resolve()

    if args.write_manifest is not None:
        write_manifest(
            args.write_manifest.resolve(),
            design_paths,
            overwrite=args.overwrite,
            run_monomer=args.run_monomer,
        )
        return

    selected_by_path: dict[Path, list[str]] = {}
    if args.selection_log is not None and args.manifest is not None:
        raise ValueError("Use only one of --selection-log or --manifest.")
    static_selection = (
        selected_ids_from_log(args.selection_log.resolve())
        if args.selection_log is not None
        else selected_ids_from_manifest(args.manifest.resolve())
        if args.manifest is not None
        else None
    )
    for design_path in design_paths:
        if static_selection is None:
            metrics_dir = design_path / "metrics"
            missing = missing_refold_paths(
                metrics_dir,
                overwrite=args.overwrite,
                run_monomer=args.run_monomer,
            )
            selected_ids = shard([path.stem for path in missing], args.shard_index, args.shard_count)
            total = len(missing)
        else:
            ids = static_selection.get(str(design_path), [])
            selected_ids = shard(ids, args.shard_index, args.shard_count)
            total = len(ids)
        selected_by_path[design_path] = selected_ids
        print_selection(design_path, total, selected_ids)

    if args.dry_run:
        return

    for design_path, selected_ids in selected_by_path.items():
        run_design_path(design_path, selected_ids, args)


if __name__ == "__main__":
    main()
