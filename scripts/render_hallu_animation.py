#!/usr/bin/env python
"""Run single ALS3 hallucinations and save ColabDesign trajectory animations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epitopecraft.steps import Hallucinate
from epitopecraft.utils import (
    AdvancedSettings,
    BinderSettings,
    DesignRecord,
    FilterSettings,
    GlobalSettings,
    TargetSettings,
)


DEFAULT_TARGETS = (
    "input/Jul_benchmark/segs/ALS3-full.pdb",
    "input/Jul_benchmark/segs/ALS3-seg_10.pdb",
)


def _init_field_names(settings_cls: type[Any]) -> set[str]:
    return {field.name for field in fields(settings_cls) if field.init}


def _advanced_dict(flat_settings: dict[str, Any]) -> dict[str, Any]:
    structured_keys = (
        _init_field_names(TargetSettings)
        | _init_field_names(BinderSettings)
        | _init_field_names(FilterSettings)
    )
    return {key: value for key, value in flat_settings.items() if key not in structured_keys}


def load_flat_settings(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def build_settings(
    flat_settings: dict[str, Any],
    target_pdb: Path,
    outdir: Path,
    length: int,
    seed: int,
    helix: float,
    hotspot: str | None,
) -> GlobalSettings:
    target_dict = {
        key: flat_settings[key]
        for key in _init_field_names(TargetSettings)
        if key in flat_settings
    }
    target_dict["starting_pdb"] = str(target_pdb)
    target_dict["full_target_pdb"] = str(target_pdb)
    target_dict["target_hotspot_residues"] = hotspot

    binder_dict = {
        key: flat_settings[key]
        for key in _init_field_names(BinderSettings)
        if key in flat_settings
    }
    binder_dict.update(
        {
            "design_path": str(outdir.parent),
            "binder_name": target_pdb.stem,
            "binder_lengths": [length],
            "random_seeds": [seed],
            "helix_values": [helix],
        }
    )

    filter_dict = {
        key: flat_settings[key]
        for key in _init_field_names(FilterSettings)
        if key in flat_settings
    }
    if filter_dict.get("filters_path") and not Path(filter_dict["filters_path"]).is_file():
        filter_dict["filters_path"] = None
    adv = _advanced_dict(flat_settings)

    return GlobalSettings(
        target_settings=TargetSettings.from_dict(target_dict),
        binder_settings=BinderSettings.from_dict(binder_dict),
        advanced_settings=AdvancedSettings(
            advanced_paths=flat_settings.get("advanced_paths", ["none"]),
            extra_patch=flat_settings.get("extra_patch", {}),
            _settings=adv,
        ),
        filter_settings=FilterSettings.from_dict(filter_dict),
    )


def run_one(
    flat_settings: dict[str, Any],
    target_pdb: Path,
    outdir: Path,
    length: int,
    seed: int,
    helix: float,
    dpi: int,
    hotspot: str | None,
) -> Path:
    if not target_pdb.is_file():
        raise FileNotFoundError(f"Target PDB not found: {target_pdb}")

    settings = build_settings(flat_settings, target_pdb, outdir, length, seed, helix, hotspot)
    hallu = Hallucinate(settings=settings)

    record = DesignRecord(id=target_pdb.stem, sequence="")
    record.update_metrics(
        {
            "config:length": length,
            "config:seed": seed,
            "config:helix": helix,
        }
    )
    hallu.process_record(record)

    outdir.mkdir(parents=True, exist_ok=True)
    animation_html = hallu.af_model.animate(dpi=dpi)
    hotspot_label = "nohotspot" if hotspot is None else "hotspot"
    output_path = outdir / (
        f"{target_pdb.stem}_hallu_{hotspot_label}_len{length}_seed{seed}_helix{helix}.html"
    )
    with output_path.open("w") as handle:
        handle.write(animation_html)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ColabDesign hallucination trajectory animations for ALS3."
    )
    parser.add_argument("--settings", default="output/ALS3/settings.json")
    parser.add_argument("--outdir", default="output/gradient/trajs")
    parser.add_argument("--length", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--helix", type=float, default=-0.3)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--hotspot",
        default="settings",
        help="Use 'settings' to preserve settings.json hotspots, or 'null'/'none' for no hotspot.",
    )
    parser.add_argument("--target", action="append", dest="targets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings_path = Path(args.settings)
    outdir = Path(args.outdir)
    targets = tuple(args.targets) if args.targets else DEFAULT_TARGETS
    hotspot = None if args.hotspot.lower() in {"null", "none"} else args.hotspot
    if hotspot == "settings":
        hotspot = load_flat_settings(settings_path).get("target_hotspot_residues")

    flat_settings = load_flat_settings(settings_path)
    for target in targets:
        output_path = run_one(
            flat_settings=flat_settings,
            target_pdb=Path(target),
            outdir=outdir,
            length=args.length,
            seed=args.seed,
            helix=args.helix,
            dpi=args.dpi,
            hotspot=hotspot,
        )
        print(output_path)


if __name__ == "__main__":
    main()
