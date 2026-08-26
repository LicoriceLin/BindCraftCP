"""EpitopeCraft command-line interface with lazy backend imports."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import click


def _load_object(spec: str):
    if ":" not in spec:
        raise click.BadParameter("Use MODULE:OBJECT, for example myproject.workflow:build")
    module_name, object_name = spec.split(":", 1)
    try:
        return getattr(importlib.import_module(module_name), object_name)
    except (ImportError, AttributeError) as error:
        raise click.BadParameter(f"Cannot import {spec}: {error}") from error


@click.group()
def cli():
    """Compose, inspect, and run protein or ligand design workflows."""


@cli.command("standard-design")
@click.option(
    "--target-settings",
    "-t",
    required=True,
    help="YAML/JSON target structure, chains, and hotspots.",
)
@click.option(
    "--binder-settings",
    "-b",
    required=True,
    help="YAML/JSON binder lengths, seeds, and output path.",
)
@click.option(
    "--advanced-settings",
    "-a",
    multiple=True,
    default=("epitopecraft/pipelines/config/base_advanced_settings.yaml",),
    help="Legacy advanced settings files, applied in order.",
)
@click.option(
    "--filter-settings",
    "-f",
    default="epitopecraft/pipelines/config/default_filter.yaml",
    help="Legacy filter thresholds and stages.",
)
def standard_design(
    target_settings: str,
    binder_settings: str,
    advanced_settings: tuple[str, ...],
    filter_settings: str,
):
    """Run the legacy HalluDesign workflow during the core-v2 migration."""

    from epitopecraft.pipelines.hallu_design import HalluDesign
    from epitopecraft.utils.settings import (
        AdvancedSettings,
        BinderSettings,
        FilterSettings,
        GlobalSettings,
        TargetSettings,
    )

    advanced = (
        AdvancedSettings.from_file(advanced_settings[0])
        if len(advanced_settings) == 1
        else AdvancedSettings(list(advanced_settings))
    )
    settings = GlobalSettings(
        target_settings=TargetSettings.from_file(target_settings),
        binder_settings=BinderSettings.from_file(binder_settings),
        advanced_settings=advanced,
        filter_settings=FilterSettings.from_file(filter_settings),
    )
    HalluDesign(settings).run()


@cli.command("design-all-in-one")
@click.argument("global_settings")
def design_all_in_one(global_settings: str):
    """Run legacy HalluDesign from one saved GlobalSettings file."""

    from epitopecraft.pipelines.hallu_design import HalluDesign
    from epitopecraft.utils.settings import GlobalSettings

    HalluDesign(GlobalSettings.from_file(global_settings)).run()


@cli.group("pipeline")
def pipeline_group():
    """Inspect a lightweight core-v2 Pipeline without running its backends."""


def _build_pipeline(factory_spec: str, config_path: Path | None):
    from epitopecraft.core.config import PipelineConfig
    from epitopecraft.core.pipeline import Pipeline

    factory = _load_object(factory_spec)
    pipeline = factory()
    if not isinstance(pipeline, Pipeline):
        raise click.ClickException(f"{factory_spec} did not return a Pipeline")
    if config_path is not None:
        pipeline.config = PipelineConfig.from_file(config_path)
    return pipeline


@pipeline_group.command("describe")
@click.argument("factory")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
@click.option("--schema", is_flag=True, help="Print the accepted config schema as JSON.")
def describe_pipeline(factory: str, config_path: Path | None, schema: bool):
    """Print graph wiring and resolved parameters for MODULE:FACTORY."""

    pipeline = _build_pipeline(factory, config_path)
    if schema:
        click.echo(json.dumps(pipeline.config_schema(), indent=2, default=str))
    else:
        click.echo(pipeline.describe())


@pipeline_group.command("plan")
@click.argument("factory")
@click.option("--config", "config_path", type=click.Path(path_type=Path))
def plan_pipeline(factory: str, config_path: Path | None):
    """Print the ordered Step execution plan without loading a backend."""

    from epitopecraft.core.pipeline import PipelineRunner

    pipeline = _build_pipeline(factory, config_path)
    pipeline.resolve_config()
    click.echo(json.dumps(PipelineRunner(pipeline).plan(), indent=2))


@cli.group("refold")
def refold_group():
    """Run or repair structure refolding workflows."""


@refold_group.command(
    "boltzgen",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.option("--environment", default="boltzgen", show_default=True)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def boltzgen_refold(environment: str, arguments: tuple[str, ...]):
    """Forward full co-fold options to the repository-owned BoltzGen runtime."""

    command = [
        "conda",
        "run",
        "-n",
        environment,
        "python",
        "-m",
        "epitopecraft.backends.boltzgen.peptide_refold_cli",
        *arguments,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise click.ClickException(f"BoltzGen refold failed with exit code {error.returncode}")


def _forward_python_module(module: str, arguments: tuple[str, ...]) -> None:
    try:
        subprocess.run([sys.executable, "-m", module, *arguments], check=True)
    except subprocess.CalledProcessError as error:
        raise click.ClickException(f"Command failed with exit code {error.returncode}")


@refold_group.command(
    "repair-missing",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def repair_missing_refolds(arguments: tuple[str, ...]):
    """Repair incomplete legacy graft/refold records; pass --help for options."""

    _forward_python_module("epitopecraft.cli.refold_missing", arguments)


@cli.group("inspect")
def inspect_group():
    """Export human-readable debugging views of pipeline artifacts."""


@inspect_group.command(
    "animation",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def render_animation(arguments: tuple[str, ...]):
    """Render a Hallucinate trajectory; pass --help for options."""

    _forward_python_module("epitopecraft.cli.render_hallu_animation", arguments)


@cli.group("redesign")
def redesign_group():
    """Run sequence redesign workflows."""


@redesign_group.command(
    "mpnn-one",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
def redesign_mpnn_one(arguments: tuple[str, ...]):
    """MPNN-redesign one cached complex; pass --help for options."""

    _forward_python_module("epitopecraft.cli.mpnn_redesign", arguments)


if __name__ == "__main__":
    cli()
