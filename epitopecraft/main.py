import click
from epitopecraft.pipelines.hallu_design import (
    TargetSettings,GlobalSettings,AdvancedSettings,
    FilterSettings,BinderSettings,HalluDesign,_dir_path
)
from typing import Tuple


@click.command()
@click.option(
    '--target-settings','-t',
    help='yaml/json file defining design target: pdb files, chains, hot spots. etc.')
@click.option(
    '--binder-settings','-b',
    help='yaml/json file defining binder conditions: lengths, helix penalty values, random seeds. etc.')
@click.option(
    '--advanced-settings','-a', multiple=True,
    default=['epitopecraft/pipelines/config/base_advanced_settings.yaml'],
    help='yaml/json file for advanced design settings. Check the default file for detailed options'
    )
@click.option(
    '--filter-settings','-f',
    default='epitopecraft/pipelines/config/default_filter.json',
    help='yaml/json files to define filter thresholds and steps when they are applied.'
    )
def standard_design(
    target_settings:str,binder_settings:str,advanced_settings:Tuple[str,...],filter_settings:str):
    if len(advanced_settings)==1:
        adv_s=AdvancedSettings.from_file(advanced_settings[0])
    else:
        adv_s=AdvancedSettings(list(advanced_settings))
    settings=GlobalSettings(
        target_settings=TargetSettings.from_file(target_settings),
        binder_settings=BinderSettings.from_file(binder_settings),
        advanced_settings=adv_s,
        filter_settings=FilterSettings.from_file(filter_settings)
        )
    hallu=HalluDesign(settings)
    batch=hallu.run()


@click.command()
@click.argument(
    'global_settings'
)
def design_all_in_one(global_settings:str):
    settings=GlobalSettings.from_file(global_settings)
    hallu=HalluDesign(settings)
    batch=hallu.run()

@click.group()
def cli():
    pass

cli.add_command(standard_design)
cli.add_command(design_all_in_one)

if __name__ == '__main__':
    cli()