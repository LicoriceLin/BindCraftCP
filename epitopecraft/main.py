import click
from epitopecraft.pipelines.hallu_design import (
    TargetSettings,GlobalSettings,AdvancedSettings,
    FilterSettings,BinderSettings,HalluDesign,_dir_path
)



@click.command()
@click.option(
    '--target-settings',
    help='JSON file defining design target: pdb files, chains, hot spots. etc.')
@click.option(
    '--binder-settings',
    help='JSON file defining binder conditions: lengths, helix penalty values, random seeds. etc.')
@click.option(
    '--advanced-settings',
    default='epitopecraft/pipelines/config/base_advanced_settings.yaml',
    help='JSON file for advanced design settings. Check the default file for detailed options'
)
@click.option(
    '--filter-settings',
    default='epitopecraft/pipelines/config/default_filter.json',
    help='JSON file to define filter thresholds and steps when they are applied.'
)
def standard_design(
    target_settings:str,binder_settings:str,advanced_settings:str,filter_settings:str):
    settings=GlobalSettings(
        target_settings=TargetSettings.from_file(target_settings),
        binder_settings=BinderSettings.from_file(binder_settings),
        advanced_settings=AdvancedSettings.from_file(advanced_settings),
        filter_settings=FilterSettings.from_file(filter_settings)
        )
    hallu=HalluDesign(settings)
    batch=hallu.run()

@click.group()
def cli():
    pass

cli.add_command(standard_design)
