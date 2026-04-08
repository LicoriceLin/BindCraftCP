from ..steps import Hallucinate,PseudoHotspot
from ..utils.preprocess import hotspots_topk_motifs

from ..utils import (
    TargetSettings,GlobalSettings,AdvancedSettings,
    FilterSettings,BinderSettings,NEST_SEP,DesignRecord
    )
from ..utils.settings import BaseSettings,dataclass
from .base_pipeline import BasePipeline,_dir_path
import json
import sys
from pathlib import Path
from typing import Dict,Any,Tuple
from functools import partial
from .base_pipeline import BasePipeline,_dir_path

def init_settings(
    starting_pdb:str,
    target_chain:str,
    design_path:str,
    advanced_paths=[_dir_path/'config/base_adv_setting.json'],
    patch:str|Dict[str,Any]|None=None
    ):
    settings=GlobalSettings(
        target_settings=TargetSettings(
            starting_pdb=starting_pdb,
            target_chain=target_chain,
        ),
        binder_settings=BinderSettings(design_path=design_path,
            binder_name=Path(starting_pdb).stem,
            binder_lengths=[50],
            random_seeds=list(range(15)),
            helix_values=[-0.3,0.0]),
        advanced_settings=AdvancedSettings(
            advanced_paths=advanced_paths,
            extra_patch=patch),
        filter_settings=FilterSettings.from_file('epitopecraft/pipelines/config/default_filter.yaml')
        )
    return settings

class EpitopeDetect(BasePipeline):
    def _init_steps(self):
        settings=self.settings
        self.hallu=Hallucinate(settings)
        self.hotspot_finder=PseudoHotspot(settings)

    def _config_steps(self):
        adv=self.settings.adv
        self.hallu.config_pdb_purge(adv.setdefault('hallu_stem','hallu'))

    def run(self):
        adv=self.settings.adv
        batch=self.hallu.process_batch(batch_cache_stem=adv.setdefault('metrics_stem','metrics'),
            overwrite=adv.get('overwrite',False))
        batch = self.hotspot_finder.process_batch(batch)
        _f_name=self.hotspot_finder.name
        freq_threshold=adv.get(f'{_f_name}-freq-threshold')
        hotspot_list = self.hotspot_finder.to_target_hotspot_residues_list(
                freq_threshold=freq_threshold
            )
        # hotspots_topk_motifs(
        #     pdb=self.settings.target_settings.full_target_pdb,
        #     hotspot_list=hotspot_list,
        #     output_dir=
        # )