
from ..steps import (Hallucinate,Filter,Refold,Graft,AnnotRMSD,AnnotGyration,
    AnnotSurf,MPNN,AnnotPolarOccupy,AnnotPTM,AnnotBCAux,AnnotPI,Relax)

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

def init_hallu_settings(
    target_settings:str|None=None,
    binder_settings:str|None=None,
    stage4_montecarlo:bool=True,
    template:bool=True,
    cyclic_peptide:bool=False,
    patch:str|Dict[str,Any]|None=None,
    filters:str|None=None):
    advanced_paths=[_dir_path/'config/base_adv_setting.json']
    if stage4_montecarlo:
        advanced_paths.append(_dir_path/'config/patch_4s_mc.json')
    if template:
        advanced_paths.append(_dir_path/'config/patch_templated.json')
    if cyclic_peptide:
        advanced_paths.append(_dir_path/'config/patch_cyc.json')
    if patch is None:
        patch={}
    else:
        if isinstance(patch,str):
            patch=json.load(open(patch,'r'))
    if filters is None:
        filters = _dir_path/'config/default_filter.yaml'
    
    settings=GlobalSettings(
        target_settings=TargetSettings.from_json(target_settings),
        binder_settings=BinderSettings.from_json(binder_settings),
        advanced_settings=AdvancedSettings(
            advanced_paths=advanced_paths,
            extra_patch=patch),
        filter_settings=FilterSettings(filters_path=filters)
        )
    settings.adv['mpnn_bias_recipe']='config/mpnn-recipes.json'
    return settings

@dataclass
class InternalTargetSetting(BaseSettings):
    template_pdb:str
    template_target_chain:str
    template_binder_chain:str
    best_refold_pdb:str
    relaxed_pdb:str


class HalluDesign(BasePipeline):
    def _init_steps(self):
        settings=self.settings
        self.hallu=Hallucinate(settings)
        self.filter=Filter(settings)
        if self.settings.adv.setdefault('templated',False):
            self.graft=Graft(settings)
        self.refold=Refold(settings)       
        self.mpnn=MPNN(settings)
        self.annot_rmsd=AnnotRMSD(settings) 
        self.annot_surf=AnnotSurf(settings)
        self.annot_polar=AnnotPolarOccupy(settings)
        self.annot_gyr=AnnotGyration(settings)
        self.relax=Relax(self.settings)
        self.annot_aux=AnnotBCAux(self.settings)
        self.annot_pi=AnnotPI(self.settings)

        if self.settings.adv.setdefault('templated',False):
            self.graft=Graft(settings)
        if self.settings.adv.setdefault('annot_ptm',False):
            self.annot_ogly=AnnotPTM(self.settings)
            self.annot_ogly.config_params(ptm_type='O-linked_glycosylation')
            self.annot_ngly=AnnotPTM(self.settings)
            self.annot_ngly.config_params(ptm_type='N-linked_glycosylation')
        self._config_steps()
        self._save_settings()

    def _config_steps(self):
        adv=self.settings.adv
        self.hallu.config_pdb_purge(adv.setdefault('hallu_stem','hallu'))
        self.filter.set_recipe("after:hallucinate")

        if self.settings.adv.setdefault('templated',False):
            self.graft.config_pdb_purge(adv.setdefault('graft_stem','graft'))
            template_pdb=self.graft.pdb_to_add[0]
            tgt=self.settings.target_settings
            template_target_chain,template_binder_chain=tgt.full_target_chain,tgt.new_binder_chain
        else:
            template_pdb=self.hallu.pdb_to_add[0]
            template_target_chain,template_binder_chain='A','B'
        self.refold.config_pdb_input_key(template_pdb)
        self.refold.config_pdb_purge(adv.setdefault('refold_stem','refold'))

        best_refold_pdb=self.refold.metrics_prefix+'best'
        self.annot_rmsd.config_pdb_input_key(
            pdb_to_take={'mobile':best_refold_pdb,'target':template_pdb})
        self.annot_rmsd.config_metrics_prefix(best_refold_pdb+NEST_SEP)
        self.annot_surf.config_pdb_input_key(
            pdb_to_take={'pdb_key':template_pdb, 'binder_chain':template_binder_chain})

        self.relax.config_pdb_input_key({'pdb_key':best_refold_pdb,'binder_chain':'B'})
        self.relax.config_pdb_purge(adv.setdefault('relax_stem','relax'))
        relaxed_pdb=self.relax.pdb_to_add[0]
        self.annot_polar.config_pdb_input_key(relaxed_pdb)

        self.mpnn.config_pdb_input_key(template_pdb)    
        adv["mpnn_binder_chain"]=template_binder_chain
        adv["mpnn_target_chain"]=template_target_chain

        self.annot_pi.config_params(pdb_to_take=relaxed_pdb)
        self.annot_aux.config_params(pdb_to_take={'pdb_key':relaxed_pdb,'binder_chain':'B'})
        
        self._internal_target_setting=InternalTargetSetting(
            template_pdb=template_pdb,
            template_target_chain=template_target_chain,
            template_binder_chain=template_binder_chain,
            best_refold_pdb=best_refold_pdb,
            relaxed_pdb=relaxed_pdb,
            )

    def run(self):
        # breakpoint() # break2
        adv=self.settings.adv
        tgt=self._internal_target_setting
        # hallucination
        batch=self.hallu.process_batch(
            batch_cache_stem=adv.setdefault('metrics_stem','metrics'),
            overwrite=adv.get('overwrite',False))
        # breakpoint() # break3
        batch.parent.set_overwrite(adv.setdefault('overwrite_refold_only',False))
        batch=self.filter.set_recipe("after:hallucinate").process_batch(batch)
        breakpoint() # break_m2

        # graft/refold
        def _graft_refold(batch):
            if self.settings.adv.get('templated',False):
                self.graft.process_batch(batch) 
            self.refold.process_batch(batch)
            return batch
        
        batch=_graft_refold(batch)

        def _score_after_refold(batch):
            self.annot_rmsd.process_batch(batch)
            self.annot_gyr.process_batch(batch)
            batch=self.filter.set_recipe("after:refold").process_batch(batch)
            self.annot_surf.process_batch(batch)
            self.relax.process_batch(batch)
            self.annot_polar.process_batch(batch)
            if self.settings.adv.setdefault('annot_ptm',False):
                self.annot_ogly.process_batch(batch)
                self.annot_ngly.process_batch(batch)
            return batch
        
        batch=_score_after_refold(batch)
        batch=self.mpnn.process_batch(batch)
        batch=_graft_refold(batch)
        batch=_score_after_refold(batch)

        def _final_scores(batch):
            self.annot_aux.process_batch(batch)
            self.annot_pi.process_batch(batch)
            batch=self.filter.set_recipe("final").process_batch(batch)
            return batch
        
        batch=_final_scores(batch)

        self.settings.binder_settings.binder_name='backup-settinga'
        self._save_settings()
        return batch
    
    @property
    def pipeline_params(self)->Tuple[str,...]:
        '''
        metrics_stem:str='metrics',
        hallu_stem:str='hallu',
        refold_stem:str='refold',
        graft_stem:str='graft',
        overwrite:bool=False,
        overwrite_refold_only:bool=False,
        annot_ptm: bool=False
        `overwrite`: rewrite everything;
        `overwrite_refold_only`: only rerun graft-refold
        `templated`: refold w/wo hallucination templates 
        '''
        return tuple([
            'metrics_stem','hallu_stem','refold_stem','annot_ptm',
            'graft_stem','overwrite','overwrite_refold_only'])