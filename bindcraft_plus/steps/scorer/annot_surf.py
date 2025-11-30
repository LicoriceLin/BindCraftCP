from .pymol_utils import rSASA
from .basescorer import BaseScorer,GlobalSettings,DesignRecord,DesignBatch 
from pymol import cmd
from typing import Optional,Dict,Tuple
from pathlib import Path
import numpy as np



def annot_ligand_surf(
    record:DesignRecord,pdb_to_take:Dict[str,str],#ligand_chain:str='B',
    ppi_threshold:float=0.1 ,core_threshold:float=0.5,metrics_prefix:str='',
    del_obj:bool=True)->DesignRecord:
    pdb_file=record.pdb_files[pdb_to_take['pdb_key']]
    ligand_chain=pdb_to_take['binder_chain']
    design_id=record.id+'-complex'
    cmd.load(pdb_file,design_id)
    cmd.h_add(design_id)
    rSASA_in_complex= np.array([i/100 for i in rSASA(design_id,ligand_chain).values()]) 
    cmd.create(record.id+'-monomer',f'{design_id} and chain {ligand_chain}')
    rSASA_in_monomer= np.array([i/100 for i in rSASA(record.id+'-monomer',ligand_chain).values()])  
    # return {'c':rSASA_in_complex,'m':rSASA_in_monomer}
    del_rSASA=rSASA_in_monomer-rSASA_in_complex
    ppi=del_rSASA>ppi_threshold
    core= (~ppi) & (rSASA_in_monomer<=core_threshold)
    surf= (~ppi) & (rSASA_in_monomer>core_threshold)
    if del_obj:
        cmd.delete(design_id)
        cmd.delete(record.id+'-monomer')
    record.ana_tracks[f'{metrics_prefix}rSASA_in_monomer']=[round(i,2) for i in rSASA_in_monomer]
    record.ana_tracks[f'{metrics_prefix}rSASA_in_complex']=[round(i,2) for i in rSASA_in_complex]
    record.ana_tracks[f'{metrics_prefix}ppi']=ppi.astype(int).tolist()
    record.ana_tracks[f'{metrics_prefix}core']=core.astype(int).tolist()
    record.ana_tracks[f'{metrics_prefix}surf']=surf.astype(int).tolist()
    return record


class AnnotSurf(BaseScorer):
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings,score_func=annot_ligand_surf)
        # default_input_key='templated' if self.settings.adv.setdefault('templated',False) else 'halu'
        # self.settings.adv.setdefault(f'{self.name}-pdb-input',default_input_key)
        # TODO register parameters like this in other steps?
        
    @property
    def _default_pdb_input_key(self):
        '''
        {'pdb_key':'halu'/'template','binder_chain'}
        '''
        if self.settings.adv.get('templated',False):
            return {'pdb_key':'halu','binder_chain':self.settings.target_settings.new_binder_chain} #'halu'
        else:
            return {'pdb_key':'template','binder_chain':self.settings.target_settings.full_binder_chain}
    
    def _init_params(self):
        self.params=dict(
            pdb_to_take=self.pdb_to_take,
            ppi_threshold=self.settings.adv.setdefault('ppi_threshold',0.1),
            core_threshold=self.settings.adv.setdefault('core_threshold',0.5),
            metrics_prefix=self.metrics_prefix,
            )
        
    @property
    def name(self)->str:
        return 'surf-annot'
    
    @property
    def params_to_take(self)->Tuple[str,...]:
        ret=[f'{self.name}-prefix',f'{self.name}-pdb-input','templated','ppi_threshold','core_threshold']
        return tuple(ret)
    
    @property
    def _default_metrics_prefix(self):
        return ''
    
    @property
    def track_to_add(self):
        return tuple([f'{self.metrics_prefix}{i}' for i in 
            ['rSASA_in_monomer','rSASA_in_complex','ppi','core','surf']])
    
    # def config_pdb_input_key(self,pdb_to_take:str|None=None):
    #     '''
    #     default: take `template` or `halu`.
    #     '''
    #     if pdb_to_take is None:
    #         self._pdb_to_take='template' if self.settings.adv.get('templated',False) else 'halu'
    #     else:
    #         self._pdb_to_take=pdb_to_take
    #     self.config_params(pdb_to_take=self.pdb_to_take)

