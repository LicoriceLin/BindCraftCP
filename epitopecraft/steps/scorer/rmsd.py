from .basescorer import BaseScorer,GlobalSettings,DesignRecord,DesignBatch 
from .pymol_utils import partial_align
from pymol import cmd
from typing import Optional,Dict,Tuple
from pathlib import Path
import numpy as np

def annot_rmsd(record:DesignRecord,pdb_to_take:dict,mobile_sel:str,
    mobile_rms_sel:str|None, target_sel:str|None=None,target_rms_sel:str|None=None,
    metrics_prefix:str='',del_obj:bool=True
    )->DesignRecord:
    record_id=record.id
    mobile_pdb,target_pdb=pdb_to_take['mobile'],pdb_to_take['target']
    cmd.load(record.pdb_files[mobile_pdb],f'{record_id}-mobile')
    cmd.load(record.pdb_files[target_pdb],f'{record_id}-target')
    rms=partial_align(f'{record_id}-mobile',mobile_sel,f'{record_id}-target',
        mobile_rms_sel,target_sel,target_rms_sel)
    record.update_metrics({
        f'{metrics_prefix}target_rmsd':rms['align_rmsd'],
        f'{metrics_prefix}binder_rmsd':rms['obj_rmsd'],
        })
    if del_obj:
        cmd.delete(f'{record_id}-mobile')
        cmd.delete(f'{record_id}-target')
    return record


class AnnotRMSD(BaseScorer):
    '''
    default selection of chains:
        mobile from refold, so A+B
        target from template, so ts.full_target_chain+ts.new_binder_chain
    '''
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings,score_func=annot_rmsd)

    def _init_params(self):
        ts=self.settings.target_settings
        self.params=dict(
            pdb_to_take=self.pdb_to_take,
            mobile_sel='chain A',
            mobile_rms_sel=f'chain {ts.full_binder_chain}' , 
            target_sel=f'chain {ts.full_target_chain}',
            target_rms_sel=f'chain {ts.new_binder_chain}',
            metrics_prefix=self.metrics_prefix
            )

    @property
    def name(self):
        return 'rmsd'
    
    @property
    def params_to_take(self)->Tuple[str,...]:
        ret=[f'{self.name}-prefix',f'{self.name}-pdb-input']
        return tuple(ret)
    
    @property
    def metrics_to_add(self):
        return tuple([self.metrics_prefix+k for k in ['target_rmsd','binder_rmsd']])

    def config_pdb_input_key(self,mobile:str|None=None,target:str|None=None,
            pdb_to_take:Dict[str,str]|None=None,_reconfig_params:bool=True):
        '''
        allow extra param of 'mobile' and 'target'.
        '''
        if pdb_to_take is not None:
            mobile=pdb_to_take.get('mobile',mobile)
            target=pdb_to_take.get('target',target)
        if mobile is None:
            mobile='refold:best'
        if target is None:
            target = 'template' if self.settings.adv.get('templated',False) else 'halu'
        super().config_pdb_input_key({"mobile":mobile,'target':target},_reconfig_params)
        # self._pdb_to_take={"mobile":mobile,'target':target}
        # if _reconfig_params:
        #     self.config_params(mobile_pdb=self.pdb_to_take['mobile'],
        #        target_pdb=self.pdb_to_take['target'])
        # if getattr(self,'params',{}) and _reconfig_params:
        #     self.config_params(pdb_to_take=self.pdb_to_take)

    @property
    def _default_pdb_input_key(self)->str:
        target = 'template' if self.settings.adv.get('templated',False) else 'halu'
        return {"mobile":'refold:best','target':target}
    
    @property
    def pdb_to_take(self)->Dict[str,str]:
        '''
        {"mobile":...,'target':...}
        '''
        if not hasattr(self,'_pdb_to_take'):
            self.config_pdb_input_key()
        return self._pdb_to_take
    

    