from .basescorer import BaseScorer,GlobalSettings,DesignRecord,DesignBatch 
from .pymol_utils import partial_align
from pymol import cmd
from typing import Optional,Dict
from pathlib import Path
import numpy as np

def annot_rmsd(record:DesignRecord,mobile_pdb:str,mobile_sel:str,target_pdb:str,
    mobile_rms_sel:str|None, target_sel:str|None=None,target_rms_sel:str|None=None,
    prefix:str='',del_obj:bool=True
    )->DesignRecord:
    record_id=record.id
    cmd.load(record.pdb_files[mobile_pdb],f'{record_id}-mobile')
    cmd.load(record.pdb_files[target_pdb],f'{record_id}-target')
    rms=partial_align(f'{record_id}-mobile',mobile_sel,f'{record_id}-target',
        mobile_rms_sel,target_sel,target_rms_sel)
    record.update_metrics({
        f'{prefix}target_rmsd':rms['align_rmsd'],
        f'{prefix}binder_rmsd':rms['obj_rmsd'],
        })
    if del_obj:
        cmd.delete(f'{record_id}-mobile')
        cmd.delete(f'{record_id}-target')
    return record


class AnnotRMSD(BaseScorer):
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings,score_func=annot_rmsd)

    def _init_params(self):
        ts=self.settings.target_settings
        self.params=dict(
            mobile_pdb=self.pdb_to_take['mobile'],
            mobile_sel='chain A',
            target_pdb=self.pdb_to_take['target'],
            mobile_rms_sel=f'chain {ts.full_binder_chain}' , 
            target_sel=f'chain {ts.full_target_chain}',
            target_rms_sel=f'chain {ts.new_binder_chain}',
            prefix=self.metrics_prefix
        )

    @property
    def name(self):
        return 'rmsd'
    
    @property
    def _default_metrics_prefix(self):
        return ''
    
    @property
    def metrics_to_add(self):
        return tuple([self.metrics_prefix+k for k in ['target_rmsd','binder_rmsd']])

    def config_pdb_input_key(self,mobile:str|None=None,target:str|None=None):
        if mobile is None:
            mobile='refold:multimer-1'
        if target is None:
            target = 'template' if self.settings.adv.get('templated',False) else 'halu'
        self._pdb_to_take={"mobile":mobile,'target':target}
        self._init_params()

    @property
    def pdb_to_take(self)->Dict[str,str]:
        '''
        {"mobile":...,'target':...}
        '''
        return self._pdb_to_take
    

    