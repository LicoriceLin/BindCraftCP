from abc import ABC,abstractmethod
from colabdesign import mk_afdesign_model
import os
from pathlib import Path
PathT=os.PathLike[str]|None

from ..utils import *

import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import List,Dict,Tuple,Optional

# from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages,target_pdb_rmsd,three_to_one_map
# from .pyrosetta_utils import pr_relax, align_pdbs,unaligned_rmsd,score_interface
# from .generic_utils import update_failures,BasicDict,backup_if_exists,backuppdb_if_multiframe

class BaseStep(ABC):
    def __init__(self,
        settings:GlobalSettings,**kwargs):
        self.settings=settings
        # initial prefix: adv[f'{self.name}-prefix'] > self._default_metrics_prefix
        self.config_metrics_prefix() 

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def _default_metrics_prefix(self):
        pass


    def check_processed(self,input: DesignRecord)->bool:
        '''
        default purge behavior:
            check if metrics_prefix.strip('-') in record.pdb_strs or pdb_files
        '''
        pdb_key=self.metrics_prefix.strip('-')
        if input.has_pdb(pdb_key):
            return True
        else:
            return False
        
    @abstractmethod
    def process_record(self, input: DesignRecord|None=None)->DesignRecord|None:
        '''
        Don't purge output pdb here!
        '''
        pass

    @abstractmethod
    def process_batch(self, 
        input: DesignBatch|None=None,
        pdb_purge_stem:Optional[str]=None,
        metrics_prefix:str|None=None
        )->DesignBatch|None:
        '''
        '''
        self.config_pdb_purge(pdb_purge_stem)
        self.config_metrics_prefix(metrics_prefix)
        return input
    
    def purge_record(self,record:DesignRecord):
        '''
        default purge behavior:
            dump record.pdb_strs[metrics_prefix.strip('-')] to self.pdb_purge_dir if it exists
        '''
        pdb_key=self.metrics_prefix.strip('-')
        if self.pdb_purge_dir is not None:
            record.purge_pdb(pdb_key,self.pdb_purge_dir/f'{record.id}.pdb')

    def config_pdb_purge(self,pdb_purge_stem:Optional[str]=None):
        '''
        default behavior:
            if pdb_purge_stem is not None, create it and set it to `pdb_purge_dir`
        '''
        if pdb_purge_stem is not None:
            pdb_purge_dir=Path(self.settings.binder_settings.design_path)/pdb_purge_stem
            pdb_purge_dir.mkdir(exist_ok=True,parents=True)
            self.pdb_purge_dir=pdb_purge_dir
        else:
            self.pdb_purge_dir=None

    def config_metrics_prefix(self,metrics_prefix:Optional[str]=None):
        '''
        metrics_prefix > adv[f'{self.name}-prefix'] > self._default_metrics_prefix
        '''
        if metrics_prefix is not None:
            self.metrics_prefix=metrics_prefix
        else:
            self.metrics_prefix=self.settings.adv.get(
                f'{self.name}-prefix',self._default_metrics_prefix)



def add_cyclic_offset(self:mk_afdesign_model, offset_type=2):
    '''add cyclic offset to connect N and C term'''
    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i,i+L],-1)
        offset = i[:,None] - i[None,:]
        c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
        if offset_type == 1:
            c_offset = c_offset
        elif offset_type >= 2:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
        if offset_type == 3:
            idx = np.abs(c_offset) > 2
            c_offset[idx] = (32 * c_offset[idx] )/  abs(c_offset[idx])
        return c_offset * np.sign(offset)
    idx = self._inputs["residue_index"]
    offset = np.array(idx[:,None] - idx[None,:])

    if self.protocol == "binder":
        c_offset = cyclic_offset(self._binder_len)
        offset[self._target_len:,self._target_len:] = c_offset

    elif self.protocol in ["fixbb","hallucination"]:
        Ln = 0
        for L in self._lengths:
            offset[Ln:Ln+L,Ln:Ln+L] = cyclic_offset(L)
            Ln += L

    elif self.protocol=="partial":
        print("Under Construction")
        raise NotImplementedError
    else:
        raise ValueError

    self._inputs["offset"] = offset