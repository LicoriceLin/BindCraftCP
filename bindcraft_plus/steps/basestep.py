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

    @abstractmethod
    def __call__(self, input: DesignBatch|None)->DesignBatch:
        pass

    def config_pdb_purge(self,pdb_purge_stem:Optional[str]=None):
        if pdb_purge_stem is not None:
            pdb_purge_dir=Path(self.settings.binder_settings.design_path)/pdb_purge_stem
            pdb_purge_dir.mkdir(exist_ok=True,parents=True)
            self.pdb_purge_dir=pdb_purge_dir
        else:
            self.pdb_purge_dir=None

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