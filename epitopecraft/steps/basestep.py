from abc import ABC,abstractmethod
from colabdesign import mk_afdesign_model
import os
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
PathT=os.PathLike[str]|None

from ..utils import *

import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import List,Dict,Tuple,Optional
from tqdm import tqdm
import pyrosetta as pr

class BaseStep(ABC):
    metrics_prefix:str
    def __init__(self,
        settings:GlobalSettings,**kwargs):
        self.settings=settings
        self.config_metrics_prefix()
        self.config_pdb_input_key()
        self.config_pdb_purge()
        
    @property
    @abstractmethod
    def name(self)->str:
        pass

    @property
    def _default_metrics_prefix(self)->str:
        return f'{self.name}{NEST_SEP}'

    @property
    def _default_pdb_input_key(self)->str:
        return ''
    
    @property
    def metrics_to_add(self)->Tuple[str,...]:
        return tuple([])

    @property
    def pdb_to_add(self)->Tuple[str,...]:
        return tuple([])
    
    @property
    def pdb_to_take(self)->str:
        '''
        scorer's input pdb is frequently changed.
        use `self.set_input_key` to reconfigure it.
        '''
        if not hasattr(self,'_pdb_to_take'):
            self.config_pdb_input_key()
        return self._pdb_to_take
    
    @property
    def params_to_take(self)->Tuple[str,...]:
        '''
        params taken from advanced settings. 
        '''
        return tuple([])
    # params_keys = params_to_take

    @property
    def track_to_add(self)->Tuple[str,...]:
        return tuple([])
    
    def check_processed(self,input: DesignRecord)->bool:
        '''
        default check behavior:
        check if all keys in {metrics,pdb,track}_to_add 
            can be find in records.
        True: already processed; 
        False: some keys are missing.
        '''
        for i in self.metrics_to_add:
            if not input.has_metric(i):
                return False
        for i in self.pdb_to_add:
            if not input.has_pdb(i):
                return False
        for i in self.track_to_add:
            if i not in input.ana_tracks or not input.ana_tracks[i]:
                return False
        return True
        
    @abstractmethod
    def process_record(self, input: DesignRecord|None=None)->DesignRecord|None:
        '''
        in-memory method to deal with Record.
        DON'T generate permanent files here.
        '''
        pass

    def process_batch(self, 
        input: DesignBatch|None=None,
        pdb_purge_stem:Optional[str]=None,
        pdb_to_take:str|None=None,
        metrics_prefix:str|None=None
        )->DesignBatch|None:
        '''
        Main interface for this step.
        expected operations here:
        Config:
            metrics_prefix: prefix in keys to save pdb/metrics
            pdb_purge_stem: path to dump pdbs
        IO:
            check check_processed / forced overwrite
            Save records / batch metrics 
            Purge pdb.
        Batch level operation:
            sort, filter(in Filter subclass), post-analysis.
        '''
        if pdb_purge_stem is not None:
            self.config_pdb_purge(pdb_purge_stem)
        if metrics_prefix is not None:
            self.config_metrics_prefix(metrics_prefix)
        if pdb_to_take is not None:
            self.config_pdb_input_key(pdb_to_take)
        for records_id,record in tqdm(input.records.items(), desc=self.name):
            if input.overwrite or not self.check_processed(record):
                breakpoint() # break4
                self.process_record(record)
                self.purge_record(record)
                input.save_record(records_id)
        return input
    
    def purge_record(self,record:DesignRecord):
        '''
        default purge behavior:
            if self.pdb_purge_dir not None:
                dump record.pdb_strs[key in `self.pdb_to_add`] to it. 
        '''
        # pdb_key=self.metrics_prefix.strip(NEST_SEP)
        if self.pdb_purge_dir is not None:
            for pdb_key in self.pdb_to_add:
                record.purge_pdb(pdb_key,self.pdb_purge_dir/f'{record.id}.pdb')

    def config_pdb_purge(self,pdb_purge_stem:Optional[str]=None):
        '''
        default behavior:
            if pdb_purge_stem is not None, create it and set it to `pdb_purge_dir`
                generated pdb will be dumped into this dir.
            otherwise, generated pdb will be saved in record.
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
        Note: usually metrics_prefix ends with `NEST_SEP`
        '''
        if metrics_prefix is not None:
            self.settings.adv[f'{self.name}-prefix']=metrics_prefix
        self.metrics_prefix=self.settings.adv.setdefault(
            f'{self.name}-prefix',self._default_metrics_prefix)
    

    def config_pdb_input_key(self,pdb_to_take:str|None=None):
        '''
        value in params > value in settings > default value
        '''
        if pdb_to_take is not None:
            self.settings.adv[f'{self.name}-pdb-input']=pdb_to_take
        self._pdb_to_take=self.settings.adv.setdefault(f'{self.name}-pdb-input',self._default_pdb_input_key)

        # if pdb_to_take is None:
        #     self._pdb_to_take=self.settings.adv.setdefault(f'{self.name}-pdb-input','')
        # else:
        #     self._pdb_to_take=pdb_to_take
        #     self.settings.adv[f'{self.name}-pdb-input']=pdb_to_take

    @contextmanager
    def record_time(self,record:DesignRecord,time_key:Optional[str]=None):
        '''
        default time_key: 
            f'time{NEST_SEP}{self.metrics_prefix.strip(NEST_SEP)}'
        '''
        if time_key is None:
            time_key=f'time{NEST_SEP}{self.metrics_prefix.strip(NEST_SEP)}'
        start = perf_counter()
        record.set_metrics(time_key,start)
        yield None  
        start=record.get_metrics(time_key)
        record.set_metrics(time_key,perf_counter() - start)

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
