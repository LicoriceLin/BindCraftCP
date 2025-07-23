from ..basestep import BaseStep,GlobalSettings,DesignRecord,DesignBatch,NEST_SEP
from abc import abstractmethod
from functools import partial
from typing import Callable,Dict,Any
class BaseScorer(BaseStep):
    '''
    a proxy between score function and design pipeline.
    fetch function params from global settings.
    dump scores into records.

    conserved keywords in score_func:
    pdb_to_take/metrics_prefix 
        (will automatically updated when calling config_metrics_prefix/config_pdb_input_key)
    override these two method if the behavior is not default.

    '''
    def __init__(self, settings:GlobalSettings,score_func:Callable):
        self._score_func=score_func
        super().__init__(settings)
        self.params:Dict[str,Any]={}
        self._init_params()
        
    def config_pdb_input_key(self,pdb_to_take:str|None=None):
        '''
        `pdb_to_take` will be updated to self.params.
        '''
        super().config_pdb_input_key(pdb_to_take)
        self.config_params(pdb_to_take=self.pdb_to_take)

    def config_metrics_prefix(self,metrics_prefix:str|None=None):
        '''
        `metrics_prefix` will be updated to self.params.
        '''
        super().config_pdb_input_key(metrics_prefix)
        self.config_params(metrics_prefix=self.metrics_prefix)

    @property
    def _default_metrics_prefix(self):
        if isinstance(self.pdb_to_take,str) and self.pdb_to_take:
            return self.pdb_to_take+NEST_SEP
        else:
            return ''
        
    @abstractmethod
    def _init_params(self):
        '''
        load scorer-related parameters from settings.
        '''
        pass
    
    def config_params(self,reinit:bool=False,**kwargs):
        '''
        reload parameters from settings.
        override them with kwargs provided.
        always call this func after params are modified.
        (e.g. global setting is changed, metrics_prefix is reconfigured)
        '''
        if reinit:
            self._init_params()
        for k,v in kwargs.items():
            if k in self.params:
                self.params[k]=v
    
    @property
    def score_func(self)->Callable[[DesignRecord],DesignRecord]:
        return partial(self._score_func,**self.params)

    @property
    def params_keys(self):
        return tuple(self.params.keys())
    
    def process_record(self, input:DesignRecord)->DesignRecord:
        input=self.score_func(input)
        return input
    
    def process_batch(self, 
        input:DesignBatch,**kwargs)->DesignRecord:
        '''
        kwargs: check self.params_keys for accepted params.
        Note: params passed here would overwrite self.params. 
        '''
        self.config_params(**kwargs)
        for records_id,record in input.records.items():
            if input.overwrite or not self.check_processed(record):
                self.process_record(record)
                input.save_record(records_id)
        return input
    

    
    