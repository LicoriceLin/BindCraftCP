from ..basestep import BaseStep,GlobalSettings,DesignRecord,DesignBatch
from abc import abstractmethod
from functools import partial
from typing import Callable,Dict,Any
class BaseScorer(BaseStep):
    '''
    a proxy between score function and design pipeline.
    fetch function params from global settings.
    dump scores into records.
    '''
    def __init__(self, settings:GlobalSettings,score_func:Callable):
        self._score_func=score_func
        super().__init__(settings)
        self.params:Dict[str,Any]={}
        self._init_params()
        
    def config_pdb_input_key(self,pdb_to_take:str|None=None):
        '''
        Note: usually pdb_input_key is passed to `_score_func` by `params`
        So remember to _init_params afer call this func. 
        '''
        super().config_pdb_input_key(pdb_to_take)
        self._init_params()

    @abstractmethod
    def _init_params(self):
        '''
        load scorer-related parameters from settings.
        '''
        pass
    
    def config_params(self,**kwargs):
        '''
        reload parameters from settings.
        override them with kwargs provided.
        always call this func after params are modified.
        (e.g. global setting is changed, metrics_prefix is reconfigured)
        '''
        self._init_params()
        for k,v in kwargs.items():
            if k in self.params:
                self.params[k]=v
    
    @property
    def score_func(self)->Callable[[DesignRecord],DesignRecord]:
        return partial(self._score_func,**self.params)

    def process_record(self, input:DesignRecord)->DesignRecord:
        input=self.score_func(input)
        return input
    
    def process_batch(self, 
        input:DesignBatch,metrics_prefix:str|None=None,**kwargs)->DesignRecord:
        '''
        kwargs: check self.params for expected 
        '''
        self.config_metrics_prefix(metrics_prefix)
        self.config_params(**kwargs)
        for records_id,record in input.records.items():
            if input.overwrite or not self.check_processed(record):
                self.process_record(record)
                input.save_record(records_id)
        return input
    

    
    