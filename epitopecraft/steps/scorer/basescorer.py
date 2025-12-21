from ..basestep import BaseStep,GlobalSettings,DesignRecord,DesignBatch,NEST_SEP
from abc import abstractmethod
from functools import partial
from typing import Callable,Dict,Any
from tqdm import tqdm
from warnings import warn
class BaseScorer(BaseStep):
    '''
    a proxy between score function and design pipeline.
    fetch function params from global settings.
    dump scores into records.

    score_func is the direct processor of `DesignRecord`, take information inside and add new metrics/tracks to it. 
    conserved keywords in score_func:
    pdb_to_take/metrics_prefix 
        (will automatically updated when calling config_metrics_prefix/config_pdb_input_key)
    override these two method if the behavior is not default.

    '''
    def __init__(self, settings:GlobalSettings,score_func:Callable[[DesignRecord],DesignRecord],decoupled:bool=False):
        self._score_func=score_func
        self.params:Dict[str,Any]={}
        self.decoupled=decoupled
        super().__init__(settings)
        self._init_params()
        if self.decoupled:
            self.warn_decoupled_scorer()
        
    def config_pdb_input_key(self,pdb_to_take:str|None=None,_reconfig_params:bool=True):
        '''
        `pdb_to_take` will be updated to self.params if `_reconfig_params`.
        '''
        super().config_pdb_input_key(pdb_to_take)
        if getattr(self,'params',{}) and _reconfig_params:
            self.config_params(pdb_to_take=self.pdb_to_take)

    def config_metrics_prefix(self,metrics_prefix:str|None=None,_reconfig_params:bool=True):
        '''
        `metrics_prefix` will be updated to self.params if `_reconfig_params`. 
        The default_metrics_prefix here is configured in a different strategy: 
        if `self.pdb_to_take` exists and is a string, 
        `self.pdb_to_take+NEST_SEP` will be used as the default prefix; 
        otherwise, it will be '' (e.g., in AnnotRMSD) 
        But same as other steps, it could be read from f'{self.name}-prefix' inadv_settings 
        '''
        super().config_metrics_prefix(metrics_prefix)
        if getattr(self,'params',{}) and _reconfig_params:
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
        initialize default params, which will be passed to self._score_func as **kwargs. 
        '''
        pass
    
    def config_params(self,reinit:bool=False,**kwargs):
        '''
        reset parameters. 
        override them with kwargs provided. 
        always call this func after params are modified. 
        (e.g. global setting is changed, metrics_prefix is reconfigured)
        '''
        if reinit:
            self._init_params()
        for k,v in kwargs.items():
            if k in self.params:
                self.params[k]=v
            if k =='metrics_prefix':
                self.config_metrics_prefix(metrics_prefix=v,_reconfig_params=False)
            if k=='pdb_to_take':
                self.config_pdb_input_key(pdb_to_take=v,_reconfig_params=False)
    
    @property
    def score_func(self)->Callable[[DesignRecord],DesignRecord]:
        return partial(self._score_func,**self.params)

    @property
    def params_keys(self):
        '''
        params to be passed to `self._score_func`
        '''
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
        for records_id,record in tqdm(input.records.items(),
            desc=f'{self.name} on {self.pdb_to_take}'):
            if input.overwrite or not self.check_processed(record):
                self.process_record(record)
                input.save_record(records_id)
        return input
    
    def warn_decoupled_scorer(self):
        warn(f''' {self.name} is minimally coupled with settings. 
        specify params in `process_batch`/`config_params`''')
    
    