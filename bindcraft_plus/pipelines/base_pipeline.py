from abc import ABC,abstractmethod
from pathlib import Path
from typing import Dict,Any,Tuple
from ..utils import (
    TargetSettings,GlobalSettings,AdvancedSettings,
    FilterSettings,BinderSettings,NEST_SEP
    )
from ..steps import BaseStep
_dir_path=Path(__file__).parent



class BasePipeline(ABC):
    def __init__(self,
        global_settings:str|GlobalSettings,
        ):
        '''
        '''
        if isinstance(global_settings,GlobalSettings):
            self.settings=global_settings
        else:
            self.settings=GlobalSettings.from_json(global_settings)
        self.design_path=Path(self.settings.binder_settings.design_path)
        self.design_path.mkdir(exist_ok=True)
        self._save_settings()
        self._init_steps()

    @abstractmethod
    def _init_steps(self):
        pass

    @abstractmethod
    def run(self,**args):
        pass
    
    @property
    def steps(self)->Dict[str,BaseStep]:
        ret={}
        for k,v in self.__dict__.items():
            if isinstance(v,BaseStep):
                ret[k]=v
        return ret
    
    @property
    def metrics_to_add(self)->Tuple[str,...]:
        ret=[]
        for k,v in self.steps.items():
            ret.extend(v.metrics_to_add)
        return tuple(ret)
    
    @property
    def pdb_to_add(self)->Tuple[str,...]:
        ret=[]
        for k,v in self.steps.items():
            ret.extend(v.pdb_to_add)
        return tuple(ret)

    @property
    def track_to_add(self)->Tuple[str,...]:
        ret=[]
        for k,v in self.steps.items():
            ret.extend(v.track_to_add)
        return tuple(ret)
    
    @property
    def params_to_take(self)->Dict[str,Tuple[str,...]]:
        ret={}
        for k,v in self.steps.items():
            ret[k]=v.params_to_take
            # ret.extend()
        ret['pipeline']=self.pipeline_params
        # ret.extend(self.pipeline_params)
        return ret #tuple(set(ret))

    @property
    def pipeline_params(self)->Tuple[str,...]:
        return tuple([])

    def _save_settings(self):
        self.settings.save(
            self.design_path/f'{self.settings.binder_settings.binder_name}.json')
