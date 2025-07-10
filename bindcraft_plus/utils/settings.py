from abc import ABC
from dataclasses import dataclass,asdict,MISSING
from typing import List,Dict,Optional
from dataclasses import dataclass,field,fields
import os,json
PathT=os.PathLike[str]|None

@dataclass
class BaseSettings(ABC):
    @property
    def settings(self):
        return asdict(self)
    
    def save(self,jsonfile:PathT):
        with open(jsonfile,'w') as f:
            json.dump(self.settings,f,indent=2)

    @classmethod
    def from_dict(cls,d:dict):
        d_={}
        for f in fields(cls):
            if f.init:
                v=d.get(f.name,f.default)
                assert v is not MISSING, f'{f.name} missing!'
                d_[f.name]=v
        return cls(**d_)

    @classmethod
    def from_json(cls,jsonfile:PathT):
        with open(jsonfile,'r') as file:
            ori_dict:dict=json.load(file)
        return cls.from_dict(ori_dict)
        # d={}
        # for f in fields(cls):
        #     v=ori_json.get(f.name,f.default)
        #     assert v is not MISSING
        #     d[f.name]=v
        # return cls(**d)

        

@dataclass
class TargetSettings(BaseSettings):
    starting_pdb:str
    chains:str
    target_hotspot_residues:Optional[str]=None
    full_target_pdb:Optional[str]=None
    full_target_chain:Optional[str]=None
    full_binder_chain:str='B'


@dataclass
class BinderSettings(BaseSettings):
    design_path:str
    binder_name:str
    binder_lengths:List[int]
    random_seeds:List[int]
    helix_values:Optional[List[int]]=None
    global_seed:int=42


BasicDict=Dict[str,str|bool|int|float]

@dataclass
class AdvancedSettings(BaseSettings):
    advanced_paths:List[str] #sequentially override
    extra_patch: dict = field(repr=False, default_factory=dict)
    _settings: dict = field(init=False, repr=False, default_factory=dict)
    @property
    def settings(self):
        if not self._settings:
            for file in self.advanced_paths:
                with open(file, 'r') as file:
                    self._settings.update(json.load(file))
            self._settings.update(self.extra_patch)
            self._settings['advanced_paths']=self.advanced_paths
            self._settings['extra_patch']=self.extra_patch
        return self._settings
    
@dataclass
class FilterSettings(BaseSettings): 
    filters_path:str
    _settings: dict = field(init=False, repr=False, default_factory=dict)
    # TODO interface AA is a disaster
    @property
    def settings(self):
        if not self._settings:
            with open(self.filters_path, 'r') as file:
                self._settings.update(json.load(file))
        self._settings['filters_path']=self.filters_path
        return self._settings
    
    def __getitem__(self,key):
        return getattr(self,key)

@dataclass
class GlobalSettings(BaseSettings):
    target_settings:TargetSettings
    binder_settings:BinderSettings
    advanced_settings:AdvancedSettings
    filter_settings:FilterSettings
    _settings: dict = field(init=False, repr=False, default_factory=dict)
    @property
    def settings(self):
        if not self._settings:
            for f in fields(self):
                if issubclass(f.type,BaseSettings):
                    self._settings.update(getattr(self,f.name).settings)
        return self._settings

    @property
    def adv(self):
        return self.advanced_settings.settings

    @classmethod
    def from_dict(cls,d:dict):
        d_={}
        for f in fields(cls):
            if issubclass(f.type,BaseSettings):
                d_[f.name]=f.type.from_dict(d)
        return cls(**d_)
    
    @classmethod
    def load_json(cls,jsonfile:PathT):
        with open(jsonfile,'r') as file:
            ori_json:dict=json.load(file)
        return cls.from_dict(ori_json)
        # d={}
        # for f in fields(cls):
        #     if issubclass(f.type,BaseSettings):
        #         d[f.name]=f.type.from_dict(ori_json)
        # return cls(**d)
    
