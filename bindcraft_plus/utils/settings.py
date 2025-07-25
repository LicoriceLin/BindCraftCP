from abc import ABC
from dataclasses import dataclass,asdict,MISSING
from typing import List,Dict,Optional,Any,Iterable
from dataclasses import dataclass,field,fields
from .utils import _iterate_set,_iterate_get,NEST_SEP
from string import ascii_uppercase
import os,json

PathT=os.PathLike[str]|None
BasicDict=Dict[str,str|bool|int|float]
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
                if v is MISSING:
                    if f.default_factory is not MISSING:
                        v=f.default_factory()
                    else:
                        raise ValueError(f'{f.name} missing!')
                d_[f.name]=v
        return cls(**d_)

    @classmethod
    def from_json(cls,jsonfile:PathT):
        with open(jsonfile,'r') as file:
            ori_dict:dict=json.load(file)
        return cls.from_dict(ori_dict)
    

@dataclass
class TargetSettings(BaseSettings):
    starting_pdb:str
    chains:str
    target_hotspot_residues:Optional[str]=None
    full_target_pdb:Optional[str]=None
    full_target_chain:Optional[str]=None
    full_binder_chain:str='B'
    new_binder_chain:str=field(repr=False,init=False,default='')
    def __post_init__(self):
        if (self.full_target_chain is not None 
            and self.full_binder_chain in self.full_target_chain):
            for i in ascii_uppercase:
                if i !='A' and i!= self.full_binder_chain and i not in self.full_target_chain.split(','):
                    self.new_binder_chain=i
                    break
        else:
            self.full_target_chain='A' # default output from colabdesign.
            self.new_binder_chain=self.full_binder_chain


@dataclass
class BinderSettings(BaseSettings):
    design_path:str
    binder_name:str
    binder_lengths:List[int]
    random_seeds:List[int]
    helix_values:Optional[List[int]]=None
    global_seed:int=42


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
    filters_path:Optional[str]=None
    thresholds:dict = field(init=True, repr=True, default_factory=dict)
    recipes:dict = field(init=True, repr=True, default_factory=dict)
    '''
    `threshold` `recipe` would override contents in `filters_path`
    '''
    
    def __post_init__(self):
        if self.filters_path is not None:
            load_json=json.load(open(self.filters_path,'r'))
            thresholds:dict=load_json.get('thresholds',{})
            recipes:dict=load_json.get('recipes',{})
            if self.thresholds:
                thresholds.update(self.thresholds)
            self.thresholds=thresholds
            if self.recipes:
                recipes.update(self.recipes)
            self.recipes=recipes
    
    def recipe_threshold(self,recipe:str='all'):
        '''
        `recipe` == "all" or `recipe` in self.recipes
        '''
        if recipe == 'all':
            return self._flatten_threshold(None)
        else:
            return self._flatten_threshold(self.recipes[recipe])
            
    def _flatten_threshold(self,subset:Iterable[str]|None)->Dict[str,Dict[str,Any]]:
        if  subset is None:
            return flatten_threshold(self.thresholds)
        else:
            ret = {}
            for element in subset:
                i=_iterate_get(self.thresholds,element)
                # if 'higher' in i:
                ret[element]=i
                # else:
                #     ret.update(flatten_threshold(i))
            return flatten_threshold(ret)

def flatten_threshold(d:Dict[str,Any], parent_key=''):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{NEST_SEP}{k}" if parent_key else str(k)
        if isinstance(v, dict) and 'higher' not in v:
            items.update(flatten_threshold(v, new_key))
        else:
            items[new_key] = v
    return items

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
    
