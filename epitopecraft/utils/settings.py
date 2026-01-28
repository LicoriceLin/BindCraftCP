from abc import ABC
from dataclasses import dataclass,asdict,MISSING
from collections.abc import Iterable as Iterable_
from typing import List,Dict,Optional,Any,Iterable
from dataclasses import dataclass,field,fields
from .utils import _iterate_set,_iterate_get,NEST_SEP
from string import ascii_uppercase
import os,json,yaml
from pathlib import Path,PurePath

yaml.add_representer(
    PurePath,
    lambda dumper, data: dumper.represent_scalar(
        'tag:yaml.org,2002:str', str(data)
    )
)


def _is_yaml_scalar(x):
    return isinstance(
        x,
        (str, int, float, bool, type(None), PurePath)
    )

def represent_list(dumper, data):
    flow = all(_is_yaml_scalar(x) for x in data)
    return dumper.represent_sequence(
        'tag:yaml.org,2002:seq',
        data,
        flow_style=flow,
    )

yaml.add_representer(list,represent_list)

PathT=os.PathLike[str]|None
BasicDict=Dict[str,str|bool|int|float]

def _default_serialize(i):
    if isinstance(i,Path):
        return i.absolute()
    else:
        return str(i)
    
def _load_json_or_yaml(file:PathT)->dict:
    suffix=Path(file).suffix
    if suffix=='.json':
        with open(file,'r') as file:
            return json.load(file)
    elif suffix=='.yaml':
        with open(file,'r') as file:
            return  yaml.load(file, Loader=yaml.SafeLoader)
    else:
        raise ValueError('only support .json & .yaml')

def _dump_json_or_yaml(d:dict,file:PathT):
    suffix=Path(file).suffix
    if suffix=='.json':
        with open(file,'w') as f:
            json.dump(d,f,indent=2,default=str)
    elif suffix=='.yaml':
        with open(file,'w') as f:
            yaml.dump(d,f,indent=2,sort_keys=False)
    else:
        raise ValueError('only support .json & .yaml')
    
@dataclass
class BaseSettings(ABC):
    @property
    def settings(self):
        return asdict(self)
    
    def save(self,file:PathT):
        _dump_json_or_yaml(self.settings,file)

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
    def from_file(cls,file:PathT):
        '''
        from .json or .yaml file
        '''
        ori_dict=_load_json_or_yaml(file)
        return cls.from_dict(ori_dict)

    @classmethod
    def from_json(cls,jsonfile:PathT):
        '''
        legacy
        '''
        with open(jsonfile,'r') as file:
            ori_dict:dict=json.load(file)
        return cls.from_dict(ori_dict)
    
    @classmethod
    def from_yaml(cls,yamlfile:PathT):
        '''
        legacy
        '''
        with open(yamlfile,'r') as file:
            ori_dict:dict=yaml.load(file, Loader=yaml.SafeLoader)
        return cls.from_dict(ori_dict)
    

@dataclass
class TargetSettings(BaseSettings):
    '''
    Following configurations only works for epitope-only design:
    full_target_pdb: path to intact target pdb
    full_target_chain: chain id of intact target
    full_binder_chain: the binder chain in hallu/refold output, usually B
    new_binder_chain: the binder chain in grafted template. 
                      Introduced to handle the situation where 'B' is in full_target_chain. 
                      No need to be initialized. 
    '''
    starting_pdb:Optional[str]=None # for boltzgen design, only full_target_pdb/full_target_chain is needed
    chains:Optional[str]=None
    target_hotspot_residues:Optional[str]=None
    full_target_pdb:Optional[str]=None
    full_target_chain:Optional[str]=None
    full_binder_chain:str=field(repr=False,init=False,default='B')
    new_binder_chain:str=field(repr=False,init=False,default='')
    def __post_init__(self):
        if self.starting_pdb is None and self. full_target_pdb is None:
            raise ValueError('`starting_pdb` essential for ColabDesign and `full_target_pdb` essential for BoltzGen')
        # if self.starting_pdb is None:
        #     self.starting_pdb=self.full_target_pdb
        if self.full_target_chain is not None:
            if self.full_binder_chain in self.full_target_chain:
                for i in ascii_uppercase:
                    if i !='A' and i!= self.full_binder_chain and i not in self.full_target_chain.split(','):
                        self.new_binder_chain=i
                        break
            else:
                self.new_binder_chain=self.full_binder_chain
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
    _settings: dict = field(init=True, repr=False, default_factory=dict)

    def __post_init__(self):
        if len(self._settings)==0:
            for file in self.advanced_paths:
                if os.path.isfile(file):
                    self._settings.update(_load_json_or_yaml(file))
                else:
                    if file.lower() != 'none':
                        Warning(f'Advanced settings file {file} not found!')
                self._settings.update(self.extra_patch)
        else:
            Warning('Advanced settings already provided, skipping loading from files.')

    @property
    def settings(self):
        return self._settings
    
    @classmethod
    def from_file(cls,file:PathT):
        ret=cls(advanced_paths=[str(file)])
        return ret

@dataclass
class FilterSettings(BaseSettings): 
    filters_path:Optional[str]=None
    thresholds:dict = field(init=True, repr=True, default_factory=dict)
    recipes:dict = field(init=True, repr=True, default_factory=dict)
    '''
    `threshold` `recipe` would override contents in `filters_path`
    '''
    
    @classmethod
    def from_file(cls,file:PathT):
        return cls(filters_path=str(file))
    
    def __post_init__(self):
        if self.filters_path is not None and self.filters_path.lower() != 'none':
            self.filters_path=str(self.filters_path)
            load_json=_load_json_or_yaml(self.filters_path)
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
                ret[element]=i
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
    # _settings: dict = field(init=False, repr=False, default_factory=dict)

    # @property
    # def settings(self):
    #     if not self._settings:
    #         for f in fields(self):
    #             if issubclass(f.type,BaseSettings):
    #                 self._settings.update(getattr(self,f.name).settings)
    #     return self._settings

    @property
    def adv(self):
        return self.advanced_settings.settings

    @classmethod
    def from_dict(cls,d:dict):
        d_={}
        for f in fields(cls):
            if issubclass(f.type,BaseSettings):
                d_[f.name]=f.type.from_dict(d[f.name])
        return cls(**d_)
    
    # @classmethod
    # def load_json(cls,jsonfile:PathT):
    #     with open(jsonfile,'r') as file:
    #         ori_json:dict=json.load(file)
    #     return cls.from_dict(ori_json)
        # d={}
        # for f in fields(cls):
        #     if issubclass(f.type,BaseSettings):
        #         d[f.name]=f.type.from_dict(ori_json)
        # return cls(**d)
    
