from abc import ABC,abstractmethod
from dataclasses import dataclass,asdict,MISSING
from typing import List,Dict,Any
from dataclasses import dataclass,field
import os,json
import pandas as pd
from pathlib import Path
PathT=os.PathLike[str]|None

@dataclass
class DesignRecord:
    id:str
    sequence:str
    pdb_strs: Dict[str,str] = field(repr=False, default_factory=dict)
    pdb_files: Dict[str,str] = field(repr=False, default_factory=dict)
    metrics:Dict[str,Any] = field( default_factory=dict)
    ana_tracks:Dict[str,List[Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_json(cls, jsonfile: PathT):
        with open(jsonfile,'r') as f:
            return cls.from_dict(json.load(f))
    
    def to_json(self,jsonfile: PathT):
        with open(jsonfile,'w') as f:
            json.dump(self.to_dict(),f,indent=2)
        
    def purge_pdb(self,pdb_key:str,pdbfile:PathT):
        pdb_str=self.pdb_strs.pop(pdb_key)
        with open(pdbfile,'w') as f:
            f.write(pdb_str)
        self.pdb_files[pdb_key]=str(pdbfile)

    def cache_pdb(self,pdb_key:str,pdbfile:PathT):
        pdb_str=self.pdb_strs[pdb_key]
        with open(pdbfile,'w') as f:
            f.write(pdb_str)
        self.pdb_files[pdb_key]=str(pdbfile)

    def load_pdb(self,pdb_key:str,pdbfile:str|None=None):
        if pdbfile is None:
            pdbfile=self.pdb_files[pdb_key]
        else:
            self.pdb_files[pdb_key]=pdbfile
        self.pdb_strs[pdb_key]=open(pdb_key,'r').read()

class DesignBatch:
    cache_dir:Path
    records:Dict[str,DesignRecord]
    overwrite:bool=False
    '''
    Overwrite=False: Cache > Batch > Record,
             =True : Cache < Batch < Record
    '''
    def __init__(self,
        cache_dir:os.PathLike[str], overwrite:bool=False):
        self.cache_dir = Path(cache_dir)
        self.overwrite=overwrite
        self.records={}
        self.fetch_cache_dir()

    def fetch_cache_dir(self,cache_dir:PathT=None)->Path:
        if cache_dir is None:
            cache_dir=self.cache_dir
        else:
            cache_dir=Path(cache_dir)
        cache_dir.mkdir(exist_ok=True,parents=True)
        return cache_dir

    def load_records(self,cache_dir:PathT=None):
        cache_dir=self.fetch_cache_dir(cache_dir)
        for i in self.cache_dir.iterdir():
            if i.suffix=='.json' and (
                (self.overwrite and i.stem not in self.records) 
                or not self.overwrite):
                # under overwrite mode, assume cache data should be discarded, 
                # and should not be used to overwrite in-memory records
                self.records[i.stem]=DesignRecord.from_json(i)

    def add_record(self,record:DesignRecord):
        if not self.overwrite:
            assert record.id not in self.records
        self.records[record.id]=record

    def save_record(self,record_id:str,cache_dir:PathT|None=None):
        assert record_id in self.records
        cache_path=self.fetch_cache_dir(cache_dir)/f'{record_id}.json'
        # if not self.overwrite:
        #     assert not cache_path.exists()
        self.records[record_id].to_json(cache_path)

    def save_records(self,cache_dir:PathT=None):
        cache_dir=self.fetch_cache_dir(cache_dir)
        for k,v in self.records.items():
            self.save_record(k,cache_dir)

    def set_overwrite(self,overwrite:bool):
        self.overwrite=overwrite

    def df(self,metrics:List[str]):
        ret=pd.DataFrame(index=self.records.keys())
        for metric in metrics:
            ret[metric]={k:v.metrics[metric] for k,v in self.records.items()}
        return ret

    def __getitem__(self,key:str):
        return self.records[key]

# InputT = TypeVar("InputT", bound=DesignBatch)
# OutputT = TypeVar("OutputT", bound=DesignBatch)

