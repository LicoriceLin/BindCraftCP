from abc import ABC,abstractmethod
from dataclasses import dataclass,asdict,MISSING
from typing import List,Dict,Any,Optional
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

    def has_pdb(self,pdb_key:str)->bool:
        if pdb_key in self.pdb_files or  pdb_key in self.pdb_strs:
            return True
        else:
            return False

class DesignBatch:
    cache_dir:Path
    records:Dict[str,DesignRecord]
    overwrite:bool=False
    '''
    Overwrite=False: Cache > Batch > Record,
             =True : Cache < Batch < Record
    '''
    def __init__(self, cache_dir:PathT, overwrite:bool=False):
        self.set_cache_dir(cache_dir)
        self.set_overwrite(overwrite)
        self.records={}
        self.metrics={}

    def _fetch_cache_dir(self,cache_dir:Optional[PathT]=None)->Path:
        if cache_dir is None:
            cache_dir=self.cache_dir
        else:
            cache_dir=Path(cache_dir)
        cache_dir.mkdir(exist_ok=True,parents=True)
        return cache_dir
    
    def set_overwrite(self,overwrite:bool):
        self.overwrite=overwrite

    def set_cache_dir(self,cache_dir:PathT):
        self.cache_dir=self._fetch_cache_dir(cache_dir)

    def load_records(self,cache_dir:PathT=None):
        cache_dir=self._fetch_cache_dir(cache_dir)
        for i in self.cache_dir.iterdir():
            if i.suffix=='.json':
                # under overwrite mode, assume cache data should be discarded, 
                # and should not be used to overwrite in-memory records
                if i.stem!='metrics' and (
                    (not self.overwrite) or (
                    self.overwrite and i.stem not in self.records) 
                    ):
                    self.records[i.stem]=DesignRecord.from_json(i)
                else:
                    with open(i,'r') as f:
                        metrics:dict=json.load(f)
                    if self.overwrite:
                        metrics.update(self.metrics)
                        self.metrics=metrics
                    else:
                        self.metrics.update(metrics)

    @classmethod
    def from_cache(cls,cache_dir:PathT):
        ret=cls(cache_dir,overwrite=False)
        ret.load_records()

    def log(self,logs:Dict[str,Any]):
        self.metrics.update(logs)

    def add_record(self,record:DesignRecord):
        if not self.overwrite:
            assert record.id not in self.records
        self.records[record.id]=record

    def save_record(self,record_id:str,cache_dir:PathT|None=None):
        assert record_id in self.records
        cache_path=self._fetch_cache_dir(cache_dir)/f'{record_id}.json'
        # if not self.overwrite:
        #     assert not cache_path.exists()
        self.records[record_id].to_json(cache_path)

    def save_records(self,cache_dir:PathT=None):
        cache_dir=self._fetch_cache_dir(cache_dir)
        for k,v in self.records.items():
            self.save_record(k,cache_dir)

    def df(self,metrics:List[str]):
        ret=pd.DataFrame(index=self.records.keys())
        for metric in metrics:
            ret[metric]={k:v.metrics[metric] for k,v in self.records.items()}
        return ret

    def __getitem__(self,key:str):
        return self.records[key]

    def __len__(self):
        return len(self.records)
    
