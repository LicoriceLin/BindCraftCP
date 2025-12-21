from abc import ABC,abstractmethod
from dataclasses import dataclass,asdict,MISSING
from typing import List,Dict,Any,Optional,Iterable,Tuple,Callable
from dataclasses import dataclass,field
import os,json
import pandas as pd
from pathlib import Path
from .utils import NEST_SEP, _iterate_get, _iterate_set,_iterate_in,_iterate_get_safe
from logging import warning
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

    def get_metrics(self,key:str|Tuple[str,...],default:Any=None):
        f'''
        key: 
        common str      : self.metrics[key]:
        str with {NEST_SEP}    : split into tuple
        tuple           : self.metrics[k1][k2][...]
        '''
        return _iterate_get_safe(self.metrics,key,default=default)
        
    def set_metrics(self,key:str|Tuple[str,...],value:Any):
        return _iterate_set(self.metrics,key,value)
 
    def has_metric(self,key:str|Tuple[str,...]):
        return _iterate_in(self.metrics,key)
        
    def update_metrics(self,metrics:Dict[str|Tuple[str,...],Any]):
        for k,v in metrics.items():
            self.set_metrics(k,v)

class DesignBatch:
    cache_dir:Path
    records:Dict[str,DesignRecord]
    overwrite:bool=False
    metrics: Dict[str,Any]={}
    '''
    Overwrite=False: Cache > Batch > New,
             =True : Cache < Batch < New
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

    def load_records(self,cache_dir:Optional[PathT]=None):
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
        return ret

    def log(self,logs:Dict[str,Any]):
        if self.overwrite:
            self.metrics.update(logs)
        else:
            logs.update(self.metrics)
            self.metrics=logs

    def add_record(self,record:DesignRecord):
        if not self.overwrite:
            assert (record.id not in self.records) or self[record.id] is record
        self.records[record.id]=record

    def save_record(self,record_id:str,cache_dir:PathT|None=None):
        '''
        this method is not protected by `overwrite`,
        as re-writing can be frequent.
        '''
        assert record_id in self.records
        cache_path=self._fetch_cache_dir(cache_dir)/f'{record_id}.json'
        # if not self.overwrite:
        #     assert not cache_path.exists()
        self.records[record_id].to_json(cache_path)

    def save_records(self,cache_dir:PathT=None):
        cache_dir=self._fetch_cache_dir(cache_dir)
        for k,v in self.records.items():
            self.save_record(k,cache_dir)
        self.save_log()

    def save_log(self):
        '''
        save self.metrics to self.log_json
        if not self.metrics: do nothing.
        '''
        if not self.metrics:
            return
        if not self.overwrite:
            if self.log_json.exists():
                prev_log=json.load(open(self.log_json,'r'))
                for i in self.metrics:
                    if i in prev_log:
                        raise ValueError(f'not in overwrite mode but metrics {i} exists')
        with open(self.log_json,'w') as f:
            json.dump(self.metrics,f,indent=2)
        
    def df(self,metrics:Iterable[str|Tuple[str,...]]):
        ret=pd.DataFrame(index=self.records.keys())
        for metric in metrics:
            ret[metric]={k:v.get_metrics(metric) for k,v in self.records.items()}
        return ret
    
    def to_fasta(self,outfile:str|None=None)->str|None:
        '''
        outfile is None: return the fasta str.
        otherwise: return None, save str to file. 
        '''
        o=[]
        for k,v in self.records.items():
            o.append(f'>{k}\n{v.sequence}')
        ret='\n'.join(o)
        if outfile is None:
            return ret
        else:
            with open(outfile,'w') as f:
                f.write(ret)

    @property
    def log_json(self):
        return self.cache_dir/'metrics.json'

    def __getitem__(self,key:str|int|Iterable[str]):
        if isinstance(key,str):
            return self.records[key]
        elif isinstance(key,int):
            return list(self.records.values())[key]
        else:
            return DesignBatchSlice(self,key)
        
    def __len__(self):
        return len(self.records)
    
    def keys(self):
        return self.records.keys()

    def select_record(self,select_fn:Callable[[DesignRecord],bool])->List[str]:
        ret = []
        for design_id,record in self.records.items():
            if select_fn(record):
                ret.append(design_id)
        return ret

    def filter(self,select_fn:Callable[[DesignRecord],bool],
            new_log_stem:Optional[str]=None)->'DesignBatchSlice':
        record_ids=self.select_record(select_fn)
        ret:DesignBatchSlice=self[record_ids]
        ret.set_log_stem(new_log_stem)
        return ret
    
    def apply(self,fn:Callable[[DesignRecord],None]):
        for k,v in self.records.items():
            fn(v)

    def shallow_copy(self,cache_dir:PathT)->'DesignBatch':
        '''
        construct softlink from disk.
        won't saving data in memory to disk.
        '''
        cache_dir=Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        for k,v in self.records.items():
            ori=self.cache_dir/f'{v.id}.json'
            new_record=cache_dir/f'{v.id}.json'
            if new_record.exists():
                if os.path.islink(new_record):
                    continue
                else:
                    warning(f'exsiting, non-softlink record: {v.id}')
            else:
                os.symlink(ori.absolute(),new_record)
        if self.log_json.exists():
            new_log=cache_dir/(self.log_json.stem+'.json')
            if not new_log.exists():
                os.symlink(self.log_json.absolute(),new_record)
        return DesignBatch.from_cache(cache_dir)


class DesignBatchSlice(DesignBatch):
    '''
    a shallow copy of DesignBatch subset,
    purpose:
        choose a subset for further process 
        maybe more scores/conformation will be added, for example, in scorer
        maybe more designs will be added, for example, in MPNN
        maybe just for analysis. In this case, `log_json` could be different.

    inherit cache_dir, records(subset), metrics,
    can not be loaded from cache_dir

    overwrite & cache_dir: always same as its parent
    '''
    def __init__(self, parent:DesignBatch,
        record_ids:Iterable[str],log_stem:str|None=None):
        if isinstance(parent,DesignBatchSlice):
            self.parent = parent.parent
        else:
            self.parent = parent
        self.metrics = {}
        self.records={i:parent.records[i] for i in record_ids}
        self.set_log_stem(log_stem)

    def add_record(self,record:DesignRecord):
        self.parent.add_record(record)
        super().add_record(record)

    def set_log_stem(self,log_stem:str|None=None):
        self._log_stem=log_stem    
    
    def push_log(self):
        self.parent.metrics.update(self.metrics)

    def pull_log(self):
        self.metrics.update(self.parent.metrics)

    @property
    def log_json(self):
        if self._log_stem is None:
            return self.parent.log_json
        else:
            return self.cache_dir/f'{self._log_stem}.json'

    @property
    def overwrite(self):
        return self.parent.overwrite
    
    @property
    def cache_dir(self):
        return self.parent.cache_dir
    
    ### inhibited methods
    def load_records(self,cache_dir):
        raise NotImplementedError("This is a Slice. Don't load to it.")
    
    @classmethod
    def from_cache(cls,cache_dir):
         raise NotImplementedError("This is a Slice. Don't load to it.")
    
    def set_cache_dir(self, cache_dir):
        raise NotImplementedError("This is a Slice. Set on its parent.")

    def set_overwrite(self, overwrite):
        raise NotImplementedError("This is a Slice. Set on its parent.")
    

