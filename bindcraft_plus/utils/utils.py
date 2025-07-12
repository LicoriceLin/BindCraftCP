from typing import Tuple,Any,Dict,Callable
NEST_SEP=':'
def _iterate_get(d:dict,keys:str|Tuple[str,...],sep:str=NEST_SEP):
    if isinstance(keys,str):
        keys=keys.split(sep)
    for k in keys:
        d=d[k]
    return d

def _iterate_set(d:dict,keys:str|Tuple[str,...],v:Any,sep:str=NEST_SEP):
    if isinstance(keys,str):
        keys=keys.split(sep)
    current=d
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    current[keys[-1]] = v

def map_nested_common_keys(d1:dict, d2:dict, func:Callable):
    if isinstance(d1, dict) and isinstance(d2, dict):
        return {
            k: map_nested_common_keys(d1[k], d2[k], func)
            for k in d1.keys() & d2.keys()
        }
    else:
        return func(d1, d2)
    
def flatten_dict(d:dict, parent_key=''):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}:{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items    

def unflatten_dict(d:Dict[str,Any], sep=NEST_SEP):
    result = {}
    for flat_key, value in d.items():
        keys = flat_key.split(sep)
        current = result
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    return result

