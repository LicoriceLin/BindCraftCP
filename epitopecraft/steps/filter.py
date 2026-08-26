from .basestep import *
from typing import Callable,Any
from ..utils.utils import unflatten_dict

class Filter(BaseStep):
    '''
    '''
    current_recipe:str
    def __init__(self, settings):
        super().__init__(settings)
        self.f_set=self.settings.filter_settings
        self.set_recipe('all')

    def set_recipe(self,recipe:str)->"Filter":
        assert recipe=='all' or recipe in self.f_set.recipes,f'unkown recipe: {recipe}'
        self.current_recipe=recipe
        self.current_threshold=self.f_set.recipe_threshold(recipe)
        return self
    
    @property
    def name(self)->str:
        return 'filter'
    
    @property
    def _default_metrics_prefix(self):
        return 'filter:'
    
    @property
    def metrics_to_add(self):
        ms=[self.metrics_prefix+i for i in self.current_threshold.keys()]
        return tuple(ms+[self.metrics_prefix+'sum'])
    
    def process_record(self,input:DesignRecord):
        ret={}
        p=self.metrics_prefix
        for k,threshold in self.current_threshold.items():
            ret[p+k]=_check_metric(input,k,threshold)
        ret[p+'sum']=all(ret.values())
        input.update_metrics(ret)
        return input

    def process_batch(self,input:DesignBatch)->DesignBatchSlice:
        '''
        different from default behavior, Filter will always re-process records.
        '''
        for records_id,record in input.records.items():
            # if input.overwrite or not self.check_processed(record):
            self.process_record(record)
            input.save_record(records_id)
        opt=input.filter(lambda x:x.get_metrics(self.metrics_prefix+'sum'))
        return opt
        

def _check_metric(record:DesignRecord,metric_key:str,threshold:dict)->bool:
    if isinstance(threshold,dict) and 'func' in threshold:
        return _check_metric_custom(record,threshold)
    val=record.get_metrics(metric_key)
    if val is None:
        return False
    if isinstance(val,bool):
        return val == threshold['higher']
    else:
        return threshold['higher'] == (val >= threshold['threshold'])


def _check_metric_custom(record:DesignRecord,threshold:dict)->bool:
    metric_spec=threshold.get('metrics',{})
    metrics=_resolve_metric_spec(record,metric_spec)
    if any(v is None for v in metrics.values()):
        return False

    env={
        'metrics':metrics,
        'value':next(iter(metrics.values())) if len(metrics)==1 else None,
        'abs':abs,
        'all':all,
        'any':any,
        'len':len,
        'max':max,
        'mean':_mean,
        'min':min,
        'sorted':sorted,
        'sum':sum,
        }
    return bool(eval(threshold['func'],{"__builtins__":{}},env))


def _resolve_metric_spec(record:DesignRecord,metric_spec:dict|list|str)->dict[str,Any]:
    if isinstance(metric_spec,dict):
        return {k:record.get_metrics(v) for k,v in metric_spec.items()}
    elif isinstance(metric_spec,list):
        return {f'm{i}':record.get_metrics(v) for i,v in enumerate(metric_spec)}
    elif isinstance(metric_spec,str):
        return {'value':record.get_metrics(metric_spec)}
    else:
        raise TypeError(f'unsupported metric_spec type: {type(metric_spec)}')


def _mean(values)->float:
    values=list(values)
    if len(values)==0:
        raise ValueError('mean() expects at least one value')
    return sum(values)/len(values)
    


    
