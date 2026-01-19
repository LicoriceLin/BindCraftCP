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
            val=input.get_metrics(k)
            if val is not None:
                ret[p+k]=_check_metric(val,threshold)
            else:
                ret[p+k]=False
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
        

def _check_metric(val:bool|int|float,threshold:dict)->bool:
    if isinstance(val,bool):
        return val == threshold['higher']
    else:
        return threshold['higher'] == (val >= threshold['threshold'])
    


    