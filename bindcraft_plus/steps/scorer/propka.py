from typing import List
from propka.run import single
from .basescorer import BaseScorer,GlobalSettings,DesignRecord,DesignBatch,NEST_SEP

def propka_single(pdbfile:str,binder_chain:str='B',optargs:List[str]=['--quiet']): #'--protonate-all',
    optargs.append(f'-c={binder_chain}')
    o=single(pdbfile,optargs=optargs,write_pka=False)
    pif,piu=o.get_pi()
    profile, [ph_opt, dg_opt], [dg_min, dg_max], [ph_min, ph_max] = (
        o.get_folding_profile())
    pis={
        'pi-fold':pif,'pi-unfold':piu,
        "pH-opt":ph_opt, "dG-opt":dg_opt
        }
    return pis

def propka_record(record:DesignRecord,pdb_to_take:str,
    binder_chain:str='B',optargs:List[str]=['--quiet'], #'--protonate-all',
    metrics_prefix:str='',only_pifold:bool=True):
    optargs.append(f'-c={binder_chain}')
    o=single(record.pdb_files[pdb_to_take],optargs=optargs,write_pka=False)
    pif,piu=o.get_pi()
    profile, [ph_opt, dg_opt], [dg_min, dg_max], [ph_min, ph_max] = (
        o.get_folding_profile())
    if only_pifold:
        record.set_metrics(f'{metrics_prefix}pi-fold',pif)
    else:
        record.update_metrics({
        f'{metrics_prefix}pi-fold':pif,
        f'{metrics_prefix}pi-unfold':piu,
        f"{metrics_prefix}pH-opt":ph_opt,
        f"{metrics_prefix}dG-opt":dg_opt})
    return record

class AnnotPI(BaseScorer):
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings, score_func=propka_record)

    def _init_params(self):
        self.params=dict(
        pdb_to_take=self.pdb_to_take,
        metrics_prefix=self.metrics_prefix,
        binder_chain='B',
        optargs=['--quiet'],
        only_pifold=True)

    @property
    def name(self)->str:
        return 'pI-annot'

    @property
    def metrics_to_add(self):
        if self.params['only_pifold']:
            return tuple([self.metrics_prefix+'pi-fold'])
        else:
            return tuple([self.metrics_prefix+i for i in 
                ['pi-fold','pi-unfold','pH-opt','dG-opt']])
        
    
    
        
