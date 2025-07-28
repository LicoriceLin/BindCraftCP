from .basestep import BaseStep,DesignRecord,DesignBatch,DesignBatchSlice
from ..utils import GlobalSettings,NEST_SEP
from .scorer.diversity_util import simple_diversity,cluster_and_get_medoids
from typing import Callable,Dict,List,Any
import pandas as pd
import numpy as np
import pickle as pkl
import math
import json

from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants

aa_order = residue_constants.restype_order
order_aa = {b:a for a,b in aa_order.items()}
all_aa=''.join(aa_order.keys())

class MPNN(BaseStep):
    def __init__(self,settings:GlobalSettings):
        super().__init__(settings)
        self.init_penalty_recipes()

    @property
    def name(self):
        return 'mpnn'
    
    @property
    def metrics_to_add(self):
        return tuple([self.metrics_prefix+'score',self.metrics_prefix+'seqid'])

    def config_pdb_input_key(self, pdb_to_take = None):
        if pdb_to_take is None:
            if self.settings.adv.get('templated',False):
                self._pdb_to_take = self.settings.adv.get('template-pdb-key','template')
            else:
                self._pdb_to_take = self.settings.adv.get('template-pdb-key','halu')
        else:
            self._pdb_to_take = None

    @property
    def mpnn_model(self):
        if getattr(self,'_mpnn_model',None) is None:
            adv=self.settings.adv
            self._mpnn_model=mk_mpnn_model(
                backbone_noise=adv.get("mpnn_backbone_noise",0.),
                model_name=adv.get("mpnn_model_path","v_48_020"),
                weights=adv.get("mpnn_weights","soluble"),
                seed=self.settings.binder_settings.global_seed)
        return self._mpnn_model
    
    def cal_bias(self,record:DesignRecord):
        track_df=self._get_track_df(record)
        bias=np.zeros((len(record.sequence),21))
        for r in self.penalty_recipes:
            bias=r.add_bias(track_df,bias)
        return bias
    
    def _get_track_df(self,record:DesignRecord):
        track_df=pd.DataFrame(record.ana_tracks)
        track_df['seq']=list(record.sequence)
        return track_df
    
    def init_penalty_recipes(self):
        bias_recipe_file=self.settings.adv.get(
            'mpnn_bias_recipe','config/mpnn-default-recipe.json')
        with open(bias_recipe_file,'r') as f:
            bias_recipe:dict=json.load(f)
        self.penalty_recipes:List[PenaltyRecipe]=[]
        for k,v in bias_recipe.items():
            v['name']=k
            self.penalty_recipes.append(PenaltyRecipe.from_dict(v))
    
    def run_mpnn(self,record:DesignRecord)->List[DesignRecord]:
        adv=self.settings.adv
        n_mpnn_samples:int=adv.get("n_mpnn_samples",20)
        max_mpnn_sequences:int=adv.get("max_mpnn_sequences",2)
        temperature:float=adv.get("mpnn_sampling_temp",0.1)
        binder_chain:str=self.settings.target_settings.new_binder_chain
        full_target_chain=self.settings.target_settings.full_target_chain
        
        bias=self.cal_bias(record)
        _fix_po=np.where(bias[:,-1]==0)[0]
        fixed_positions = full_target_chain +','+ ','.join(
            f'{binder_chain}{i+1}' for i in _fix_po)
        
        length=len(record.sequence)
        pdb_file=record.pdb_files[self.pdb_to_take]

        self.mpnn_model.prep_inputs(pdb_file,
            f'{full_target_chain},{binder_chain}',fix_pos=fixed_positions)
        ori_score=self.mpnn_model.score()['score']
        record.update_metrics({
            f'{self.metrics_prefix}score':ori_score,
            f'{self.metrics_prefix}seqid':1.})
        self.mpnn_model._inputs['bias'][-length:]+=bias[:,:-1]
        mpnn_sequences = self.mpnn_model.sample(temperature=temperature, 
            num=n_mpnn_samples, batch=n_mpnn_samples)
        mpnn_df=self._dedup_mpnn_results(mpnn_sequences,length,record.id,record.sequence)
        dis_df=simple_diversity(mpnn_df)
        if len(dis_df)>max_mpnn_sequences:
            medoids, labels=cluster_and_get_medoids(dis_df.to_numpy(),max_mpnn_sequences)
            mpnn_df=mpnn_df.iloc[medoids].copy()
            mpnn_df.index=[f'{record.id}-{self.metrics_prefix.strip(NEST_SEP)}{i+1}' 
                for i in range(len(mpnn_df))]

        ret=[record]
        for i,s in mpnn_df.iterrows():
            r=DesignRecord(
                id=i,sequence=s['sequence'],pdb_files={self.pdb_to_take:pdb_file})
            r.update_metrics({f'{self.metrics_prefix}score':s['score'],
                f'{self.metrics_prefix}seqid':s['seqid']})
            ret.append(r)
        return ret

    def _dedup_mpnn_results(self,mpnn_sequences,
            length:int,design_id:str,ori_seq:str|None=None):
        _={
            mpnn_sequences['seq'][n][-length:]: {
                'seq': mpnn_sequences['seq'][n][-length:],
                'score': mpnn_sequences['score'][n],
                'seqid': mpnn_sequences['seqid'][n]
            } for n in range(len(mpnn_sequences['seq']))}
        if ori_seq in _:
            _.pop(ori_seq)
        mpnn_df=pd.DataFrame()
        mpnn_df['sequence']=[i['seq'] for i in _.values()]
        mpnn_df['score']=[i['score'] for i in _.values()]
        mpnn_df['seqid']=[i['seqid'] for i in _.values()]
        mpnn_df=mpnn_df.sort_values(by='score',ascending=False)
        mpnn_df.index=[f'{design_id}-{self.metrics_prefix.strip(NEST_SEP)}{i+1}' 
            for i in range(len(mpnn_df))]
        return mpnn_df
    
    def process_record(self,input:DesignRecord)->List[DesignRecord]:
        with self.record_time(input):
            ret=self.run_mpnn(input)
        return ret
    
    def process_batch(self,input:DesignBatch,
        metrics_prefix:str|None=None,pdb_to_take:str=None,
        )->DesignBatch:
        if metrics_prefix is not None:
            self.config_metrics_prefix(metrics_prefix)
        if pdb_to_take is not None:
            self.config_pdb_input_key(pdb_to_take)
        mpnn_suffix=self.metrics_prefix.strip(NEST_SEP)
        new_designs:List[DesignRecord]=[]
        for records_id,record in input.records.items():
            if mpnn_suffix not in records_id:
                sampled= f'{records_id}-{mpnn_suffix}1' in input.records
                if isinstance(input,DesignBatchSlice):
                    sampled = sampled or f'{records_id}-{mpnn_suffix}1' in input.parent.records
                if input.overwrite or not sampled:
                    new_designs.extend(self.process_record(record)[1:])
                    input.save_record(record.id) 
        for i in new_designs:
            input.add_record(i)
            input.save_record(i.id)
        return input
    
_default_penalties_values=dict(
    Sp=math.log(0.5),Hp=math.log(0.25),
    No=-float(1e6),Null=0.,
    Sr=math.log(2),Hr=math.log(4))

class PenaltyRecipe:
    def __init__(self,name:str,
        select_func_expr:str="lambda s: s['seq']=='C' and s['surf']",
        penalties_aas:Dict[str,str]={'Null':all_aa},
        penalties_values:Dict[str,float]=_default_penalties_values,
        select_func:Callable[[pd.Series],bool]|None=None
        ):
        '''
        select aa positions by `select_func`
        make them mutable by MPNN (or fix these position)
        apply MPNN bias by penalties_aas & penalties_values
        --- ---
        select_func_expr: something could be solved by `eval` into select_func
        penalties_aas: something like: 
            {'No':'C','Hp':'N','Sp':'DE'} : 
                for selected pos, C is not allowed, N is discouraged, DE is slightly discouraged. 
            {'fix':''}: 
                once "fix" is in `penalties_aas`, fix these positions from MPNN.
        penalties_values: something like: 
            {'Hp':math.log(0.25)}:
                aa denoted as 'Hp' will only have 25% of original probability to be sampled
        select_func: a *named function* that would override select_func_expr.
        '''
        self.name=name
        if select_func is not None:
            self.select_func:Callable[[pd.Series],bool]=select_func
            self.select_func_expr=''
        else:
            self.select_func:Callable[[pd.Series],bool]=eval(select_func_expr)
            self.select_func_expr=select_func_expr
        self.penalties_aas=penalties_aas
        self.penalties=penalties_values
    
    def add_bias(self,track_df:pd.DataFrame,bias:np.ndarray):
        '''
        track_df: pd.DataFrame(ana_track)
        '''
        sel=track_df.apply(self.select_func,axis=1).astype(bool)
        if sel.any():
            pos=track_df.loc[sel].index.to_list()
            if 'fix' not in self.penalties_aas:
                for k,aas in self.penalties_aas.items():
                    penalty=self.penalties.get(k,0.)
                    bias=self._add_bias(bias,pos,aas,penalty)
            else:
                bias=self._fix_pos(bias,pos)
        return bias
    
    def _add_bias(self,bias:np.ndarray,pos:List[int],aas:str,penalty:float):
        aa=[aa_order[a] for a in aas]
        bias[np.ix_(pos,aa)]+=penalty
        bias[pos,-1]=1
        return bias
    
    def _fix_pos(self,bias:np.ndarray,pos:List[int]):
        bias[pos,-1]=0
        return bias
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if not is_pickleable(state.get('select_func','')):
            state['select_func'] = None  
        return state

    def __setstate__(self, state:Dict[str,Any]):
        if state['select_func'] is None and state.get('select_func_expr','None'):
            state['select_func'] = eval(state['select_func_expr'])
        self.__dict__.update(state)

    @classmethod
    def from_dict(cls,d:dict):
        return cls(**d)
        

def is_pickleable(obj) -> bool:
    try:
        pkl.dumps(obj)
        return True
    except Exception:
        return False

recipe_noC_noPPI={
    'No-C-No-PPI':{
        "select_func_expr":"lambda s: not s['ppi']",
        "penalties_aas":{'No':'C'},
        }
    }

recipe_noC={
    'No-C':{
        "select_func_expr":"lambda s: True",
        "penalties_aas":{'No':'C'},
        }
    }

    
