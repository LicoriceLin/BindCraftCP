import tempfile
from subprocess import run
from .basescorer import BaseScorer,GlobalSettings,DesignRecord,DesignBatch,NEST_SEP
import os
from pathlib import Path
from typing import Dict,Tuple,List,Callable
from functools import partial
import numpy as np
single_mutsite_parse=Dict[str,List[Tuple[int,str,float]]]

AVAILABLE_PTM=[
    'Hydroxylysine',
    'Hydroxyproline',
    'S-palmitoyl_cysteine',
    'O-linked_glycosylation',
    'Methylarginine',
    'N6-acetyllysine',
    'Ubiquitination',
    'Phosphoserine_Phosphothreonine',
    'Methyllysine',
    'SUMOylation',
    'N-linked_glycosylation',
    'Phosphotyrosine',
    'Pyrrolidone_carboxylic_acid']

def parse_musite_single(ptm_file:str)->single_mutsite_parse:
    o={}
    k=''
    vs=[]
    tag=''
    for line in open(ptm_file,'r').readlines():
        if line.startswith('Position') or not line.strip():
            continue
        elif line.startswith('>'):
            if k!='':
                o[k]=vs
            k=line.strip()[1:]
            vs=[]
        else:
            v=line.strip().split()[-1]
            if not tag:
                tag=line.strip().split()[-2].split(':')[0]
            l_=line.strip().split()
            vs.append((int(l_[0]),l_[1],float(l_[2].split(':')[-1])))
    return o


def musite_on_fasta(fasta:str,model:str,
    env:str='musite',repo_dir:str="../MusiteDeep_web"):
    if '/' in env:
        flag='-p'
    else:
        flag='-n'
    with tempfile.TemporaryDirectory() as tdir:
        run(['conda','run',flag,env,
            'python',
            'predict_multi_batch.py',
            '-input',Path(fasta).absolute(),
            '-output',f'{tdir}/{model}',
            '-model-prefix',f'models/{model}',],
            cwd=f'{repo_dir}/MusiteDeep',
            stdout=None,stderr=None)
        o=parse_musite_single(f'{tdir}/{model}_results.txt')
    return o


def run_musite(batch:DesignBatch,model:str,
    env:str='musite',repo_dir:str="../MusiteDeep_web",
    metrics_prefix:str=''):
    if '/' in env:
        flag='-p'
    else:
        flag='-n'
    
    with tempfile.TemporaryDirectory() as tdir:
        batch.to_fasta(f'{tdir}/in.fa')
        run(['conda','run',flag,env,
            'python',
            'predict_multi_batch.py',
            '-input',f'{tdir}/in.fa',
            '-output',f'{tdir}/{model}',
            '-model-prefix',f'models/{model}',],
            cwd=f'{repo_dir}/MusiteDeep',
            stdout=None,stderr=None)
        o=parse_musite_single(f'{tdir}/{model}_results.txt')
    
    for record_id,record, in batch.records.items():
        entry=o.get(record_id,[])
        track=np.zeros((len(record.sequence),)).astype(float)
        for i in entry:
            track[i[0]-1]+=i[2]
        record.ana_tracks[f'{metrics_prefix}{model}']=[round(i,2) for i in track]
    
    return batch


class AnnotPTM(BaseScorer):
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings, score_func=run_musite)

    def _init_params(self):
        self.params=dict(
        env=self.settings.adv.get('musite_env','musite'),
        repo_dir=self.settings.adv.get('musite_repo',"../MusiteDeep_web"),
        metrics_prefix=self.metrics_prefix)

    @property
    def name(self)->str:
        return 'PTM-annot'
    
    @property
    def track_to_add(self):
        return tuple([f'{self.metrics_prefix}{self.current_ptm}'])

    @property
    def current_ptm(self):
        return self.params['model']

    def process_record(self, input:DesignRecord):
        '''
        Not implemented. Call `process_batch` instead.
        '''
        raise NotImplementedError(f'Run {self.name} in Batches!')
    
    @property
    def score_func(self)->Callable[[DesignBatch],DesignBatch]:
        return partial(self._score_func,**self.params)
    
    def process_batch(self, input:DesignBatch, **kwargs):
        self.config_params(**kwargs)
        sub_batch=input.filter(lambda i: not self.check_processed(i))
        self.score_func(sub_batch)
        sub_batch.save_records()


    