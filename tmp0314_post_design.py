import json
from typing import Literal
from ast import literal_eval
import glob
import os
# from pathlib import Path
from pathlib import PosixPath as Path
from tqdm import tqdm
from subprocess import run

from typing import Iterable,Union,Callable,Generator,List,Dict,Tuple
from tempfile import TemporaryDirectory
import pickle as pkl
import json
from ast import literal_eval
from collections.abc import Iterable as collections_Iterable
from itertools import tee


import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statannotations.Annotator import Annotator

import Bio.PDB as BP
from Bio.PDB import PDBParser
from Bio.PDB.Entity import Entity
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.SASA import ShrakeRupley
from Bio.Data import PDBData

from pymol import cmd


from colabdesign.mpnn import mk_mpnn_model
from colabdesign import clear_mem

from functions import hotspot_residues,mk_afdesign_model,add_cyclic_offset
from tmp0304_metrics import (
    run_musite,parse_musite_dir,run_esm_if,gen_ana_tracks,pdb2seq,propka_single,ptm_propka,musite_parse_recipe
)
from tmp0314_mda_utils import cal_ppi_interacts,cal_rog

# %% metrics IO
def id2pdbfile(design_id:str,outdir:str,mode:Literal['relax','mpnn','design','single','design_relax']='relax'):
    if mode not in ['design','design_relax']:
        f=glob.glob(f'{outdir}/Accepted/{design_id}_model*.pdb')[0]
    if mode == 'relax':
        return f    
    elif mode == 'mpnn':
        return f.replace('/Accepted','/MPNN')
    elif mode == 'design':
        return f'{outdir}/Trajectory/{design_id}.pdb'
    elif mode == 'single':
        return f.replace('/Accepted','/MPNN/Binder')
    elif mode == 'design_relax':
        return f'{outdir}/Trajectory/Relaxed/{design_id}.pdb'
    else:
        raise NotImplementedError

outdir_from_metrics=lambda metrics:str(Path(metrics.iloc[0]['pdbfile']).parents[1])

def _avg_InterfaceAAs(aa_dicts:List[Dict[str,int|float]]):
    o=aa_dicts[0].copy()
    l=len(aa_dicts)
    for aa_dict in aa_dicts[1:]:
        for k,v in aa_dict.items():
            o[k]+=v
    o={k:v/l for k,v in o.items()}
    return o

_meta_cols=[
    'Design','Length','Target_Hotspot','Sequence','InterfaceResidues']
_metrics_col=['pLDDT','pTM','i_pTM','pAE','i_pAE','i_pLDDT','ss_pLDDT',
                'Unrelaxed_Clashes','Relaxed_Clashes','ShapeComplementarity','PackStat',
                'Binder_Energy_Score','dG','dSASA','dG/dSASA','Interface_SASA_%',
                'Interface_Hydrophobicity','Surface_Hydrophobicity',
                'n_InterfaceResidues','n_InterfaceHbonds','n_InterfaceUnsatHbonds',
                'Interface_Helix%','Interface_BetaSheet%','Interface_Loop%',
                'Binder_Helix%','Binder_BetaSheet%','Binder_Loop%',
                'Binder_pLDDT','Binder_pTM','Binder_pAE']
_aa_col='InterfaceAAs'

def read_bc_metrics(outdir:str,first_two_only:bool=False):
    '''
    read `final_design_stats`
    fixed rules: `Design` should always be the index col. 
    '''
    final_score_df=pd.read_csv(f'{outdir}/final_design_stats.csv'
        ).drop_duplicates(subset=['Design'], keep='last')
    if not first_two_only:
        used_cols=_meta_cols+[f'Average_{i}' for i in _metrics_col+[_aa_col]]
        bc_metrics=final_score_df[used_cols].copy()
        bc_metrics.columns=[i.replace('Average_','') for i in bc_metrics.columns]
        bc_metrics[f'{_aa_col}']=bc_metrics[f'{_aa_col}'].apply(lambda x:literal_eval(x))
    else:
        aa_col_=[f'{i}_{_aa_col}' for i in [1,2]]
        arr_1=final_score_df[[f'1_{i}' for i in _metrics_col]].to_numpy()
        arr_2=final_score_df[[f'2_{i}' for i in _metrics_col]].to_numpy()
        arr_avg=(arr_1+arr_2)/2
        df_avg=pd.DataFrame(arr_avg,columns=[f'{i}' for i in _metrics_col])
        
        df_aas=final_score_df[aa_col_]
        df_avg[f'{_aa_col}']=df_aas.apply(lambda s:_avg_InterfaceAAs([literal_eval(s[i]) 
                        for i in aa_col_]),axis=1)
        
        bc_metrics=pd.concat([final_score_df[_meta_cols],df_avg],axis=1)
        
    bc_metrics=bc_metrics.set_index('Design')
    bc_metrics['pdbfile']=[id2pdbfile(i,outdir) for i in bc_metrics.index]
    return bc_metrics

def read_design_metrics(outdir:str):
    design_metrics=pd.read_csv(f'{outdir}/trajectory_stats.csv'
        ).drop_duplicates(subset=['Design'], keep='last').set_index('Design')
    design_metrics['InterfaceAAs']=design_metrics['InterfaceAAs'].apply(literal_eval)
    design_metrics['InterfaceResidues'].fillna('',inplace=True)
    design_metrics['pdbfile']=[id2pdbfile(i,outdir=outdir,mode='design') for i in design_metrics.index]
    return design_metrics

def dump_metrics(df:pd.DataFrame,file:str):
    '''
    syntactic sugar
    .to_csv(...,index_label='Design')
    '''
    df.to_csv(file,index_label='Design')

# %% metrics filters
def _load_default_filters(bindcraft_benchmark:bool=False):
    '''
    Default default_filters,  
    remove "Average_" for compatibility with `design_stat`  
    adjust several thresholds (pLDDT, ShapeComplementarity, Surface_Hydrophobicity, Binder_Loop%)
    '''
    filter:dict=json.load(open('settings_filters/default_filters.json','r'))
    design_stat_col=['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot',
       'Sequence', 'InterfaceResidues', 'pLDDT', 'pTM', 'i_pTM', 'pAE',
       'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes',
       'Binder_Energy_Score', 'Surface_Hydrophobicity', 'ShapeComplementarity',
       'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%',
       'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds',
       'InterfaceHbondsPercentage', 'n_InterfaceUnsatHbonds',
       'InterfaceUnsatHbondsPercentage', 'Interface_Helix%',
       'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
       'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Target_RMSD',
       'TrajectoryTime', 'Notes', 'TargetSettings', 'Filters',
       'AdvancedSettings']
    
    filter={k.replace("Average_",""):v for k,v in filter.items() if "Average_" in k and (v.get("threshold",None) is not None or k =='Average_InterfaceAAs') }
    filter={i:filter[i] for i in filter if i in design_stat_col}
    if bindcraft_benchmark:
        return filter
    else:
        filter.pop('InterfaceAAs')
        # filter['InterfaceAAs']['C']={'threshold': 1, 'higher': False}
        filter['pLDDT']={'threshold': 0.75, 'higher': True}
        filter['ShapeComplementarity']={'threshold': 0.55, 'higher': True}
        filter['Surface_Hydrophobicity']={'threshold': 0.55, 'higher': False}
        filter['Binder_Loop%']={'threshold': 35, 'higher': False}
    return filter
    # filter['n_InterfaceUnsatHbonds']={'threshold': 40, 'higher': False}

_bc_benchmark_filters=_load_default_filters(True)
_default_filters=_load_default_filters()
_simpliest_filter={'pLDDT':{'threshold': 0.70, 'higher': True}}

def _check_filters(entry:pd.Series, filters:dict=_default_filters):
    # check mpnn_data against labels
    # mpnn_dict = {label: value for label, value in zip(design_labels, mpnn_data)}

    unmet_conditions = []

    # check filters against thresholds
    for label, conditions in filters.items():
        # special conditions for interface amino acid counts
        if label == 'InterfaceAAs':
            for aa, aa_conditions in conditions.items():
                if entry.get(label) is None:
                    continue
                value = entry.get(label).get(aa)
                if value is None or aa_conditions["threshold"] is None:
                    continue
                if aa_conditions["higher"]:
                    if value < aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
                else:
                    if value > aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
        elif label in ['InterfaceResidues','Target_Hotspot']:
            # conditions should be a string like "B1,B7,..."
            expected_res:List[str]= conditions.split(",")
            res:List[str]= entry.get(label).split(",")
            for r in expected_res:
                if res not in expected_res:
                    unmet_conditions.append(label)
                    break
               
        else:
            # if no threshold, then skip
            value = entry.get(label)
            if value is None or conditions["threshold"] is None:
                continue
            if conditions["higher"]:
                if value < conditions["threshold"]:
                    unmet_conditions.append(label)
            else:
                if value > conditions["threshold"]:
                    unmet_conditions.append(label)

    # if all filters are passed then return True
    if len(unmet_conditions) == 0:
        return "All_Good"
    # if some filters were unmet, print them out
    else:
        return ';'.join(unmet_conditions)

def check_filters(stat:pd.DataFrame,filters:dict=_default_filters):
    o=[]
    for _,s in stat.iterrows():
        o.append(_check_filters(s,filters))
    stat['filt']=o
    return stat

def show_filter(stat:pd.DataFrame,filters:dict=_default_filters):
    check_filters(stat,filters)
    o=stat['filt'].to_list()

    failed_labels=[]
    for i in o:
        failed_labels.extend(i.split(';'))
    fig,ax=plt.subplots(1,1)
    s=pd.Series(failed_labels).value_counts().sort_values()/len(stat)
    # s=s[[i for i in s.index if s!='Pass']+['Pass']]
    ax=sns.barplot(s,ax=ax,order=[i for i in s.index if i!='All_Good']+['All_Good'])
    ax.set_ylim([0,1])
    ax.bar_label(ax.containers[0], fontsize=10,fmt=lambda x:f'{x*100:.1f}')
    ax.set_ylabel('Failed Rate')
    ax.set_xlabel('Failed Criteria')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_verticalalignment('top')
        tick.set_horizontalalignment('right')
        if tick.get_text()=='All_Good':
            tick.set_color('red')
            # tick.set_text('All Good')

    return fig,ax

# %% post analysis
def match_pattern(seq:str)->np.ndarray:
    arr = np.array(list(seq))
    n = len(arr)
    result = np.zeros(n, dtype=int)
    if n >= 4:
        windows_4 = sliding_window_view(arr, 4)
        match_4A = np.all(windows_4 == 'A', axis=1)
        for i in range(4):
            result[i:n-3+i] |= match_4A
    if n >= 3:
        for code in 'KQER':
            windows_3 = sliding_window_view(arr, 3)
            match_3KQ = np.all(windows_3 == code, axis=1)
            for i in range(3):
                result[i:n-2+i] |= match_3KQ  
    return result

def _default_ptm_feat_recipe(ptm_track:Dict[str,str|np.ndarray]):
    o={}
    seq_array=np.array(list(ptm_track['seq']))
    o['Surf-C'] = ((seq_array=='C') & (ptm_track['surf'])).sum()
    o['Surf-Gly'] = ( (ptm_track['Gly']>0.5) & (ptm_track['surf'])).sum()
    o['Surf-MeR'] = ( (ptm_track['MeR']>0.5) & (ptm_track['surf'])).sum()
    return o

class Metrics:
    def __init__(self,
        outdir:str,
        mode:Literal['design','mpnn','slice']='design',
        # ptm:bool=True,
        # esm:bool=True,
        # pi:bool=True,
        # ptm_pi:bool=True,
        # misc:bool=True,
        # mut_recipe:Dict[str,Tuple[str,str]]=musite_parse_recipe,
        ):
        
        self.outdir=outdir
        self.mode=mode
        if mode=='design':
            self.metrics:pd.DataFrame=read_design_metrics(outdir)
            _s=self._subdir='Trajectory'
            os.makedirs(f'{outdir}/{_s}/Post',exist_ok=True)
            self.ana_paths=dict(
                    fasta=f'{outdir}/{_s}/Post/design.fa',
                    musite_dir=f'{outdir}/{_s}/Ptm',
                    esm_if_file=f'{outdir}/{_s}/Post/esm_if.pkl',
                    ana_tracks_file=f'{outdir}/{_s}/Post/init_ana_tracks.pkl',
                    pis_file=f'{outdir}/{_s}/Post/pis.pkl',
                    modi_pis_file=f'{outdir}/{_s}/Post/modi_pis.pkl',
                    mda_ana_file=f'{outdir}/{_s}/Post/mda.pkl',
                    Metrics=f'{outdir}/{_s}/Post/Metrics.pkl',
                )
        elif mode=='mpnn':
            self.metrics:pd.DataFrame=read_bc_metrics(outdir)
            os.makedirs(f'{outdir}/MPNN/Post',exist_ok=True)
            _s=self._subdir='MPNN'
            self.ana_paths=dict(
                    fasta=f'{outdir}/{_s}/Post/design.fa',
                    musite_dir=f'{outdir}/{_s}/Ptm',
                    esm_if_file=f'{outdir}/{_s}/Post/esm_if.pkl',
                    ana_tracks_file=f'{outdir}/{_s}/Post/init_ana_tracks.pkl',
                    pis_file=f'{outdir}/{_s}/Post/pis.pkl',
                    modi_pis_file=f'{outdir}/{_s}/Post/modi_pis.pkl',
                    mda_ana_file=f'{outdir}/{_s}/Post/mda.pkl',
                    Metrics=f'{outdir}/{_s}/Post/Metrics.pkl',
                )
        elif mode=='slice':
            # self.outdir=f'slice:{self.outdir}'
            pass
        else:
            raise NotImplementedError
        
    def post_process(self,prcess_recipe:Literal['miniprot_full','minimal']='miniprot_full'):
        if prcess_recipe=='miniprotein_full':
            self.cal_ptm()
            self.gen_ana_tracks()
            self.cal_ptm_feat()
            self.cal_pi()
            self.cal_ptm_pi()
            self.cal_esm()
            self.cal_lcr()
            self.cal_interact()
            self.cal_rog()
            self.save()
        elif prcess_recipe=='minimal':
            self.gen_ana_tracks()
            self.cal_interact()
            self.cal_rog()
            self.save()
        
    def gen_ana_tracks(self,sasa_threshold:float=0.4,force_regen:bool=False):
        print('generate Residue Tracks')
        ana_tracks_file=self.ana_paths['ana_tracks_file']
        self.sasa_threshold=sasa_threshold
        ptm=getattr(self,'ptms',None)
        if force_regen or not os.path.exists(ana_tracks_file):
            self.ana_tracks=gen_ana_tracks(self.metrics,ptm,self.sasa_threshold)
            # if ptm is None:
            #     self._ptm_in_track=False
            # else:
            #     self._ptm_in_track=True
        else:
            self.ana_tracks:Dict[str,Dict[str,str|np.ndarray]]=pkl.load(open(ana_tracks_file,'rb'))
            # t_=next(iter(self.ana_tracks.values()))
            # if 'Gly' in t_:
            #     self._ptm_in_track=True
            # else:
            #     self._ptm_in_track=False
            
            new_entries=[i for i in self.metrics.index if i not in self.ana_tracks]
            if len(new_entries)>0:
                if (ptm is None)^(self._ptm_in_track):
                    self.cal_ptm()
                    self.ana_tracks=gen_ana_tracks(self.metrics,ptm,self.sasa_threshold)
                    # self._ptm_in_track=True
                else:
                    self.ana_tracks.update(
                        gen_ana_tracks(self.metrics.loc[new_entries],getattr(self,'ptms',None))
                    )
        with open(ana_tracks_file,'wb') as f:
            pkl.dump(self.ana_tracks,f)

    def cal_ptm_feat(self,
        ptm_feat_recipe:Callable[[Dict[str,str|np.ndarray]],Dict[str,float|int]]=_default_ptm_feat_recipe):
        print('cal PTM features')
        assert self._ptm_in_track
        ptm_feats_df=pd.DataFrame({k:ptm_feat_recipe(v) for k,v in self.ana_tracks.items()}).T
        
        if ptm_feats_df.columns[0] in self.metrics.columns:
            self.metrics.drop(ptm_feats_df.columns,axis=1,inplace=True)
        self.metrics=pd.merge(left=self.metrics,right=ptm_feats_df,left_index=True,right_index=True)

    def cal_ptm(self):
        print('run MusiteDeep')
        fasta=self.ana_paths['fasta']
        musite_dir=self.ana_paths['musite_dir']
        with open(fasta,'w') as f:
            for pdb in self._pdbs:
                f.write(f'>{Path(pdb).stem}\n')
                f.write(f'{pdb2seq(pdb)[0]}\n')

        musite_ckpt=f'{musite_dir}/MUSITE_DONE'
        if not os.path.exists(musite_ckpt):
            run_musite(fasta,musite_dir)
        else:
            done_id_=[i.strip() for i in open(musite_ckpt,'r').readlines()]
            done_id=set([i for i in done_id_ if len(i)>0])
            if len(set(self.metrics.index).difference(done_id))>0:
                run_musite(fasta,musite_dir)

        with open(musite_ckpt,'w') as f:
            f.write('\n'.join(self.metrics.index))

        self.ptms=parse_musite_dir(musite_dir)
    
    def cal_pi(self):
        print('calculate PI')
        pis_file=self.ana_paths['pis_file']
        if os.path.exists(pis_file):
            pis:Dict[str,Dict[str,float]]=pkl.load(open(pis_file,'rb'))
        else:
            pis={}
        for k,v in tqdm(self.metrics['pdbfile'].items()):
            if k not in pis:
                pis[k]=propka_single(v)
        self.metrics['pi-fold']=[pis[i]['pi-fold'] for i in self.metrics.index]
        pkl.dump(pis,open(pis_file,'wb'))


        self.metrics['pi-fold']=[pis[i]['pi-fold'] for i in self.metrics.index]
        self.pis=pis
        pkl.dump(pis,open(pis_file,'wb'))

    def cal_ptm_pi(self,ptm_threshold:float=0.5,sasa_threshold:float=0.4,
        mut_recipe:Dict[str,Tuple[str,str]]=musite_parse_recipe
        ):
        print('calculate PTM-modified PI')
        if getattr(self,'ptms',None) is None:
            'ptm needed. running `cal_ptm`'
            self.cal_ptm()
        ptms=self.ptms
        
        _=getattr(self,'sasa_threshold',None)
        if _ is not None and _!=sasa_threshold:
            print(f'sasa_threshold predefined as {_:.2f}. Given parameter is overwrited.')
            sasa_threshold=_
        self.sasa_threshold=sasa_threshold

        modi_pis_file=self.ana_paths['modi_pis_file']
        if os.path.exists(modi_pis_file):
            modi_pis:Dict[str,Dict[str,float]]=pkl.load(open(modi_pis_file,'rb'))
        else:
            modi_pis={}

        for k,v in tqdm(self.metrics['pdbfile'].items()):
            if k not in modi_pis:
                modi_pis[k]=ptm_propka(v,ptms,ptm_threshold=ptm_threshold,
                    sasa_threshold=sasa_threshold,mut_recipe=mut_recipe)
        
        self.metrics['modi-pi-fold']=[modi_pis[i]['pi-fold'] for i in self.metrics.index]
        self.modi_pis=modi_pis
        pkl.dump(modi_pis,open(modi_pis_file,'wb'))

    def cal_esm(self):
        print('run ESM-IF')
        esm_if_file=self.ana_paths['esm_if_file']
        if not os.path.exists(esm_if_file):
            esm_if_o=run_esm_if(self._pdbs,'B')
            esm_if_o={Path(k).stem:v for k,v in esm_if_o.items()}
            with open(esm_if_file,'wb') as f:
                pkl.dump(esm_if_o,f)
        else:
            esm_if_o:Dict[str,pd.DataFrame]=pkl.load(open(esm_if_file,'rb'))
            new_entry=[i for i in self.metrics.index if i not in esm_if_o]
            if len(new_entry)>0:
                new_pdbs=self.metrics.loc[new_entry]['pdbfile']
                new_esm_if_o=run_esm_if(new_pdbs,'B')
                esm_if_o.update({Path(k).stem:v for k,v in new_esm_if_o.items()})
                with open(esm_if_file,'wb') as f:
                    pkl.dump(esm_if_o,f)
        
        self.metrics['stab']=[esm_if_o[i]['score'].iloc[-1] for i in self.metrics.index]
        for k,v in self.ana_tracks.items():
            v['esm_if']=esm_if_o[k]['score'][:-2].to_numpy()
        self.esm_if_o=esm_if_o
    
    def cal_lcr(self):
         print('match Low Complexity Region')
         for k,v in self.ana_tracks.items():
            v['LCR']=match_pattern(v['seq'])
            self.metrics['lcr']=int(v['LCR'].sum())
    
    def cal_interact(self):
        print('annot Hydro-Bonds/Salt-Bridges')
        mda_ana_file=self.ana_paths['mda_ana_file']
        if not os.path.exists(mda_ana_file):
            inters={}
        else:
            inters=pkl.load(open(mda_ana_file,'rb'))

        for design_id,s in tqdm(self.metrics.iterrows()):            
            if design_id in inters:
                (ppi_interacts,b_inter_res)=inters[design_id]
            else:
                pdbfile=s['pdbfile'].replace('Trajectory/','Trajectory/Relaxed/')
                inter_res=s['InterfaceResidues']
                (ppi_interacts,b_inter_res)=cal_ppi_interacts(pdbfile,inter_res)
                inters[design_id]=(ppi_interacts,b_inter_res)
            
            track=self.ana_tracks[design_id]
            interacts=np.zeros(len(track['seq']))
            for i, (s,p) in enumerate(zip(track['seq'],track['ppi'])):
                if p and s in 'STYNQDERKH':
                    i_=i+1
                    if i_ in b_inter_res['all']:
                        interacts[i]=1
                    else:
                        interacts[i]=-1
            
            track['interacts']=interacts
            
        pkl.dump(inters,open(mda_ana_file,'wb'))

    def cal_rog(self):
        print('cal RoG')
        self.metrics['rog']=self.metrics['pdbfile'].apply(cal_rog)
    
    def save(self,ckpt:str|None=None):
        print('save CkPt')
        if ckpt is None:
            ckpt=self.ana_paths['Metrics']
        with open(ckpt, 'wb') as f:
            pkl.dump(self.__dict__, f)

    @classmethod
    def load(cls, outdir:str|None=None,mode:Literal['design','mpnn','slice']='design',ckpt:str|None=None):
        if outdir is not None:
            if ckpt is not None:
                print('`ckpt` are overloaded by `outdir`')
            m=f'{outdir}/Trajectory/Post/Metrics.pkl'
            ret=cls(outdir,mode)
            with open(m, 'rb') as f:
                ret.__dict__.update(pkl.load(f))
            return ret
        else:
            assert ckpt is None
            with open(ckpt, 'rb') as f:
                d:dict=pkl.load(f)
            ret=cls(d['outdir'],'slice')
            ret.__dict__.update(d)
            return ret
    
    @property
    def _pdbs(self):
        '''
        direct af2 models
        '''
        if getattr(self,'_pdbs_',None) is None:
            if self._subdir=='Trajectory':
                self._pdbs_=self.metrics['pdbfile']
            elif self._subdir=='MPNN':
                self._pdbs_=pd.Series({i:id2pdbfile(i,outdir=self.outdir,mode='mpnn') 
                    for i in self.metrics.index})
        return self._pdbs_

    @property
    def _relaxed_pdbs(self):
        if getattr(self,'_relaxed_pdbs_',None) is None:
            if self._subdir=='MPNN':
                self._relaxed_pdbs_=self.metrics['pdbfile']
            elif self._subdir=='Trajectory':
                self._relaxed_pdbs_=pd.Series({i:id2pdbfile(i,outdir=self.outdir,mode='design_relax') 
                    for i in self.metrics.index})
        return self._relaxed_pdbs_
    
    @property
    def _ptm_in_track(self):
        if hasattr(self,'ana_tracks'):
            if 'Gly' in next(iter(self.ana_tracks.values())):
                return True
            else:
                return False
        else:
            return False

    def keys(self):
        if hasattr(self,'metrics'):
            return self.metrics.index.to_list()
        else:
            return []
        
    def __getitem__(self, key:str|Iterable[str]|int|Iterable[int]):
        if isinstance(key,str):
            key=[key]
        elif isinstance(key,int):
            key=[self.keys()[key]]
        else:
            key = list(key)
            if  isinstance(key[0],int):
                key=[self.keys()[i] for i in key]
            
        ret=Metrics(outdir=self.outdir,mode='slice')
        d=self.__dict__.copy()
        d['mode']='slice'
        if 'ana_paths' in d:
            d['ana_paths']={k: v+'.slice' if v.endswith('.slice') else v 
                for k,v in self.ana_paths.items()}
        
        if 'metrics' in d:
            d['metrics']=self.metrics.loc[key].copy()
        for i in ['ana_tracks','pis','modi_pis','esm_if_o']:
            if i in d:
                d[i]={k:v for k,v in getattr(self,i).items() if k in key}
        for i in ['_relaxed_pdbs_','_pdbs_']:
            if i in d:
                d.pop(i)
        if 'ptms' in d:
            for k,v in self.ptms.items():
                v_={k_:v_ for k_,v_ in v[0].items() if k_ in key}
                d['ptms'][k]=(v_,v[1])
            
        ret.__dict__.update(d)
        return ret
        
    def __len__(self):
        return len(self.metrics)
    
class MetricsVisualizer:
    def __init__(self,
        metrics:Metrics,
        ana_dir_name:str='Post',
        filters:Dict[str,Dict[str,float|int|bool]]=_simpliest_filter):
        self.Metrics=metrics
        self._metrics=self.Metrics.metrics.copy()
        self.filters=filters
        subdir=metrics._subdir
        self.anadir=self.Metrics.outdir+f'/{subdir}/{ana_dir_name}'
        os.makedirs(self.anadir,exist_ok=True)

    def plot_filter(self,stem:str='filter'):
        fig,ax=show_filter(stat=self._metrics,filters=self.filters)
        fig.savefig(f'{self.anadir}/{stem}.svg')
        fig.savefig(f'{self.anadir}/{stem}.png')
        return fig,ax

    def plot_dist(self,stem:str='metric_dist',
            metrics_cols:List[str]=[],exclude_cols:List[str]=[]):
        filter_dict=self.filters
        if len(metrics_cols)==0:
            metrics_cols=[i for i in self._metrics if i not in _meta_cols+[_aa_col]+exclude_cols]
        with PdfPages(f'{self.anadir}/{stem}.pdf') as pdf:
            for i in metrics_cols:
                fig,ax=plt.subplots(1,1,figsize=(6,6))
                sns.histplot(self._metrics,x=i,stat='probability',ax=ax, kde=True)
                ax.set_title(i)
                if i in filter_dict:
                    c='tab:red' if filter_dict[i]['higher'] else 'tab:green'
                    thresh=filter_dict[i]['threshold']
                    xmin,xmax=ax.get_xlim()
                    ymin,ymax=ax.get_ylim()
                    arrow_y=(ymin+ymax)/2
                    dx=(xmax-xmin)*0.05 if filter_dict[i]['higher'] else -(xmax-xmin)*0.05
                    ax.vlines(thresh,xmin,xmax,colors=c,linestyles='--')
                    ax.arrow(arrow_y,thresh,0,dx,color=c,head_width=0.05,head_length=abs(dx*0.5),length_includes_head=True)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    def plot_boxplot(self,
        tag_col:str='tag',hue_col:str|None='',stem:str='metric_boxplot',
        metrics_cols:List[str]=[],exclude_cols:List[str]=[]):
        raise NotImplementedError
    
    def plot_tracks(self):
        raise NotImplementedError
    
class HallucinationDesigner:
    def __init__(self):
        pass

class ReScorer:
    def __init__(self,
        metrics:Metrics,
        cyclic:bool=False,
        model_kwargs:Dict[str,bool]=dict(
            use_multimer=False, 
            use_initial_guess=True, 
            use_initial_atom_pos=True)):
        self.Metrics=metrics
        self.cyclic=cyclic
        self.model_kwargs=model_kwargs
    
    def rescore_seqonly(self,
        rescore_target:str,rescore_chain:str,rescore_dir:str='rescore',
        seed:int=42,pred_models=[0,1]):
        '''
        '''
        self.rescore_target=rescore_target
        self.rescore_chain=rescore_chain
        self.cyclic=self.cyclic
        self.seed=seed
        self.pred_models=pred_models
        r_dir=self.Metrics.outdir+f'/{self.Metrics._subdir}/'+rescore_dir
        self.r_dir=r_dir
        os.makedirs(r_dir,exist_ok=True)

        logs={}
        prev_l=-1
        model=self.complex_prediction_model

        m_=self.Metrics['Sequence'].sort_values(key=lambda x:x.str.len())
        for design_id,seq in tqdm(m_.items()):
            l=len(seq)
            if l!= prev_l:
                model.prep_inputs(
                        pdb_filename=rescore_target,#
                        chain=rescore_chain, 
                        binder_len=l, 
                        rm_target_seq=False,
                        rm_target_sc=False,
                        seed=seed)
                if self.cyclic:
                    add_cyclic_offset(model, offset_type=2)
                prev_l=l
            aux:dict=model.predict(seq=seq,models=pred_models,
                num_models=len(pred_models),num_recycles=3,return_aux=True)['log']
            for i in ['hard','soft','temp','seqid','recycles','models']:
                aux.pop(i,None)
            model_pdb_path=f'{r_dir}/{design_id}.pdb'
            model.save_current_pdb(model_pdb_path)
            binder_contacts = hotspot_residues(model_pdb_path)
            aux['n_contact'] = len(binder_contacts.items())
            aux['contacts']=binder_contacts
            aux['r_pdb']=model_pdb_path
            logs[design_id]=aux
        self.rescore_log=logs
        self.rescore_df=pd.DataFrame(self.rescore_log).T
        with open(r_dir+'/log.pkl','wb') as f:
            pkl.dump(logs,f)
        self.rescore_df.to_csv(r_dir+'/log.pkl',index_label='Design')
        return self.rescore_log
    
    def rescore_grafttemp(self,rescore_target:str,rescore_chain:str,
        graft_dir:str='graft',rescore_dir:str='rescore',
        seed:int=42,pred_models=[0,1]):
        '''
        '''
        assert 'B' not in rescore_chain

        g_dir=self.Metrics.outdir+f'/{self.Metrics._subdir}/'+graft_dir
        self.g_dir=g_dir
        r_dir=self.Metrics.outdir+f'/{self.Metrics._subdir}/'+rescore_dir
        self.r_dir=r_dir
        os.makedirs(g_dir,exist_ok=True)
        os.makedirs(r_dir,exist_ok=True)
        
        model=self.complex_prediction_model
        logs={}
        for design_id,pdb_ in tqdm(self.Metrics._pdbs.items()):
            g_pdb=g_dir+f'/{design_id}.pdb'
            _graft_binder(pdb_,rescore_target,rescore_chain,g_pdb)
            # grafted_pdbs[design_id]=g_pdb
            
            s=self.Metrics.metrics.loc[design_id]
            seq=s['Sequence']
            model.prep_inputs(
                pdb_filename=g_pdb, 
                chain=rescore_chain, binder_chain='B', binder_len=len(seq), 
                use_binder_template=True, rm_target_seq=False,
                rm_target_sc=False, rm_template_ic=True,seed=seed)
            if self.cyclic:
                add_cyclic_offset(model, offset_type=2)
            aux:Dict[str,float]=model.predict(seq=seq,models=pred_models,
                num_models=len(pred_models),num_recycles=3,return_aux=True)['log']
            for i in ['hard','soft','temp','seqid','recycles','models']:
                aux.pop(i,None)
            model_pdb_path=f'{r_dir}/{design_id}.pdb'
            if os.path.exists(model_pdb_path):
                os.remove(model_pdb_path)
            model.save_current_pdb(model_pdb_path)
            binder_contacts = hotspot_residues(model_pdb_path)
            aux['n_contact'] = len(binder_contacts.items())
            aux['contacts']=binder_contacts
            aux['g_pdb']=g_pdb
            aux['r_pdb']=model_pdb_path
            logs[design_id]=aux
        
        self.rescore_log=logs
        with open(r_dir+'/log.pkl','wb') as f:
            pkl.dump(logs,f)
        self.rescore_df.to_csv(r_dir+'/log.pkl',index_label='Design')
        return self.rescore_log

    def merge_rescore_metrics(self,use_col=['rmsd','plddt','n_contact','i_pae']):
        m_df:pd.DataFrame=self.rescore_df[use_col]
        m_df.columns=[f'r:{i}' for i in m_df]
        for c in m_df.columns:
            if c in self.Metrics.metrics.columns:
                self.Metrics.metrics.drop(c,axis=1,inplace=True)
        self.Metrics.metrics=pd.merge(left=self.Metrics.metrics,right=m_df,left_index=True,right_index=True)
        return self.Metrics.metrics[m_df.columns]

    def reload_rescore_log(self,rescore_dir:str='rescore'):
        r_dir=self.Metrics.outdir+f'/{self.Metrics._subdir}/'+rescore_dir
        self.r_dir=r_dir
                
        with open(r_dir+'/log.pkl','wb') as f:
            logs=pkl.load(f)
        self.rescore_log=logs
        self.rescore_df=pd.DataFrame(self.rescore_log).T
        return self.rescore_log
    
    @property
    def complex_prediction_model(self):
        if getattr(self,'_complex_prediction_model',None) is None:
            self._complex_prediction_model = mk_afdesign_model(
            protocol="binder", 
            num_recycles=3,
            data_dir='.',
            **self.model_kwargs)
        return self._complex_prediction_model
    
    @property
    def rescore_cols(self):
        return list(next(iter(self.rescore_log.values())).keys())
        
def rescore_design(metrics:pd.DataFrame,rescore_target:str,chain:str,cyclic:bool=False,seed:int=42,pred_models=[0,1]):
    outdir=outdir_from_metrics(metrics)
    os.makedirs(f'{outdir}/Trajectory/rescore',exist_ok=True)
    complex_prediction_model = mk_afdesign_model(
            protocol="binder", 
            num_recycles=3,
            data_dir='.',
            use_multimer=False)
    plddts,binder_contacts_n={},{}
    prev_l=-1
    for design_id,seq in tqdm(metrics['Sequence'].sort_values(key=lambda x:x.str.len()).items()):
        l=len(seq)
        if l!= prev_l:
            complex_prediction_model.prep_inputs(
                    pdb_filename=rescore_target,#
                    chain=chain, 
                    binder_len=l, 
                    rm_target_seq=False,
                    rm_target_sc=False,
                    seed=seed)
            if cyclic:
                add_cyclic_offset(complex_prediction_model, offset_type=2)
            prev_l=l
        aux=complex_prediction_model.predict(seq=seq,models=pred_models,
                num_models=len(pred_models),num_recycles=3,return_aux=True)
        p_=aux['log']['plddt']
        plddts[design_id]=p_
        # if p_>0.65:
        model_pdb_path=f'{outdir}/Trajectory/rescore/{design_id}.pdb'
        complex_prediction_model.save_current_pdb(model_pdb_path)
        binder_contacts = hotspot_residues(model_pdb_path)
        binder_contacts_n[design_id] = len(binder_contacts.items())
    metrics['rescore_plddt']={k:round(i,2) for k,i in plddts.items()}
    metrics['rescore_contact']=binder_contacts_n
    return metrics,complex_prediction_model

def _graft_binder(trajectory_pdb:str,rescore_target:str,rescore_chain:str,outpdb:str):
    '''
    make sure traj/rescore pdb are pre-aligned.
    '''
    cmd.load(trajectory_pdb,'trajpdb')
    cmd.load(rescore_target,'rescorepdb')
    cmd.select('to_write',f" ( trajpdb and chain B ) or (rescorepdb and (chain {rescore_chain}))")
    cmd.save(outpdb,'to_write')
    cmd.delete('trajpdb')
    cmd.delete('rescorepdb')

def grafted_rescore_design(metrics:pd.DataFrame,rescore_target:str,chain:str,cyclic:bool=False,seed:int=42,pred_models=[0,1]):
    assert 'B' not in chain
    outdir=outdir_from_metrics(metrics)
    graft_dir,g_s_dir=f'{outdir}/Trajectory/graft',f'{outdir}/Trajectory/graft_rescore'
    os.makedirs(graft_dir,exist_ok=True)
    os.makedirs(g_s_dir,exist_ok=True)

    plddts,binder_contacts_n={},{}
    complex_prediction_model = mk_afdesign_model(
        protocol="binder", 
        num_recycles=3, 
        data_dir='.', 
        use_multimer=False, 
        use_initial_guess=True, 
        use_initial_atom_pos=True)
    for design_id,s in tqdm(metrics.sort_values(by='Sequence', key=lambda x:x.str.len()).iterrows()):
        # only consider rescore from `output/ClpP_a_noLT/trajectory_stats.csv`
        _graft_binder(s['pdbfile'],rescore_target,chain,f'{graft_dir}/{design_id}.pdb')
        seq=s['Sequence']
        complex_prediction_model.prep_inputs(
            pdb_filename=f'{graft_dir}/{design_id}.pdb', 
            chain=chain, binder_chain='B', binder_len=len(seq), 
            use_binder_template=True, rm_target_seq=False,
            rm_target_sc=False, rm_template_ic=True)
        if cyclic:
            add_cyclic_offset(complex_prediction_model, offset_type=2)
        aux=complex_prediction_model.predict(seq=seq,models=pred_models,
                num_models=len(pred_models),num_recycles=3,return_aux=True)
        p_=aux['log']['plddt']
        plddts[design_id]=p_

        model_pdb_path=f'{g_s_dir}/{design_id}.pdb'
        complex_prediction_model.save_current_pdb(model_pdb_path)
        binder_contacts = hotspot_residues(model_pdb_path)
        binder_contacts_n[design_id] = len(binder_contacts.items())
    metrics['rescore_plddt']={k:round(i,2) for k,i in plddts.items()}
    metrics['rescore_contact']=binder_contacts_n
    return metrics,complex_prediction_model
    # return None
