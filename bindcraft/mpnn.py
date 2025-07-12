from __future__ import annotations
from typing import TYPE_CHECKING
from ._import import *
from .util import id2pdbfile,is_pickleable
from itertools import product, combinations
from sklearn.cluster import SpectralClustering
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.mpnn import mk_mpnn_model
if TYPE_CHECKING:
    from .post_design import Metrics

from .diversity_util import simple_diversity,cluster_and_get_medoids

# %% MPNN biased generation

aa_order = residue_constants.restype_order
order_aa = {b:a for a,b in aa_order.items()}
all_aa=''.join(aa_order.keys())

def _add_bias(bias:np.ndarray,pos:List[int],aas:str,penalty:float):
    aa=[aa_order[a] for a in aas]
    bias[np.ix_(pos,aa)]-=penalty
    bias[pos,-1]=1
    return bias

def _fix_pos(bias:np.ndarray,pos:List[int]):
    raise NotImplementedError

_ptm_threshold=0.5
_ppi_esm_threshold=0.05
_other_esm_threshold=0.5
_default_penalties_values=dict(Sp=math.log(1/0.5),Hp=math.log(1/0.25),No=float(1e6),Null=0.)

class PenaltyRecipe:
    def __init__(self,name:str,select_func_expr:str="lambda s: s['seq']=='C' and s['surf']",
        aas_s:Dict[str,str]={'Null':all_aa},penalties_values:Dict[str,float]=_default_penalties_values,
        select_func:Callable[[pd.Series],bool]|None=None
        ):
        '''
        select_func_expr: something could be solved by `eval` into select_func
        aas_s: something like: {'No':'C','Hp':'N','Sp':'DE'}
        penalties_values: something like: {'Hp',-4}
        select_func would override select_func_expr. only use named func for it.
        '''
        self.name=name
        if select_func is not None:
            self.select_func:Callable[[pd.Series],bool]=select_func
            self.select_func_expr=''
        else:
            self.select_func:Callable[[pd.Series],bool]=eval(select_func_expr)
            self.select_func_expr=select_func_expr
        self.aas_s=aas_s
        self.penalties=penalties_values
    
    def add_bias(self,track_df:pd.DataFrame,bias:np.ndarray):
        sel=track_df.apply(self.select_func,axis=1).astype(bool)
        if sel.any():
            pos=track_df.loc[sel].index.to_list()
        # if len(pos)>0:
            for k,aas in self.aas_s.items():
                penalty=self.penalties.get(k,0.)
                bias=_add_bias(bias,pos,aas,penalty)
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
        # if penalty is None:
        #     self.penalty=penalties.get(self.name.split('-')[-1],0.)
        # else:
        #     self.penalty=penalty

default_penalty_recipes=[
    PenaltyRecipe('Surf-C',"lambda s: s['seq']=='C' and s['surf']",{'No':'C','Sp':'DE'}),
    PenaltyRecipe('Surf-DE',"lambda s: s['seq'] in ['D','E'] and s['surf']",{'No':'C','Sp':'DE'}),
    PenaltyRecipe('Surf-MeR',"lambda s: s['MeR']>_ptm_threshold and s['surf']",{'No':'C','Sp':'DE','Hp':'R'}),
    
    PenaltyRecipe('Surf/PPi-GlyN',"lambda s: s['GlyN']>_ptm_threshold and (not s['core'])",{'No':'C','Hp':'N','Sp':'DE'}),
    PenaltyRecipe('Surf/PPi-GlyO',"lambda s: s['GlyO']>_ptm_threshold and (not s['core'])",{'No':'C','Hp':'ST','Sp':'DE'}),

    PenaltyRecipe('PPI-C',"lambda s: s['seq']=='C' and s['ppi']",{'Sp':'MK','Hp':'C'}),
    PenaltyRecipe('PPI-MeR',"lambda s: s['MeR']>_ptm_threshold and s['ppi']",{'Sp':'M','Hp':'CR'}),
    PenaltyRecipe('PPI-AcK',"lambda s: s['AcK']>_ptm_threshold and s['ppi']",{'Sp':'M','Hp':'CK'}),
    
    PenaltyRecipe('PPI-esm_if',"lambda s: s['esm_if']<_ppi_esm_threshold and s['ppi']", {'Sp':'MK','Hp':'C'}),
    # PenaltyRecipe('Surf-esm_if',lambda s: s['esm_if']<0.5 and s['surf'], {'No':'C','Sp':'DE'}),
    PenaltyRecipe('Surf-esm_if',"lambda s: s['esm_if']<_other_esm_threshold and s['surf']", {'No':'C','Hp':'DE'}),
    PenaltyRecipe('Core-esm_if',"lambda s: s['esm_if']<_other_esm_threshold and s['core']", {'Null':all_aa}),

    PenaltyRecipe('PPI-interact',"lambda s: s['interacts']==-1",{'Hp':'C'}),
    ]

minimal_penalty_recipes=[
    PenaltyRecipe('Surf',"lambda s: bool(s['surf'])",{'No':'C'}),
    PenaltyRecipe('Core',"lambda s: bool(s['core'])",{'Null':all_aa}),
    ]

def gen_mpnn_bias(
    metrics:pd.DataFrame,
    ana_tracks:Dict[str,Dict[str,str|np.ndarray]],
    penalty_recipes:List[PenaltyRecipe]=default_penalty_recipes,
    filter:bool=True,
    ):
    '''
    '''
    # design_metrics=check_filters(design_metrics,filters)

    mpnn_bias:Dict[str,np.ndarray]={}
    for design_id,s in tqdm(metrics.iterrows()):
        if filter and s['filt']!='All_Good':
            # o[s['Design']]=(False, np.zeros((len(track['seq']),21)))
            continue
        else:
            bias=np.zeros((len(s['Sequence']),21))
            
            track=ana_tracks[design_id].copy()
            track['seq']=list(track['seq'])
            track.pop('name')
            track_df=pd.DataFrame(track)
            # break
            for r in penalty_recipes:
                bias=r.add_bias(track_df,bias)
            mpnn_bias[design_id] = bias
    return metrics,ana_tracks,mpnn_bias

# %% post process of MPNN results
def dedup_mpnn_results(mpnn_sequences,length:int,design_id:str):
    _={
        mpnn_sequences['seq'][n][-length:]: {
            'seq': mpnn_sequences['seq'][n][-length:],
            'score': mpnn_sequences['score'][n],
            'seqid': mpnn_sequences['seqid'][n]
        } for n in range(30)}
    mpnn_df=pd.DataFrame()
    mpnn_df['seq']=[i['seq'] for i in _.values()]
    mpnn_df['score']=[i['score'] for i in _.values()]
    mpnn_df=mpnn_df.sort_values(by='score',ascending=False)
    mpnn_df.index=[f'{design_id}-m{i+1}' for i in range(len(mpnn_df))]
    return mpnn_df

# def identity(x:str,y:str):
#     m=0
#     for i,j in zip(x,y):
#         if i==j:
#             m+=1
#     return m/len(x)
    
# def run_diversity(df:pd.DataFrame):
#     '''
#     same-length simple diversity
#     '''
#     odf=pd.DataFrame(columns=df.index,index=df.index)
#     for i, j in combinations(df['seq'].index, 2):
#         odf.at[i, j] = odf.at[j, i] = 1/(identity(df['seq'][i], df['seq'][j])+ 1e-3)
#     odf=odf.fillna(1.)
#     return odf

# def cluster_and_get_medoids(distance_matrix, num_clusters=5):
#     spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
#     labels = spectral.fit_predict(distance_matrix)

#     medoids = []
#     for cluster_id in range(num_clusters):
#         cluster_points = np.where(labels == cluster_id)[0]  
#         # if len(cluster_points) == 0:
#         #     continue  

#         # cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
        
#         # medoid_index = cluster_points[np.argmin(cluster_distances.sum(axis=1))]
#         medoids.append(cluster_points[0])

#     return medoids, labels

# %% Run MPNN
def run_mpnn(
    ana_tracks:Dict[str,Dict[str,str|np.ndarray]],
    designid2pdb:Callable[[str],str]|None=None,
    mode:Literal['bias',"whole",'non_ppi']='bias', # redundant: "whole",'non_ppi' should be treat as a certain type of bias
    mpnn_bias:Dict[str,np.ndarray]|None=None,
    max_mpnn_sequences:int=5,num_sample:int=30,seed:int=42,
    outdir:str|None=None,
    model_params:Dict[str,Any]=dict(backbone_noise=0.0,model_name='v_48_020',weights='soluble',seed=42)
    ):
    '''
    `outdir` are used for default `designid2pdb`
    `seed` would overwrite seed in `model_params`
    '''
    model_params.update({'seed':seed})
    mpnn_dfs=[]
    # with open(f'{outdir}/Trajectory/Post/mpnn.fa','w') as f:
    if mpnn_bias is None:
        assert mode!='bias'
        mpnn_bias={}
    if designid2pdb is None:
        assert outdir is not None
        designid2pdb=lambda designid:id2pdbfile(designid,outdir,'design')

    # for design_id,bias in tqdm(mpnn_bias.items()):
    mpnn_model = mk_mpnn_model(**model_params)
    for design_id,ana_track in tqdm(ana_tracks.items()):
        if mode =='bias':
            bias=mpnn_bias.get(design_id,None)
            if bias is None:
                continue #weird: just skip those without bias? How about treat them as non-biased?
            # length=bias.shape[0]
            fix_po=np.where(bias[:,-1]==0)[0]
            fixed_positions = 'A,' + ','.join(f'B{i+1}' for i in fix_po)
        elif mode == 'non_ppi':
            fix_po=np.where(ana_track['ppi']==1)[0]
            fixed_positions = 'A,' + ','.join(f'B{i+1}' for i in fix_po)
        elif mode == 'whole':
            fixed_positions = 'A' 
        else:
            raise ValueError
        
        length=len(ana_track['seq'])
        pdbfile=designid2pdb(design_id)
        if mode =='bias':
            mpnn_model.prep_inputs(pdbfile,'A,B',fix_pos=fixed_positions)
            mpnn_model._inputs['bias'][-length:]+=bias[:,:-1]
        else:
            mpnn_model.prep_inputs(pdbfile,'A,B',fix_pos=fixed_positions,rm_aa='C')

        mpnn_sequences = mpnn_model.sample(temperature=0.1, num=num_sample, batch=num_sample)
        mpnn_df=dedup_mpnn_results(mpnn_sequences,length,design_id)
        dis_df=simple_diversity(mpnn_df)
        if len(dis_df)>max_mpnn_sequences:
            medoids, labels=cluster_and_get_medoids(dis_df.to_numpy(),max_mpnn_sequences)
            mpnn_df_=mpnn_df.iloc[medoids].copy()
        else:
            mpnn_df_=mpnn_df
        target_seq=mpnn_sequences['seq'][0].split('/')[0]
        ori_score=mpnn_model.score(target_seq+ana_track['seq'])['score']
        mpnn_df_.loc[design_id]={'seq':ana_track['seq'],'score':ori_score}
        mpnn_df_['design_id']=design_id
        mpnn_dfs.append(mpnn_df_)
    if len(mpnn_dfs)>0:
        sum_mpnn_df=pd.concat(mpnn_dfs)
    else:
        sum_mpnn_df=pd.DataFrame(columns=['Design','seq','score','design_id']).set_index('Design')
    return sum_mpnn_df

_default_mpnn_model_params=dict(backbone_noise=0.0,model_name='v_48_020',weights='soluble',seed=42)
class MPNNSampler:
    def __init__(self,
        metrics:Metrics,
        filter:bool=True,
        penalty_recipes:List[PenaltyRecipe]=default_penalty_recipes,
        model_params:Dict[str,Any]=_default_mpnn_model_params
        ):
        '''
        filter: only run those with 'All_Good' in metrics.metrics['filt']
        '''
        self.Metrics=metrics
        self.filter=filter
        self.penalty_recipes=penalty_recipes
        self.model_params=model_params

    def cal_bias(self):
        self.bias=gen_mpnn_bias(self.Metrics.metrics,
            self.Metrics.ana_tracks,self.penalty_recipes,self.filter)[-1]
        return self.bias
    
    def run_mpnn(self,
        mode:Literal['biased',"whole",'non_ppi']='bias',
        max_mpnn_sequences:int=5,num_sample:int=30,seed:int=42):

        metrics,ana_tracks=self.Metrics.metrics,self.Metrics.ana_tracks
        if self.filter:
            ana_tracks={k:v for k,v in ana_tracks.items() if metrics['filt'][k]=='All_Good'}

        if mode=='bias':
            if not hasattr(self,'bias'):
                self.cal_bias()
            bias=self.bias
        else:
            bias=None
        if hasattr(self.Metrics,'_rescorer'):
            designid2pdb=lambda i:self.Metrics._rescorer.rescore_df['r_pdb'][i]
        else:
            designid2pdb=lambda i:self.Metrics._pdbs[i]
        self.mpnn_df=run_mpnn(
            ana_tracks=ana_tracks,
            designid2pdb=designid2pdb,
            mode=mode,mpnn_bias=bias,
            max_mpnn_sequences=max_mpnn_sequences,
            num_sample=num_sample,seed=seed,
            model_params=self.model_params,
            )
        
        return self.mpnn_df

    def dump_samples(self,dir_name:str='Post',file_stem:str='MPNN'):
        self.mpnn_df.to_csv(self.Metrics.outdir+'/'+self.Metrics._subdir+f'/{dir_name}/{file_stem}.csv',index_label='Design')

    def load_samples(self,dir_name:str='Post',file_stem:str='MPNN'):
        self.mpnn_df=pd.read_csv(self.Metrics.outdir+'/'+self.Metrics._subdir+f'/{dir_name}/{file_stem}.csv').set_index('Design')
# def mpnn_grafted_rescore():
#     pass