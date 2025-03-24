from __future__ import annotations
from typing import TYPE_CHECKING
from ._import import *
from .util import (id2pdbfile,outdir_from_metrics,
    read_bc_metrics,read_design_metrics,_meta_cols,_aa_col,
    _simpliest_filter,show_filter,check_filters,filters_type
    )

from numpy.lib.stride_tricks import sliding_window_view
from colabdesign import clear_mem
from functions import hotspot_residues,mk_afdesign_model,add_cyclic_offset

from .biophy_metrics import (
    run_musite,parse_musite_dir,run_esm_if,gen_ana_tracks,
    pdb2seq,propka_single,ptm_propka,musite_parse_recipe
)
from .mda_metrics import cal_ppi_interacts,cal_rog

# if TYPE_CHECKING:
from .mpnn import MPNNSampler,minimal_penalty_recipes,PenaltyRecipe


# %% post analysis
def _match_pattern(seq:str)->np.ndarray:
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
            v['LCR']=_match_pattern(v['seq'])
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
        saved_internal=['_subdir',]
        if ckpt is None:
            ckpt=self.ana_paths['Metrics']
        d={k:v for k,v in self.__dict__.items() if (not k.startswith('_')) and (k not in saved_internal)}
        with open(ckpt, 'wb') as f:
            pkl.dump(d, f)

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
    
    def filter(self,filters:filters_type=_simpliest_filter):
        self.filters=filters
        check_filters(self.metrics,filters)

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
        
    def __getitem__(self, key:str|Iterable[str]|int|Iterable[int]|slice):
        if isinstance(key,str):
            key=[key]
        elif isinstance(key,int):
            key=[self.keys()[key]]
        elif isinstance(key,slice):
            key=self.keys()[key]
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
    
    def init_rescore(self,
        cyclic:bool=False,
        model_kwargs:Dict[str,bool]=dict(
            use_multimer=False, 
            use_initial_guess=True, 
            use_initial_atom_pos=True),
        reload_rescore_dir:str|None=None,
        reload_mpnn:bool=False):
        self._rescorer=ReScorer(self,cyclic=cyclic,model_kwargs=model_kwargs)
        if reload_rescore_dir is not None:
            self._rescorer.reload_rescore_log(rescore_dir=reload_rescore_dir,mpnn=reload_mpnn)
        return self._rescorer
    
    def init_visualize(self,
        ana_dir_name:str='Post',
        filters:Dict[str,Dict[str,float|int|bool]]=_simpliest_filter):
        self.filters=filters
        self._visualizer=MetricsVisualizer(self,ana_dir_name)
        return self._visualizer
    
    def init_mpnn_sample(self,filter=True,
        penalty_recipes:List[PenaltyRecipe]=minimal_penalty_recipes,
        model_params:Dict[str,Any]=dict(backbone_noise=0.0,model_name='v_48_020',weights='soluble',seed=42)):
        self._mpnn_sampler=MPNNSampler(self,filter=filter,penalty_recipes=penalty_recipes,model_params=model_params)

        return self._mpnn_sampler
        
        
class MetricsVisualizer:
    def __init__(self,
        metrics:Metrics,
        ana_dir_name:str='Post',
        # filters:Dict[str,Dict[str,float|int|bool]]=_simpliest_filter
        ):
        self.Metrics=metrics
        self._metrics=self.Metrics.metrics.copy()
        # self.filters=filters
        subdir=metrics._subdir
        self.anadir=self.Metrics.outdir+f'/{subdir}/{ana_dir_name}'
        os.makedirs(self.anadir,exist_ok=True)

    def plot_filter(self,stem:str='filter'):
        assert hasattr(self.Metrics,'filters'), 'run `self.Metrics.filter` first.'
        fig,ax=show_filter(metrics=self._metrics) #,filters=self.filters
        plt.savefig(f'{self.anadir}/{stem}.svg')
        plt.savefig(f'{self.anadir}/{stem}.png')
        return fig,ax

    def plot_dist(self,stem:str='metric_dist',
            metrics_cols:List[str]=[],exclude_cols:List[str]=[]):
        assert hasattr(self.Metrics,'filters'), 'run `self.Metrics.filter` first.'
            
        filter_dict=self.Metrics.filters
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
        seed:int=42,pred_models=[0]):
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
        self.rescore_df.to_csv(r_dir+'/log.csv',index_label='Design')
        return self.rescore_log
    
    def rescore_grafttemp(self,rescore_target:str,rescore_chain:str,
        graft_dir:str='graft',rescore_dir:str='rescore',
        seed:int=42,pred_models=[0]):
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
        self.rescore_df=pd.DataFrame(self.rescore_log).T
        self.rescore_df.to_csv(r_dir+'/log.csv',index_label='Design')
        return self.rescore_log

    def rescore_from_mpnn(self,mpnn_df:pd.DataFrame,mpnn_rescore_dir:str='mpnn_rescore',seed=42,pred_models=[0]):
        # for design_id, sub_df in mpnn_df.groupby(by='design_id')
        m_dir=self.Metrics.outdir+f'/{self.Metrics._subdir}/'+mpnn_rescore_dir
        self.m_dir=m_dir
        os.makedirs(m_dir,exist_ok=True)
        model=self.complex_prediction_model
        logs={}
        for design_id,sub_df in tqdm(mpnn_df.groupby(by='design_id')):
            model.prep_inputs(
                pdb_filename=self.rescore_df['r_pdb'][design_id], 
                chain='A', binder_chain='B', binder_len=len(sub_df['seq'].iloc[0]), 
                use_binder_template=True, rm_target_seq=False,
                rm_target_sc=False, rm_template_ic=True,seed=seed)
            if self.cyclic:
                add_cyclic_offset(model, offset_type=2)
            for mpnn_id,seq in sub_df['seq'].items():
                aux:Dict[str,float]=model.predict(seq=seq,models=pred_models,
                    num_models=len(pred_models),num_recycles=3,return_aux=True)['log']
                for i in ['hard','soft','temp','seqid','recycles','models']:
                    aux.pop(i,None)
                model_pdb_path=f'{m_dir}/{mpnn_id}.pdb'
                if os.path.exists(model_pdb_path):
                    os.remove(model_pdb_path)
                model.save_current_pdb(model_pdb_path)
                binder_contacts = hotspot_residues(model_pdb_path)
                aux['n_contact'] = len(binder_contacts.items())
                aux['contacts']=binder_contacts
                aux['r_pdb']=model_pdb_path
                logs[mpnn_id]=aux
        self.mpnn_rescore_log=logs
        with open(m_dir+'/log.pkl','wb') as f:
            pkl.dump(logs,f)
        self.mpnn_rescore_df=pd.DataFrame(self.mpnn_rescore_log).T
        self.mpnn_rescore_df.to_csv(m_dir+'/log.csv',index_label='Design')
        return self.mpnn_rescore_df
    
    def merge_rescore_metrics(self,use_col=['rmsd','plddt','n_contact','i_pae']):
        m_df:pd.DataFrame=self.rescore_df[use_col]
        m_df.columns=[f'r:{i}' for i in m_df]
        for c in m_df.columns:
            if c in self.Metrics.metrics.columns:
                self.Metrics.metrics.drop(c,axis=1,inplace=True)
        self.Metrics.metrics=pd.merge(left=self.Metrics.metrics,right=m_df,left_index=True,right_index=True)
        return self.Metrics.metrics[m_df.columns]

    def reload_rescore_log(self,rescore_dir:str='rescore',mpnn:bool=False):
        r_dir=self.Metrics.outdir+f'/{self.Metrics._subdir}/'+rescore_dir
        with open(r_dir+'/log.pkl','rb') as f:
            rescore_log=pkl.load(f)
        rescore_df=pd.DataFrame(self.rescore_log).T

        if mpnn:
            self.m_dir=r_dir
            self.mpnn_rescore_log=rescore_log
            self.mpnn_rescore_df=rescore_df
        else:
            self.r_dir=r_dir
            self.rescore_log=rescore_log
            self.rescore_df=rescore_df
        return rescore_df
    
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
        return self.rescore_df.columns.to_list()
   
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

