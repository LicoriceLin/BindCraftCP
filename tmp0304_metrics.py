# %% Dependencies
import glob
import os
# from pathlib import Path
from pathlib import PosixPath as Path
from tqdm import tqdm
from subprocess import run
from itertools import product, combinations
from typing import Iterable,Union,Callable,Generator,List,Dict,Tuple
from tempfile import TemporaryDirectory
import pickle as pkl
import json
from ast import literal_eval

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import math

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
from collections.abc import Iterable as collections_Iterable
from Bio.Data import PDBData

import pdbfixer
from openmm.app import PDBFile

from propka.run import single

# %% plt funcs
xkcd_color=lambda x:mcolors.to_rgb(mcolors.XKCD_COLORS[f'xkcd:{x}'])
def configure_rcParams():
    c_rcParams=plt.rcParamsDefault
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{amsmath}",
        'svg.fonttype':'none',
        'font.sans-serif':['Arial','Helvetica',
            'DejaVu Sans',
            'Bitstream Vera Sans',
            'Computer Modern Sans Serif',
            'Lucida Grande',
            'Verdana',
            'Geneva',
            'Lucid',
            'Avant Garde',
            'sans-serif'],
        "pdf.use14corefonts":False,
        'pdf.fonttype':42,
        'text.color':xkcd_color('dark grey'),
        'axes.labelweight':'heavy',
        'axes.titleweight':'extra bold',
        'figure.facecolor':'none',
        'savefig.transparent':True,
            })
    return c_rcParams

def plot_protein_features(
        seq:str, features:List[np.ndarray], 
        feature_names:List[str],colors:List[str|tuple],
        chunk_size:int=30,
        width:float=10.,height_single:float=1.5,
        exclude_annot:List[str]=[]):
    L = len(seq)
    N = len(features)
    # chunk_size = 30
    num_rows = math.ceil(L / chunk_size)
    
    fig, axes = plt.subplots(num_rows*(N + 2),1, figsize=(width, num_rows * height_single),sharex=True)

    if num_rows == 1:
        axes = [axes]
    axes:List[plt.Axes]
    

    def to_annot(feature:np.ndarray,threshold:float=0.5):
        o=[]
        for i in feature.reshape(-1).tolist():
            if i>threshold:
                o.append(f'{int(i*100)}')
            else:
                o.append('')
        return o
    for row in range(num_rows):
        start = row * chunk_size
        end = min((row + 1) * chunk_size, L)
        
        for i, feature in enumerate(features):
            data = feature[start:end].reshape(1, -1)
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["white", colors[i]])
            axes[row*(N+2)+i].imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1,alpha=0.7)
            axes[row*(N+2)+i].set_xticks([])
            axes[row*(N+2)+i].set_yticks([])
            axes[row*(N+2)+i].set_ylabel(feature_names[i], fontsize=6, rotation=0, labelpad=10, va="center")
            if feature_names[i] not in exclude_annot:
                threshold =0.5 if feature_names[i] != 'sasa' else 0.4
                for j, letter in enumerate(to_annot(data,threshold)):
                    axes[row*(N+2)+i].text(j, 0, letter, ha="center", va="center", fontsize=6, fontweight="bold",color=colors[i])
            # axes[row*(N+2)+i]
        # Residue sequence track
        axes[row*(N+2)+N].imshow(np.zeros((1, end - start)), aspect="auto", cmap="Greys")
        axes[row*(N+2)+N].spines['bottom'].set_visible(False)
        axes[row*(N+2)+N].set_xticks([])
        axes[row*(N+2)+N].set_yticks([])
        
        for j, letter in enumerate(seq[start:end]):
            axes[row*(N+2)+N].text(j, 0, letter, ha="center", va="center", fontsize=6, fontweight="bold")
        
        # Residue index
        axes[row*(N+2)+N].set_ylabel(f"{start + 1}-{end}", fontsize=6, rotation=0, labelpad=10, va="center")
        axes[row*(N+2)+N+1].set_axis_off()
    plt.subplots_adjust(wspace=0, hspace=0)

    for ax in axes:
        ax.grid('off')
        ax.margins(x=0.)
        ax.yaxis.get_label().set_horizontalalignment('right')
        for i in ['left','right']: #'top','bottom',
            ax.spines[i].set_visible(False)
    return fig, axes

def plot_feat_MPOP1(df:pd.DataFrame,outpdf:str,cols:List[str]|None=None):
    '''
    hue, order, palette & statistic pairs are hard-coded for MPOP1 system.
    '''
    if cols is None:
        cols=[i for i in df.columns if i not in ['binder','assay_target', 'tag']]

    with PdfPages(outpdf) as pdf:
        for y in tqdm(cols):#['pi-fold', 'pi-unfold', 'pH-opt', 'dG-opt']:
            fig,ax=plt.subplots(1,1,figsize=(6,6))
            order=['NonHit', 'Hit','Promiscuous']
            hue_order=['Skp1', 'FBXW7', 'TcdB']
            palette=sns.color_palette(['tab:blue','tab:orange','tab:purple'])
            sns.boxplot(df,y=y,x='tag',hue='assay_target',fliersize=0,boxprops={'alpha': 0.4},order=order,ax=ax,hue_order=hue_order,palette=palette)
            # sns.violinplot(df,y=y,x='tag',hue='assay_target',alpha=0.4,order=order,ax=ax,hue_order=hue_order,palette=palette)
            sns.stripplot(df,y=y,x='tag',hue='assay_target',ax=ax,order=order,dodge=True,hue_order=hue_order,palette=palette)
            pairs=[('Promiscuous', 'Hit'), ('Hit', 'NonHit')]
            annotator = Annotator(ax, pairs, data=df, x='tag', y=y, order=order,plot='violinplot')
            # pairs=result
            # annotator = Annotator(ax, pairs, data=df, x='tag', y=y, order=order,hue='assay_target',hue_order=hue_order)
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
            annotator.apply_and_annotate()

            ax.set_xticklabels([i.get_text().replace('-','\n') for i in ax.get_xticklabels()])
            ax.set_xlabel('')

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=[(handles[i], handles[i+3]) for i in range(3)],
                labels=hue_order)
            fig.suptitle(y)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def plot_ptm_tracks_MPOP1(ptm_tracks:dict,propka_metrics:pd.DataFrame):
    '''
    ptm_tracks: from `gen_ptm_tracks_MPOP1`
    '''
    feature_names=['ppi', 'sasa', 'MeR','MeK', 'AcK','SuK', 'PiX', 'HoX', 'UbK','Gly'] #'PrQ',
    colors=['navy','tab:blue']+['cyan','teal']+['limegreen']+['crimson','coral']+['rosybrown','slategray','darkseagreen']
    with PdfPages('view_ptm_2.pdf') as pdf:
        for n,o in tqdm(ptm_tracks.items()):
            # o=parse_ptm_track(s,0.5)
            # n=o['name']
            # ptm_tracks[n]=o
            fig,axes=plot_protein_features(
                seq=o['seq'], 
                features=[o[i] for i in feature_names], 
                feature_names=feature_names, 
                colors=colors,
                chunk_size=25,width=8,height_single=2,
                exclude_annot=['ppi'])
            fig.set_dpi(400)
            
            fig.suptitle(o['tag']+'-'+f"{propka_metrics.loc[n]['pi-fold']:.1f}"+'\n'+propka_metrics.loc[n]['binder'])
            pdf.savefig(fig)
            plt.close(fig)
            
# %% MPOP1 sys funcs:
def peel_pdbfile(pdbfile:str):
    '''
    two possible inputs:
        - pdbfile: output/*/Accepted/*_model*.pdb
        - id: shorter_6c0b_500x5_52_dldesign_3_af2pred,pae=5.628-TcdB
    '''
    if pdbfile.endswith('.pdb'):
        pdbfile=Path(pdbfile).stem[:-7]
    if '-' in pdbfile[-10:]:
        return pdbfile
    else:
        return pdbfile+'-Skp1'
    
id2pdbfile=lambda x:glob.glob(f'output/MPOP1-benchmark*/Accepted/{x.replace("-Skp1","")}*.pdb')[0]

def read_bc_metrics_MPOP1()->pd.DataFrame:
    dfs=[]
    for i in glob.glob('output/MPOP1-benchmark*/final_design_stats.csv'):
        df=pd.read_csv(i)
        df['Design']=df['Design'].apply(peel_pdbfile)
        dfs.append(df)
    used_cols=[
        'Design','Length','Target_Hotspot','Sequence','InterfaceResidues',
        'Average_pLDDT','Average_pTM','Average_i_pTM','Average_pAE','Average_i_pAE','Average_i_pLDDT','Average_ss_pLDDT',
        'Average_Unrelaxed_Clashes','Average_Relaxed_Clashes',
        'Average_Binder_Energy_Score',
        'Average_ShapeComplementarity','Average_PackStat',
        'Average_dG','Average_dSASA','Average_dG/dSASA','Average_Interface_SASA_%',
        'Average_Interface_Hydrophobicity','Average_Surface_Hydrophobicity',
        'Average_n_InterfaceResidues','Average_n_InterfaceHbonds','Average_n_InterfaceUnsatHbonds',
        # 'Average_InterfaceUnsatHbondsPercentage','Average_InterfaceHbondsPercentage',
        'Average_Interface_Helix%','Average_Interface_BetaSheet%','Average_Interface_Loop%',
        'Average_Binder_Helix%','Average_Binder_BetaSheet%','Average_Binder_Loop%',
        'Average_InterfaceAAs',
        'Average_Binder_pLDDT','Average_Binder_pTM','Average_Binder_pAE','Average_Binder_RMSD',
        ]
    ori_bc_metrics=pd.concat(dfs,ignore_index=True)[used_cols]
    ori_bc_metrics.columns=[i.replace('Average_','') for i in ori_bc_metrics.columns]
    ori_bc_metrics=ori_bc_metrics.set_index('Design')
    ori_bc_metrics['pdbfile']=[id2pdbfile(i) for i in ori_bc_metrics.index]
    return ori_bc_metrics

def patch_feats(df:pd.DataFrame,ref_csv:str='MPOP1-2.csv')->pd.DataFrame:
    '''
    inplace operations to pad df with 'binder','assay_target' & 'tag'
    '''
    ref_df=pd.read_csv(ref_csv).set_index('Design')
    df['binder']=[ref_df.loc[i]['binder'] for i in df.index]
    df['assay_target']=[i.split('-')[-1] for i in df.index]
    df['tag']=[ref_df.loc[i]['tag'] for i in df.index]

    tag_dtype = CategoricalDtype(categories=['NonHit','Hit','Promiscuous'], ordered=True)
    assay_dtype = CategoricalDtype(categories=['Skp1', 'FBXW7', 'TcdB'], ordered=True)
    df['tag'] = df['tag'].astype(tag_dtype)
    df['assay_target'] = df['assay_target'].astype(assay_dtype)
    df = df.sort_values(by=['tag', 'assay_target'])
    return df


# %% BioPython Utils 
def pdb2seq(pdbfile:str,chain:str='B'):
    # pdbfile='output/PDL1-ls-mcmc/Trajectory/mc_l95_h-8_s33-b1.pdb'
    # chain='B'
    pdb=PDBParser(QUIET=True).get_structure('tmp',pdbfile)[0][chain]
    seq_=[]
    plddt=[]
    for residue in pdb.get_residues():
        seq_+=PDBData.protein_letters_3to1.get(residue.get_resname(),'X')
        plddt.append(residue['CA'].bfactor)
    return ''.join(seq_),plddt

GeorgeDSASA_scale = {
    'ILE':1.850,'VAL':1.645,'LEU':1.931,'PHE':2.228,'CYS':1.461,'CYX':1.461,
    'MET':2.034,'ALA':1.118,'GLY':0.881,'THR':1.525,'TRP':2.663,'SER':1.298,
    'TYR':2.368,'PRO':1.468,'HYP':1.468,'HIS':2.025,'HIP':2.025,'HID':2.025,'HIE':2.025,
    'GLU':1.862,'GLH':1.862,'GLN':1.932,'ASP':1.587,'ASH':1.587,'ASN':1.655,'LYS':2.258,'ARG':2.560,
    }

def cal_sasa(pdbfile:str,chain:str='B',r:bool=True)->np.ndarray:
    '''
    r: relative or absolute
    '''
    pdb=PDBParser(QUIET=True).get_structure(id,pdbfile)[0][chain]
    _ShrakeRupley=ShrakeRupley()
    
    # SASA/seq
    _ShrakeRupley.compute(pdb,level="R")
    sasas=[]
    if r:
        norm=lambda residue: residue.sasa/GeorgeDSASA_scale[residue.get_resname()]/100
    else:
        norm=lambda residue: residue.sasa
    for residue in pdb.get_residues():
        sasas.append(norm(residue))
    return np.array(sasas)

def write_out(strcture:Entity,file:str='tmp.pdb',write_end:bool=True, preserve_atom_numbering:bool=False)->None:
    '''
    write out an Entity (from the whole structure to a single atom) to pdb file.
    use the `write_end` to write an END line, 
    the `preserve_atom_numbering` to renumber atom from 1.
    will be compatible with all `allowed_residue_source` in the future
    '''
    io = BP.PDBIO()
    io.set_structure(strcture)
    io.save(file,write_end=write_end,preserve_atom_numbering=preserve_atom_numbering)

# %% MuteSiteDeep Util
def run_musite(fasta:str,outdir:str,
    repo_dir='/hpf/projects/mkoziarski/zdeng/MusiteDeep_web/',
    ):
    python='/hpf/projects/mkoziarski/zdeng/miniconda3/envs/musite/bin/python'
    outdir_=str(Path(outdir).absolute())
    for i in glob.glob(f'{repo_dir}/MusiteDeep/models/*'):
        model=Path(i).stem
        run([python,
            'predict_multi_batch.py',
            '-input',Path(fasta).absolute(),
            '-output',f'{outdir_}/{model}',
            '-model-prefix',f'models/{model}',],
            cwd=f'{repo_dir}/MusiteDeep')
    with open(f'{outdir}/MUSITE_DONE', 'w') as f:
        f.write('Musite Done\n')
        
single_mutsite_parse=Dict[str,List[Tuple[int,str,float]]]
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

musite_parse_recipe={
    'Methyllysine':('MeK','HIS'),
    'N6-acetyllysine':('AcK','GLN'),
    'Methylarginine':('MeR',''),
    'Phosphoserine_Phosphothreonine':('PiST','GLU'),
    'Phosphotyrosine':('PiY',''),
    'Ubiquitination':('UbK',''),
    'SUMOylation':('SuK',''),
    'O-linked_glycosylation':('GlyO',''),
    'N-linked_glycosylation':('GlyN',''),
    'Hydroxyproline':('HoP',''),
    'Hydroxylysine':('HoK',''),
    'Pyrrolidone_carboxylic_acid':('PrQ',''),
    'S-palmitoyl_cysteine':('SpC',''),
    }

def parse_musite_dir(
    mutsite_dir:str,
    )->Dict[str,Tuple[single_mutsite_parse,str]]:
    '''
    only use recipe's name `k` / short name `v[0]`
    should be safe & rigid.
    '''
    recipe=musite_parse_recipe
    ptms={}
    for k,v in recipe.items():
        ptm_file=Path(mutsite_dir)/f'{k}_results.txt'
        if ptm_file.exists():
            ptms[v[0]]=(parse_musite_single(ptm_file.absolute()),v[1])
        
    return ptms

def gen_ptm_tracks_MPOP1(
    bc_metrics:pd.DataFrame,ptms:Dict[str,Tuple[single_mutsite_parse,str]],
    sasa_threshold:float=0.4,
    ):
    '''
    bc_metrics: from `read_bc_metrics`. future: keep the format, make it generalizable.
    ptms: from `parse_musite_dir`
    '''
    output={}
    for i,s in tqdm(bc_metrics.iterrows()):
        o={}
        o['name']=i
        o['seq']=s['Sequence']
        o['tag']=f"{s['tag']}-{s['assay_target']}"

        o['sasa']=cal_sasa(s['pdbfile'])

        o['ppi']=np.zeros(len(o['seq']),dtype=int)
        o['ppi'][[int(i[1:])-1 for i in s['InterfaceResidues'].split(',')]]=1
        o['ppi'] = o['ppi'] & (o['sasa']>sasa_threshold)
        o['surf']=(o['sasa']>sasa_threshold) & (~o['ppi'])
        o['core']=(o['sasa']<=sasa_threshold).astype(int)

        for k,v in ptms.items():
            o[k]=np.zeros(len(o['seq']),dtype=float)
            ptm=v[0].get(o['name'],[])
            for i in ptm:
                o[k][i[0]-1]=i[2]

        o['PiX']=o['PiST']+o['PiY']
        o['Gly']=o['GlyO']+o['GlyN']
        o['HoX']=o['HoP']+o['HoK']

        output[o['name']]=o

    return output
    
def gen_ana_tracks(
    bc_metrics:pd.DataFrame,
    ptms:Dict[str,Tuple[single_mutsite_parse,str]]|None=None,
    sasa_threshold:float=0.4,
    )->Dict[str,Dict[str,str|np.ndarray]]:
    '''
    bc_metrics: from `read_bc_metrics`. future: keep the format, make it generalizable.
    ptms: from `parse_musite_dir`
    '''
    output={}
    for i,s in tqdm(bc_metrics.iterrows()):
        o={}
        if 'Design' in s:
            o['name']=s['Design']
        else:
            o['name']=i
        o['seq']=s['Sequence']
        # o['tag']=f"{s['tag']}-{s['assay_target']}"

        o['sasa']=cal_sasa(s['pdbfile'])

        o['ppi']=np.zeros(len(o['seq']),dtype=int)
        if s['InterfaceResidues']!='':
           o['ppi'][[int(i[1:])-1 for i in s['InterfaceResidues'].split(',')]]=1
        o['ppi'] = o['ppi'] & (o['sasa']>sasa_threshold)
        o['surf']=(o['sasa']>sasa_threshold) & (~o['ppi'])
        o['core']=(o['sasa']<=sasa_threshold).astype(int)
        if ptms is not None:
            for k,v in ptms.items():
                o[k]=np.zeros(len(o['seq']),dtype=float)
                ptm=v[0].get(o['name'],[])
                for i in ptm:
                    o[k][i[0]-1]=i[2]

            o['PiX']=o['PiST']+o['PiY']
            o['Gly']=o['GlyO']+o['GlyN']
            o['HoX']=o['HoP']+o['HoK']

        output[o['name']]=o

    return output

def keep_max_ptm(ptm_track:Dict[str,str|np.ndarray],
    used_ptms:List[str]=['MeK', 'AcK', 'MeR', 'PiST', 'PiY', 'UbK', 'SuK','GlyO', 'GlyN','HoP', 'HoK']): #'PrQ', 'HoP', 'HoK' are weird
    arrays= [ptm_track[i] for i in used_ptms]
    A = np.stack(arrays)
    max_values = np.max(A, axis=0) 
    mask = A == max_values
    A_filtered = A * mask
    ptm_track.update(
        {k:A_filtered[i] for i,k in enumerate(used_ptms)})
    return ptm_track

def cal_ptm_feat(ptm_track:Dict[str,str|np.ndarray],
        used_ptms=['MeK', 'AcK', 'MeR','PiX','UbK', 'SuK', 'Gly','HoX' ],#'HoP', 'HoK' #,'PrQ'
        used_loc=['ppi','surf','all_surf'], #'core'
        max_only:bool=False,
        threshold=0.5):
    # ptm_track=ptm_tracks['5v4b_chainA_1000x2_984_dldesign_0_af2pred,pae=4.79-Skp1']
    if 'all_surf' in used_loc and 'all_surf' not in ptm_track:
        ptm_track['all_surf']=ptm_track['ppi']+ptm_track['surf']
    if max_only:
        ptm_track=keep_max_ptm(ptm_track.copy(),used_ptms)
    o={}
    for ptm in used_ptms:
        for loc in used_loc:
            o[f'{loc}-{ptm}']=((ptm_track[ptm]>threshold) & (ptm_track[loc])).sum()
    return o

# %% ProPka Util
def propka_single(pdbfile:str,optargs:List[str]=['-c=B','--protonate-all']):
    o=single(pdbfile,optargs=optargs,write_pka=False)
    pif,piu=o.get_pi()
    profile, [ph_opt, dg_opt], [dg_min, dg_max], [ph_min, ph_max] = (
        o.get_folding_profile())
    pis={
        'pi-fold':pif,'pi-unfold':piu,
        "pH-opt":ph_opt, "dG-opt":dg_opt
        }
    return pis

def ptm_propka(pdbfile:str,
    ptms:Dict[str,Tuple[Dict[str,Tuple[int,str,float]],str]],
    ptm_threshold:float=0.5,
    mut_recipe:Dict[str,Tuple[str,str]]=musite_parse_recipe,
    sasa_threshold:float=0.,
    MPOP1:bool=False):
    '''
    ptms: from `parse_musite_dir`

    '''
    # ptms=ptms.copy()
    mut_recipe_={v[0]:v[1] for v in mut_recipe.keys()}
    for k in ptms.keys():
        ptms[k]=(ptms[k][0],mut_recipe_.get(k,''))
    
    with TemporaryDirectory() as tmpdir:
        chain='B'
        pdb=PDBParser(QUIET=True).get_structure('tmp',pdbfile)[0][chain]
        write_out(pdb,f'{tmpdir}/{Path(pdbfile).stem}.pdb')
        if sasa_threshold>0:
            sasas=cal_sasa(f'{tmpdir}/{Path(pdbfile).stem}.pdb')
        
        mutstrs={}
        for k,v in ptms.items():
            # no need to overwrite 
            if v[1]:
                if MPOP1:
                    des=peel_pdbfile(pdbfile)
                else:
                    des=Path(pdbfile).stem[:-7]
                for l in v[0].get(des,[]):
                    t=l[2]>ptm_threshold and l[2]>=mutstrs.get(l[0],('',0.))[1]
                    if sasa_threshold>0:
                        t= t and sasas[l[0]-1]>sasa_threshold
                    if t:
                        # if v[1]:
                        mutstrs[l[0]]=(f"{PDBData.protein_letters_1to3[l[1]]}-{l[0]}-{v[1]}",l[2])
        # return mutstrs
                        # else:
                        #     if l[0] in mutstrs:
                        #         mutstrs.pop(l[0])

        if len(mutstrs)>0:    
            fixer = pdbfixer.PDBFixer(filename=f'{tmpdir}/{Path(pdbfile).stem}.pdb')
            fixer.applyMutations([i[0] for i in mutstrs.values()], 'B')
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            PDBFile.writeFile(fixer.topology, fixer.positions, open(f'{tmpdir}/{Path(pdbfile).stem}.pdb', 'w'))
            # return PDBParser(QUIET=True).get_structure('tmp',f'{tmpdir}/{Path(pdbfile).stem}.pdb')[0]
            pis=propka_single(f'{tmpdir}/{Path(pdbfile).stem}.pdb',optargs=[f'-c=A'])
        else:
            pis=propka_single(f'{tmpdir}/{Path(pdbfile).stem}.pdb',optargs=[f'-c=B'])
        return pis
    
# %% ESM_IF Utils
def run_esm_if(
    pdbs:List[str],
    chains:str|List[str],
    )->Dict[str,pd.DataFrame]:
    python='/hpf/projects/mkoziarski/zdeng/miniconda3/envs/stab_esm_if/bin/python'
    script='/hpf/projects/mkoziarski/zdeng/BindCraft/tmp0305_esm_stab.py'
    pdbs_=[]
    for i in pdbs:
        if ';' in i:
            raise ValueError('No ";" allowed in file name')
        pdbs_.append(str(Path(i).absolute()))
    # return pdbs_
    if isinstance(chains,list):
        c=';'.join(chains)
    else:
        c=chains
    
    with TemporaryDirectory() as tdir:
        run(
            [python,script,
             '--pdbs',';'.join(pdbs_),
             '--chains',c,
             '--outpkl','stb.pkl'],
             cwd=tdir
        )
        o=pkl.load(open(f'{tdir}/stb.pkl','rb'))
    return o