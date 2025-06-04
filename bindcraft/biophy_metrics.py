# %% Dependencies
# import glob
# import os
# # from pathlib import Path
# from pathlib import PosixPath as Path
# from tqdm import tqdm
# from subprocess import run
# from itertools import product, combinations
# from typing import Iterable,Union,Callable,Generator,List,Dict,Tuple
# from tempfile import TemporaryDirectory
# import pickle as pkl
# import json
# from ast import literal_eval

# import pandas as pd
# from pandas.api.types import CategoricalDtype
# import numpy as np
# import math

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns
# from statannotations.Annotator import Annotator

# import Bio.PDB as BP
# from Bio.PDB import PDBParser
# from Bio.PDB.Entity import Entity
# from Bio.PDB.Atom import Atom
# from Bio.PDB.Residue import Residue

# from Bio.PDB.SASA import ShrakeRupley
# from collections.abc import Iterable as collections_Iterable
# from Bio.Data import PDBData
from ._import import *
from .util import write_out,pdbfile2id
import pdbfixer
from openmm.app import PDBFile
from propka.run import single

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
        # o['ppi1'] = o['ppi'] #& (o['sasa']>sasa_threshold)
        o['surf']=(o['sasa']>sasa_threshold) & (~o['ppi'])
        o['core']=(o['sasa']<=sasa_threshold) & (~o['ppi'])  #(o['sasa']<=sasa_threshold).astype(int)
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
    ): #MPOP1:bool=False
    '''
    ptms: from `parse_musite_dir`

    '''
    # ptms=ptms.copy()
    mut_recipe_={v[0]:v[1] for v in mut_recipe.values()}
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
                # if MPOP1:
                #     des=peel_pdbfile(pdbfile)
                # else:
                des=pdbfile2id(pdbfile)
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
        pis['mutstrs']=mutstrs
        return pis
    
# %% ESM_IF Utils
def run_esm_if(
    pdbs:List[str],
    chains:str|List[str],
    )->Dict[str,pd.DataFrame]:
    python='/hpf/projects/mkoziarski/zdeng/miniconda3/envs/stab_esm_if/bin/python'
    script='/hpf/projects/mkoziarski/zdeng/BindCraft/bindcraft/esm_stab.py'
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