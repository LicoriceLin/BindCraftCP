from pymol import cmd
import matplotlib.pyplot as plt
import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1
import pandas as pd
from typing import List,Tuple

def no_organic_purify(obj:str):
    cmd.remove(f'{obj} and not polymer')
    cmd.h_add(obj)

def cal_sasa(obj:str):
    cmd.set('dot_solvent',1)
    cmd.set('dot_density',3)
    cmd.get_area(obj,load_b=1)

def get_group_sasa(sel:str):
    bs = []
    cmd.iterate(sel, 'bs.append(b)',space={'bs':bs})
    return sum(bs)

def get_resname(chain:str,resi:int,obj:str=''):
    result = []
    sel=f'chain {chain} and resi {resi}'
    if obj:
        sel += f' and {obj}'
    cmd.iterate(sel , "result.append(resn)", space={'result': result})
    return result[0] if result else None

def resn3_to_1(resn):
    return protein_letters_3to1.get(resn.capitalize(), 'X')

def _to_sel(chain:str, resi:str,obj:str=''):
    sel = f'chain {chain} and resi {resi}'
    if obj:
        sel += f' and {obj}'
    return sel

def res_sasa(chain:str, resi:str,obj:str=''):
    sel=_to_sel(chain,resi,obj)
    bs=[]
    cmd.iterate(sel , "bs.append(b)", space={'bs': bs})
    return sum(bs)


def sasa_scale(resn:str)->float:
    return GeorgeDSASA_scale.get(resn,1.118)


def rSASA(obj:str,chain:str):
    cal_sasa(obj)
    residues = {}
    cmd.iterate(f'name ca  and (chain {chain}) and {obj}', 
        'residues.update({(chain, resi): res_sasa(chain, resi, obj)/sasa_scale(resn)}) ', # resn3_to_1(resn),
        space={"residues":residues,"resn3_to_1":resn3_to_1,'res_sasa':res_sasa,"sasa_scale":sasa_scale,"obj":obj})
    return residues

GeorgeDSASA_scale = {
    'ILE':1.850,'VAL':1.645,'LEU':1.931,'PHE':2.228,'CYS':1.461,'CYX':1.461,
    'MET':2.034,'ALA':1.118,'GLY':0.881,'THR':1.525,'TRP':2.663,'SER':1.298,
    'TYR':2.368,'PRO':1.468,'HYP':1.468,'HIS':2.025,'HIP':2.025,'HID':2.025,'HIE':2.025,
    'GLU':1.862,'GLH':1.862,'GLN':1.932,'ASP':1.587,'ASH':1.587,'ASN':1.655,'LYS':2.258,'ARG':2.560,
    }

def hotspots_by_ligand(obj:str,target_chain:str,ligand_chain:str):
    cmd.create('complex',f'{obj} and (chain {target_chain},{ligand_chain})')
    holo_rsasa=rSASA('complex',f'{target_chain},{ligand_chain}')
    cmd.create('target',f'{obj} and (chain {target_chain})')
    apo_rsasa=rSASA('target',target_chain)
    hotspots=[]
    for k,v in apo_rsasa.items():
        # diff=
        if v-holo_rsasa[k]>0:
            hotspots.append(k)
    return {'hotspots':hotspots,'holo_rsasa':holo_rsasa,'apo_rsasa':apo_rsasa}


def _reduce_hotspot_list(hotspots:List[Tuple[str,str]]):
    res={}
    for h in hotspots:
        res.setdefault(h[0],[]).append(int(h[1]))
    return res

def hotspots_to_seg_continue(obj:str,hotspots:List[Tuple[str,str]],opt_obj:str='continue_seg',pad=25):
    res=_reduce_hotspot_list(hotspots)
    res_be=[]
    for k,v in res.items():
        res_be.append(f'(chain {k} and resi {max(min(v)-pad,1)}-{max(v)+pad})')
    cmd.copy_to(opt_obj,f'{obj} and ( {" or ".join(res_be) } )')

def hotspots_to_seg_surf(obj:str,hotspots:List[Tuple[str,str]],opt_obj:str='surf',vicinity=10):
    res=_reduce_hotspot_list(hotspots)
    res_sel=[]
    for k,v in res.items():
        res_sel.append(f'(chain {k} and resi {",".join([str(i) for i in v])})')
    cmd.select('hotspots',f'{obj} and ( {" or ".join(res_sel) } )')
    cmd.create(opt_obj,f'byres ( ({obj} and chain {",".join(res.keys())}) within {vicinity} of hotspots)')


### ###


def no_inorganic_purify(obj:str):
    cmd.remove(f'{obj} and inorganic')
    cmd.h_add(obj)



    