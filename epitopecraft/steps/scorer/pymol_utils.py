from pymol import cmd
import matplotlib.pyplot as plt
import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1
import pandas as pd
from typing import Optional,List,Tuple,Dict
from pathlib import Path

### pre-process ###
def default_vis_config():
    cmd.set('ray_trace_mode',1)
    cmd.set('ray_shadow','off')
    cmd.set('antialias',2)
    cmd.set('ambient',0.6)
    cmd.set('specular',0.5)
    cmd.set('reflect',0.2)
    cmd.set('ray_opaque_background',0)

def no_organic_purify(obj:str):
    cmd.remove(f'{obj} and not polymer')
    cmd.h_add(obj)

def no_inorganic_purify(obj:str):
    cmd.remove(f'{obj} and not organic and not polymer')
    cmd.h_add(obj)

### SASA ###
GeorgeDSASA_scale = {
    'ILE':1.850,'VAL':1.645,'LEU':1.931,'PHE':2.228,'CYS':1.461,'CYX':1.461,
    'MET':2.034,'ALA':1.118,'GLY':0.881,'THR':1.525,'TRP':2.663,'SER':1.298,
    'TYR':2.368,'PRO':1.468,'HYP':1.468,'HIS':2.025,'HIP':2.025,'HID':2.025,'HIE':2.025,
    'GLU':1.862,'GLH':1.862,'GLN':1.932,'ASP':1.587,'ASH':1.587,'ASN':1.655,'LYS':2.258,'ARG':2.560,
    }

def _cal_sasa(obj:str):
    cmd.set('dot_solvent',1)
    cmd.set('dot_density',3)
    cmd.flag('ignore','solvent')
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

def sasa_scale(resn:str)->float:
    return GeorgeDSASA_scale.get(resn,1.118)

def res_sasa(chain:str, resi:str,obj:str=''):
    sel=_to_sel(chain,resi,obj)
    bs=[]
    cmd.iterate(sel , "bs.append(b)", space={'bs': bs})
    return sum(bs)

def iterate_resi(chain:str,obj:str='',only_canonical:bool=True)->str:
    if obj:
        obj_=f' and {obj}'
    else:
        obj_=''
    if only_canonical:
        o=[]
        cmd.iterate(
            f'name CA and (chain {chain}) {obj_}',
            'o.append(str(index))',
            space={'o':o}
            )
        return '( index '+ ','.join(o)+ ' )'
    else:
        o={}
        cmd.iterate(
            f'(chain {chain}) {obj_}',
            'o[resi]=str(index)',
            space={'o':o}
            )
        return '( index '+ ','.join(o.values())+ ' )'
    
def rSASA(obj:str,chain:str):
    _cal_sasa(obj)
    residues = {}
    sel=' or '.join([iterate_resi(c_,obj,only_canonical=False) for c_ in chain.split(',')])
    if 'or' in sel:
        sel=f' ( {sel} ) '
    cmd.iterate(f'{sel} and {obj}',#f'name ca  and (chain {chain}) and {obj}', 
        'residues.update({(chain, resi): res_sasa(chain, resi, obj)/sasa_scale(resn)}) ', # resn3_to_1(resn),
        space={"residues":residues,"resn3_to_1":resn3_to_1,'res_sasa':res_sasa,"sasa_scale":sasa_scale,"obj":obj})
    return residues

def cal_rSASA(pdb_file:str,chain:str,design_id:Optional[str]=None)->np.ndarray[float]:
    '''
    light-weighted function.
    calculate rSASA directly from file.
    '''
    if design_id is None:
        design_id=Path(pdb_file).stem
    cmd.load(pdb_file,design_id)
    cmd.h_add(design_id)
    ret=rSASA(design_id,chain)
    cmd.delete(design_id)
    return np.array([i/100 for i in ret.values()])

### selection ###
def resn3_to_1(resn):
    return protein_letters_3to1.get(resn.capitalize(), 'X')

def _to_sel(chain:str, resi:str,obj:str=''):
    sel = f'chain {chain} and resi {resi}'
    if obj:
        sel += f' and {obj}'
    return sel

### target seg selection ###
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
        res.setdefault(h[0],[]).append(h[1]) #int(h[1])
    return res

def _sort_resi_list(resi_list:List[str]):
    def key(s):
        if s[-1].isalpha():  
            return int(s[:-1]), s[-1]
        return int(s), '' 
    return sorted(resi_list, key=key)

def hotspots_to_seg_continue(obj:str,hotspots:List[Tuple[str,str]],opt_obj:str='continue_seg',pad=25):
    def _tmp(s:str):
        if s[-1].isalpha():  
            return int(s[:-1])
        return int(s)
    res=_reduce_hotspot_list(hotspots)
    res_be=[]
    for k,v in res.items():
        v=_sort_resi_list(v)
        min_v,max_v=_tmp(v[0]),_tmp(v[-1])
        res_be.append(f'(chain {k} and resi {max(min_v-pad,1)}-{max_v+pad})')
    cmd.copy_to(opt_obj,f'{obj} and ( {" or ".join(res_be) } )')

def hotspots_to_seg_surf(obj:str,hotspots:List[Tuple[str,str]],opt_obj:str='surf',vicinity=10):
    res=_reduce_hotspot_list(hotspots)
    res_sel=[]
    for k,v in res.items():
        res_sel.append(f'(chain {k} and resi {",".join([str(i) for i in v])})')
    cmd.select('hotspots',f'{obj} and ( {" or ".join(res_sel) } )')
    cmd.create(opt_obj,f'byres ( ({obj} and chain {",".join(res.keys())}) within {vicinity} of hotspots)')

### RMSD calculation ###
def partial_align(mobile:str,mobile_sel:str,target:str,
    mobile_rms_sel:str|None=None, target_sel:str|None=None,target_rms_sel:str|None=None):
    if target_sel is None:
        target_sel=mobile_sel
    if mobile_rms_sel is None:
        mobile_rms_sel=f'not ({mobile_sel})'
    if target_rms_sel is None:
        target_rms_sel=mobile_rms_sel
    
    cmd.create(f'{mobile}-aln',f'{mobile} and ({mobile_sel})')
    cmd.create(f'{target}-aln',f'{target} and ({target_sel})')
    ret1=cmd.align(f'{mobile}-aln',f'{target}-aln')
    cmd.align(f'{mobile} and ({mobile_sel})', f'{mobile}-aln') ## f'{mobile} and ({mobile_sel})',mobile_sel
    cmd.delete(f'{mobile}-aln')
    cmd.delete(f'{target}-aln')

    ############# debug
    # print("objects:", cmd.get_names("objects"))
    # print("mobile sel:", f"{mobile} and ({mobile_rms_sel})")
    # print("target sel:", f"{target} and ({target_rms_sel})")
    # print("mobile atoms:", cmd.count_atoms(f"{mobile} and ({mobile_rms_sel})"))
    # print("target atoms:", cmd.count_atoms(f"{target} and ({target_rms_sel})"))
    # breakpoint()
    ############# debug

    ret2=cmd.align(f'{mobile} and ({mobile_rms_sel})',f'{target} and ({target_rms_sel})' ,cycles=0,transform=0)[0]
    return {'align_rmsd':ret1[0],'obj_rmsd':ret2}


def sort_distance_to_hotspots(obj:str,hotspot_list:list)->list:
    '''
    obj: curated pymol object (e.g. no water/ions, no alt)
    hotspot_list: [('A',12),('A',17),('B',5),...]

    return: same format as hotspot_list
    '''
    res=_reduce_hotspot_list(hotspot_list)
    res_sel=[]
    for k,v in res.items():
        res_sel.append(f'(chain {k} and resi {",".join([str(i) for i in v])})')
    cmd.select('hotspots',f'{obj} and ( {" or ".join(res_sel) } )')
    ref_coord=cmd.get_coords('hotspots and name CA')

    o=[]
    cmd.iterate_state(0,
        f'({obj} and chain {",".join(res.keys())}) and (not hotspots) and name CA',
        'o.append([(chain,resi),[x,y,z]])',space={'o':o}
        )
    cmd.delete('hotspots')
    other_coord=np.array([i[1] for i in o])
    other_res=[i[0] for i in o]
    dis=np.sqrt(np.sum((other_coord[:,None,:]-ref_coord[None,:,:])**2,axis=2))
    order = np.argsort(dis.min(axis=1))
    other_res_sorted = [other_res[i] for i in order]
    return other_res_sorted

def top_k_epitope(obj:str,hotspot_list:list,other_res_sorted_list:list,k=100):
    more_needed=k - len(hotspot_list)
    remaining=len(other_res_sorted_list)

    if more_needed<0 or more_needed>remaining:
        raise ValueError
    else:
        sum_list = hotspot_list+other_res_sorted_list[:more_needed]

    res=_reduce_hotspot_list(sum_list)
    res_sel=[]
    for k_,v in res.items():
        res_sel.append(f'(chain {k_} and resi {",".join([str(i) for i in v])})')
    cmd.select(f'{obj}_top{k}',f'{obj} and ( {" or ".join(res_sel) } )')
    return sum_list