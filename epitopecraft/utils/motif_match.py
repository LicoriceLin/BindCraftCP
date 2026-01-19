from ..steps.scorer.pymol_utils import *

from functools import partial
from typing import Callable,Any
import numpy as np
# cmd.delete('all')

def load_ClpP_motif():
    cmd.load('input/Jul_benchmark/motif/ClpP-IGF-motif-tmp1_model_0.cif','control')
    cmd.select('IGF','control and chain C and resi 2-4')
    ref_coord=cmd.get_coords('IGF and not backbone and not element H')
    partial_align_fn=partial(partial_align, target='control', target_sel='chain A,B',target_rms_sel='chain C and resi 2-4')
    return ref_coord,partial_align_fn

def load_WDR5_motif():
    cmd.load('input/Jul_benchmark/2g99.pdb','control')
    cmd.select('conserved_N','control and chain D and resi 2 and element N')
    ref_coord=cmd.get_coords('conserved_N and not backbone and not element H')
    partial_align_fn=partial(partial_align, target='control', target_sel='chain A',target_rms_sel='chain C and resi 2-4')
    return ref_coord,partial_align_fn


def match_count(
    pdb_file:str,mobile_sel:str,mobile_rms_sel:str,
    ref_coord:np.ndarray,partial_align_fn:Callable[[str,str],Any],
    match_threshold:float=1.5,obj_name:str|None=None,):
    '''
    ref_coord & partial_align_fn should come from `load_xxxx`
    '''
    if obj_name is None:
        obj_name=Path(pdb_file).stem
    cmd.load(pdb_file,obj_name)
    partial_align_fn(mobile=obj_name,mobile_sel=mobile_sel,mobile_rms_sel=mobile_rms_sel)
    design_coord=cmd.get_coords(
        f'{obj_name} and ({mobile_rms_sel}) and (not element H)')
    match_dis=np.linalg.norm(ref_coord[:,None,:]- design_coord[None,:,:],axis=-1).min(axis=1)
    cmd.delete(obj_name)
    return (match_dis<match_threshold).sum().item()
    # return (match_dis<1.5).sum().item()