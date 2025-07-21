import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis.base import AnalysisFromFunction
from typing import List,Dict
import pandas as pd
import numpy as np
from .basescorer import BaseScorer,DesignRecord,DesignBatch

def _ppi_track_to_ires_sel(ppi_track:List[int],ligand_chain:str='B',target_chain:str='A'):
    target_chain=target_chain.replace(',',' ')
    inter_res=np.where(np.array(ppi_track))[0]+1
    resnums=' '.join([str(i) for i in inter_res])
    b_ires=f'( segid {ligand_chain} and resnum {resnums} and not backbone)'
    inter_res=f'(segid {target_chain} or {b_ires})'
    return inter_res

def hbond(pdbfile:str,ppi_track:List[int],
    ligand_chain:str='B',target_chain:str='A',
    d_a_cutoff:float=3.5,d_h_a_angle_cutoff:float=100)->Dict[str,pd.DataFrame]:
    '''
    Note: the pdbfile must contain H atoms (ideally by relaxation)
    hbond between sidechain of ligand's PPI residues & Targets.
    intermediate result to count H-Bond / Unsat H-Bond on PPI.
    '''
    sel=_ppi_track_to_ires_sel(ppi_track,ligand_chain,target_chain)
    u=mda.Universe(pdbfile,dt=1.)

    hbond_analysis = HydrogenBondAnalysis(
            universe=u, 
            donors_sel=f"{sel} and name O* N*",
            hydrogens_sel=f"{sel} and name *H*",
            acceptors_sel = f"{sel} and name *O* *N*",
            d_a_cutoff=d_a_cutoff, 
            d_h_a_angle_cutoff=d_h_a_angle_cutoff
            )
    hbond_analysis.run()

    hbonds=hbond_analysis.results.hbonds
    dor_=u.atoms[hbonds[:,1].astype(int)]
    acc_=u.atoms[hbonds[:,3].astype(int)]

    o={}
    ab_hb_pairs,bb_hb_pairs=[],[]
    cp=[]
    for tc in target_chain.split(','):
        cp.extend([(tc,ligand_chain),(ligand_chain,tc)])
    for p_, (d,r) in enumerate(zip(dor_,acc_)):
        pair=(d.segid,r.segid)
        if pair in cp:
            ab_hb_pairs.append([d.resname, d.resid, d.resnum, d.segid,
                r.resname, r.resid, r.resnum, r.segid,hbonds[p_,4],hbonds[p_,5]])
        elif pair==(ligand_chain,ligand_chain):
            bb_hb_pairs.append([d.resname, d.resid, d.resnum, d.segid,
                r.resname, r.resid, r.resnum, r.segid,hbonds[p_,4],hbonds[p_,5]])
        o['hb_inter']=pd.DataFrame(ab_hb_pairs,
            columns=["D_Res", "D_ID", "D_num", "D_Ch", 
                "A_Res", "A_ID", "A_num", "A_Ch", "Dis","Ang"])
        o['hb_intra']=pd.DataFrame(bb_hb_pairs,
            columns=["D_Res", "D_ID", "D_num", "D_Ch", 
                "A_Res", "A_ID","A_num", "A_Ch", "Dis","Ang"])
        
    return o


def salt_bridge(pdbfile:str,ppi_track:List[int],
    ligand_chain:str='B',target_chain:str='A',
    distance_cutoff:float=4.0)->Dict[str,pd.DataFrame]:
    
    sel=_ppi_track_to_ires_sel(ppi_track,ligand_chain,target_chain)
    u=mda.Universe(pdbfile,dt=1.)
    # Acid
    acidic_res = ["ASP", "GLU"]
    acidic_atoms = ["OD1", "OD2", "OE1", "OE2"]
    acids = u.select_atoms(f"( segid A or {sel} ) and resname {' '.join(acidic_res)} and name {' '.join(acidic_atoms)}")

    # Alkaline
    basic_res = ["ARG", "LYS", "HIS"]
    basic_atoms = ["NH1", "NH2", "NZ", "ND1", "NE2"]
    bases = u.select_atoms(f"(segid A or {sel} ) and resname {' '.join(basic_res)} and name {' '.join(basic_atoms)}")

    # distance
    dist_matrix = distance_array(acids.positions, bases.positions)
    contact_pairs = np.argwhere(dist_matrix < distance_cutoff)

    # results
    salt_bridge_pairs,bb_salt_bridge_pairs = [],[]
    for i, j in contact_pairs:
        res1 = acids[i].resname, acids[i].resid, acids[i].resnum, acids[i].segid
        res2 = bases[j].resname, bases[j].resid, bases[j].resnum, bases[j].segid
        if (acids[i].segid,bases[j].segid) in [('A','B'),('B','A')]:
            salt_bridge_pairs.append([*res1, *res2, dist_matrix[i, j]])
        elif (acids[i].segid,bases[j].segid)==('B','B'):
            salt_bridge_pairs.append([*res1, *res2, dist_matrix[i, j]])
        
    df_salt_bridges = pd.DataFrame(salt_bridge_pairs, columns=["A_Res", "A_ID", "A_Ch", "B_Res", "B_ID", "B_Ch", "Dis"])
    bb_df_salt_bridges = pd.DataFrame(bb_salt_bridge_pairs, columns=["A_Res", "A_ID", "A_Ch", "B_Res", "B_ID", "B_Ch", "Dis"])
    return {'salt_inter':df_salt_bridges,'salt_intra':bb_df_salt_bridges}

