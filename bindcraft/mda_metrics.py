from ._import import *
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis.base import AnalysisFromFunction

# import pandas as pd
# import numpy as np
# from typing import List,Dict

def _ires_to_sel(inter_res:str):
    if inter_res:
        resnums=' '.join([i[1:] for i in inter_res.split(',')])
        b_ires=f'( segid B and resnum {resnums} and not backbone)'
    # return b_ires
        ret=f'(segid A or {b_ires})'
    else:
        ret='(segid A)'
    return ret

def hbond(pdbfile:str,inter_res:str)->Dict[str,pd.DataFrame]:
    '''
    hard-coded for Target A + Binder B
    inter_res: from metrics, "B1,B5,..."
    '''
    sel=_ires_to_sel(inter_res)
    u=mda.Universe(pdbfile,dt=1.)

    hbond_analysis = HydrogenBondAnalysis(
            universe=u, 
            donors_sel=f"{sel} and name O* N*",
            hydrogens_sel=f"{sel} and name *H*",
            acceptors_sel = f"{sel} and name *O* *N*",
            d_a_cutoff=3.5, 
            d_h_a_angle_cutoff=100
            # between=["segid A","segid B"], 
            
            )

    hbond_analysis.run()

    hbonds=hbond_analysis.results.hbonds
    dor_=u.atoms[hbonds[:,1].astype(int)]
    acc_=u.atoms[hbonds[:,3].astype(int)]

    o={}
    # ab,bb=[],[]
    ab_hb_pairs,bb_hb_pairs=[],[]
    for p_, (d,r) in enumerate(zip(dor_,acc_)):
        pair=(d.segid,r.segid)
        if pair in [('A','B'),('B','A')]: #,('B','B')
            # ab.append(p_)
            ab_hb_pairs.append([d.resname, d.resid, d.segid,
                                r.resname, r.resid, r.segid,
                                hbonds[p_,4],hbonds[p_,5]
                                ])
        elif pair==('B','B'):
            bb_hb_pairs.append([d.resname, d.resid, d.segid,
                                r.resname, r.resid, r.segid,
                                hbonds[p_,4],hbonds[p_,5]
                                ])

        o['hb_AB']=pd.DataFrame(ab_hb_pairs, columns=["D_Res", "D_ID", "D_Ch", "A_Res", "A_ID", "A_Ch", "Dis","Ang"])
        o['hb_BB']=pd.DataFrame(bb_hb_pairs, columns=["D_Res", "D_ID", "D_Ch", "A_Res", "A_ID", "A_Ch", "Dis","Ang"])
    return o

def salt_bridge(pdbfile:str,inter_res:str)->Dict[str,pd.DataFrame]:
    distance_cutoff=4.0
    u = mda.Universe(pdbfile)
    sel=_ires_to_sel(inter_res)
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
        res1 = acids[i].resname, acids[i].resid, acids[i].segid
        res2 = bases[j].resname, bases[j].resid, bases[j].segid
        if (acids[i].segid,bases[j].segid) in [('A','B'),('B','A')]:
            salt_bridge_pairs.append([*res1, *res2, dist_matrix[i, j]])
        elif (acids[i].segid,bases[j].segid)==('B','B'):
            salt_bridge_pairs.append([*res1, *res2, dist_matrix[i, j]])
        
    df_salt_bridges = pd.DataFrame(salt_bridge_pairs, columns=["A_Res", "A_ID", "A_Ch", "B_Res", "B_ID", "B_Ch", "Dis"])
    bb_df_salt_bridges = pd.DataFrame(bb_salt_bridge_pairs, columns=["A_Res", "A_ID", "A_Ch", "B_Res", "B_ID", "B_Ch", "Dis"])
    return {'salt_AB':df_salt_bridges,'salt_BB':bb_df_salt_bridges}

def ds_bond(pdbfile:str,chain:str='B'):
    raise NotImplementedError
    distance_cutoff=3.0
    u = mda.Universe(pdbfile)
    sgs=u.select_atoms(f"segid {chain} and name SG")
    dist_matrix = distance_array(sgs.positions, sgs.positions)
    

def pi_stacking(pdbfile:str,inter_res:str)->Dict[str,pd.DataFrame]:
    raise NotImplementedError

def _unique_B(df:pd.DataFrame,prefix:str):
    return df[df[f'{prefix}_Ch']=='B'][f'{prefix}_ID'].unique().tolist()

def _involved_B(df,ps:List[str]):
    o=[]
    for p in ps:
        o+=_unique_B(df,p)
    return set(o)

def cal_ppi_interacts(pdbfile,inter_res):
    '''
    warning: use relaxed pdbfile in `{outdir}/Trajectory/Relaxed`
    '''
    ppi_interacts=hbond(pdbfile,inter_res)
    ppi_interacts.update(salt_bridge(pdbfile,inter_res))
    b_inter_res={'all':set()}
    for k,ps in zip(['hb_AB','hb_BB','salt_AB','salt_BB'],['AD','AD','AB','AB']):
        b_inter_res[k]=_involved_B(ppi_interacts[k],ps) 
        b_inter_res['all']=b_inter_res['all']|b_inter_res[k]
    return ppi_interacts,b_inter_res

def radgyr(atomgroup, masses, total_mass=None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates-center_of_mass)**2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
    sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
    sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses*sq_rs, axis=1)/total_mass
    # square root and return
    return np.sqrt(rog_sq)

def cal_rog(pdbfile:str):
    u=mda.Universe(pdbfile,dt=1)
    
    binder=u.select_atoms('segid B')
    rog = AnalysisFromFunction(radgyr, u.trajectory,
                            binder, binder.masses,
                            total_mass=np.sum(binder.masses))
    rog.run()
    return rog.results['timeseries'][0,0]