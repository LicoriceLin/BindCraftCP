import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.analysis.base import AnalysisFromFunction
from typing import List,Dict,Tuple
import pandas as pd
import numpy as np
from .basescorer import (
    BaseScorer,DesignRecord,DesignBatch,
    GlobalSettings,NEST_SEP)


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
            donors_sel=f"{sel} and name *O* *N*",
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
    for p_, (d,r) in enumerate(zip(dor_,acc_)):
        pair=(d.segid,r.segid)
        if pair in _ligand_target_chain_pair(target_chain,ligand_chain):
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
    '''
    Acidic/basic atom pairs within a distance range.
    '''
    sel=_ppi_track_to_ires_sel(ppi_track,ligand_chain,target_chain)
    u=mda.Universe(pdbfile,dt=1.)
    # Acid
    acidic_res = ["ASP", "GLU"]
    acidic_atoms = ["OD1", "OD2", "OE1", "OE2"]
    acids = u.select_atoms(f"{sel}  and resname {' '.join(acidic_res)} and name {' '.join(acidic_atoms)}")
    # Alkaline
    basic_res = ["ARG", "LYS", "HIS"]
    basic_atoms = ["NH1", "NH2", "NZ", "ND1", "NE2"]
    bases = u.select_atoms(f"{sel}  and resname {' '.join(basic_res)} and name {' '.join(basic_atoms)}")

    # distance
    dist_matrix = distance_array(acids.positions, bases.positions)
    contact_pairs = np.argwhere(dist_matrix < distance_cutoff)

    # results
    salt_bridge_pairs,bb_salt_bridge_pairs = [],[]
    for i, j in contact_pairs:
        res1 = acids[i].resname, acids[i].resid, acids[i].resnum, acids[i].segid
        res2 = bases[j].resname, bases[j].resid, bases[j].resnum, bases[j].segid
        if (acids[i].segid,bases[j].segid) in _ligand_target_chain_pair(target_chain,ligand_chain):
            salt_bridge_pairs.append([*res1, *res2, dist_matrix[i, j]])
        elif (acids[i].segid,bases[j].segid)==(ligand_chain,ligand_chain):
            salt_bridge_pairs.append([*res1, *res2, dist_matrix[i, j]])
        
    df_salt_bridges = pd.DataFrame(salt_bridge_pairs, columns=["A_Res", "A_ID", "A_num", "A_Ch", 
                "B_Res", "B_ID", "B_num", "B_Ch","Dis"])
    bb_df_salt_bridges = pd.DataFrame(bb_salt_bridge_pairs, columns=["A_Res", "A_ID", "A_num", "A_Ch", 
                "B_Res", "B_ID", "B_num", "B_Ch","Dis"])
    return {'salt_inter':df_salt_bridges,'salt_intra':bb_df_salt_bridges}


def _ppi_track_to_ires_sel(ppi_track:List[int],ligand_chain:str='B',target_chain:str='A'):
    target_chain=target_chain.replace(',',' ')
    inter_res=np.where(np.array(ppi_track))[0]+1
    if len(inter_res)>0:
        resnums=' '.join([str(i) for i in inter_res])
        b_ires=f'( segid {ligand_chain} and resnum {resnums} and not backbone)'
        inter_res=f'(segid {target_chain} or {b_ires})'
    else:
        inter_res=f'(segid {target_chain})'
    return inter_res


def _ligand_target_chain_pair(target_chain:str,ligand_chain:str):
    '''
    target could sometimes be multi-chain e.g. "A,B" vs "C"
    use this util to get chain pairs:
        [(A,C),(B,C),(C,A),(C,B)]
    '''
    cp=[]
    for tc in target_chain.split(','):
        cp.extend([(tc,ligand_chain),(ligand_chain,tc)])
    return cp

aatype_dict={
    "+":"RHK",
    "-":"DE",
    "*":"CGP",
    "H":"STNQ",
    "A":"AVILM",
    "F":"FYW"
    }

def annot_aatype(seq:str,type_dict:Dict[str,str]=aatype_dict):
    idx={}
    for k,v in type_dict.items():
        for i in v:
            idx[i]=k
    return [idx.get(i,'*') for i in seq]


def annot_polar_occupy(record:DesignRecord,pdb_to_take:str,
    ligand_chain:str='B',target_chain:str='A',
    d_a_cutoff:float=3.5,d_h_a_angle_cutoff:float=100,sb_distance_cutoff:float=4.0,
    ppi_track_prefix:str='',metrics_prefix:str='',ret_misc:bool=False,
    ):
    ppi_track=record.ana_tracks[ppi_track_prefix+'ppi']
    pdbfile=record.pdb_files[pdb_to_take]
    salt_bridge_ret=salt_bridge(pdbfile,ppi_track,
        ligand_chain,target_chain,sb_distance_cutoff)
    inter_salt=salt_bridge_ret['salt_inter']
    salt_id=np.array(list(set(inter_salt[inter_salt['A_Ch']==ligand_chain]["A_num"].unique())
        |set(inter_salt[inter_salt['B_Ch']==ligand_chain]["B_num"].unique())))
    
    salt_track=np.zeros((len(record.sequence),)).astype(int)
    if len(salt_id)>0:
        salt_track[salt_id-1]=1
    hbond_ret=hbond(pdbfile,ppi_track,
        ligand_chain,target_chain,d_a_cutoff,d_h_a_angle_cutoff)
    inter_hb=hbond_ret['hb_inter']
    hb_id=np.array(list(set(inter_hb[inter_hb['D_Ch']==ligand_chain]["D_num"].unique())
        |set(inter_hb[inter_hb['A_Ch']==ligand_chain]["A_num"].unique())))
    hb_track=np.zeros((len(record.sequence),)).astype(int)
    if len(hb_id)>0:
        hb_track[hb_id-1]=1
    aatype_track=np.array(annot_aatype(record.sequence))
    ppi_track_np=np.array(ppi_track)
    is_polar=np.vectorize(lambda x:x in '+-H')(aatype_track)
    unsat_ppi_polar= (~(hb_track | salt_track)) & ppi_track_np & is_polar
    record.ana_tracks[f'{metrics_prefix}h-bond']=hb_track.tolist()
    record.ana_tracks[f'{metrics_prefix}salt-bridge']=salt_track.tolist()
    record.ana_tracks[f'{metrics_prefix}unsat_ppi_polar']=unsat_ppi_polar.tolist()
    record.ana_tracks[f'{metrics_prefix}aatype']=aatype_track.tolist()
    if ret_misc:
        record.ana_tracks['misc:polar_occ']={
            'sb':salt_bridge_ret,
            'hb':hbond_ret}
    return record

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

# def cal_rog(pdbfile:str,ligand_chain:str='B'):
#     u=mda.Universe(pdbfile,dt=1)
#     binder=u.select_atoms(f'segid {ligand_chain}')
#     rog = AnalysisFromFunction(radgyr, u.trajectory,
#                             binder, binder.masses,
#                             total_mass=np.sum(binder.masses))
#     rog.run()
#     return rog.results['timeseries'][0,0]

def cal_gyration_metrics(pdbfile:str,ligand_chain:str='B'):
    u=mda.Universe(pdbfile,dt=1)
    binder=u.select_atoms(f'segid {ligand_chain} and name CA')
    metrics = gyration_metrics(binder, mass_weighted=False)  
    return metrics

def gyration_metrics(atomgroup, mass_weighted=True):
    """
    Compute Rg and shape anisotropy metrics from an MDAnalysis AtomGroup.
    Uses the gyration tensor (second moment / covariance-like) around COM/COG.

    Returns a dict with:
      - Rg
      - lambdas (sorted desc)
      - kappa2 (relative shape anisotropy, 0 sphere -> 1 rod)
      - elongation (sqrt(l1/l3))
      - asphericity (b, l1 - 0.5 * (l2 + l3))
      - acylindricity (c, l2 - l3)
    """
    pos = atomgroup.positions.astype(np.float64)  # (N, 3)

    if mass_weighted:
        m = atomgroup.masses.astype(np.float64)
        msum = m.sum()
        if msum <= 0:
            raise ValueError("Total mass is non-positive; cannot mass-weight.")
        com = (pos * m[:, None]).sum(axis=0) / msum
        x = pos - com
        # gyration tensor S = (1/M) * sum_i m_i * x_i x_i^T
        S = (x.T * m) @ x / msum
    else:
        cog = pos.mean(axis=0)
        x = pos - cog
        # unweighted gyration tensor
        S = (x.T @ x) / x.shape[0]

    # Eigenvalues of gyration tensor (may have tiny negative due to numerics)
    evals = np.linalg.eigvalsh(S)
    evals = np.clip(evals, 0.0, None)  # safeguard
    # Sort descending: l1 >= l2 >= l3
    l1, l2, l3 = np.sort(evals)[::-1]

    Rg2 = l1 + l2 + l3
    Rg = np.sqrt(Rg2)

    # Relative shape anisotropy: κ^2 = 1 - 3*(l1l2+l2l3+l3l1)/(l1+l2+l3)^2
    denom = (Rg2 ** 2) if Rg2 > 0 else np.nan
    kappa2 = 1.0 - 3.0 * (l1*l2 + l2*l3 + l3*l1) / denom if denom and np.isfinite(denom) else np.nan

    # Elongation (rod-like if large)
    elongation = np.sqrt(l1 / l3) if l3 > 0 else np.inf

    # Asphericity and acylindricity (common polymer shape descriptors)
    asphericity = l1 - 0.5 * (l2 + l3)          # b
    acylindricity = l2 - l3                      # c

    return {
        "Rg": Rg,
        "lambdas": (l1, l2, l3),
        "kappa2": kappa2,
        "elongation": elongation,
        "asphericity": asphericity,
        "acylindricity": acylindricity,
    }

def annot_gyration(record:DesignRecord,pdb_to_take:str,
    ligand_chain:str='B',metrics_prefix:str='aux:'):
    pdbfile=record.pdb_files[pdb_to_take]
    gyr_metrics=cal_gyration_metrics(pdbfile,ligand_chain)
    record.set_metrics(f'{metrics_prefix}rog',float(gyr_metrics['Rg']))
    record.set_metrics(f'{metrics_prefix}kappa2',float(gyr_metrics['kappa2']))
    return record

class AnnotPolarOccupy(BaseScorer):
    '''
    Designed to work with relaxed conformation.
    '''
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings,score_func=annot_polar_occupy,decoupled=True)

    @property
    def name(self)->str:
        return 'polar-occupy-annot'
    
    @property
    def params_to_take(self)->Tuple[str,...]:
        ret=[f'{self.name}-prefix',f'{self.name}-pdb-input']
        return tuple(ret)
    
    def _init_params(self):
        '''
        '''
        self.params=dict(
            pdb_to_take=self.pdb_to_take,
            ligand_chain='B',
            target_chain='A',
            d_a_cutoff=3.5,
            d_h_a_angle_cutoff=100,
            sb_distance_cutoff=4.0,
            ppi_track_prefix='',
            metrics_prefix=self.metrics_prefix,
            ret_misc=False,
            )
    
    @property
    def track_to_add(self):
        return tuple( [f'{self.metrics_prefix}{i}' for i 
            in ['h-bond','salt-bridge','unsat_ppi_polar','aatype']])
    
    @property
    def metrics_to_add(self):
        return tuple([])


class AnnotGyration(BaseScorer):
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings,score_func=annot_gyration)

    @property
    def _default_metrics_prefix(self):
        return 'aux:'

    @property
    def name(self)->str:
        return 'gyration-annot'

    @property
    def params_to_take(self)->Tuple[str,...]:
        ret=[f'{self.name}-prefix',f'{self.name}-pdb-input']
        return tuple(ret)

    def _init_params(self):
        self.params=dict(
            pdb_to_take=self.pdb_to_take,
            ligand_chain='B',
            metrics_prefix=self.metrics_prefix,
            )

    @property
    def metrics_to_add(self):
        return tuple([f'{self.metrics_prefix}kappa2',f'{self.metrics_prefix}rog'])
    
    @property
    def _default_pdb_input_key(self):
        return 'refold:best'