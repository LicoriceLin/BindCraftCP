import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.distances import distance_array
# from MDAnalysis.analysis.base import AnalysisFromFunction
from typing import List,Dict
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
    resnums=' '.join([str(i) for i in inter_res])
    b_ires=f'( segid {ligand_chain} and resnum {resnums} and not backbone)'
    inter_res=f'(segid {target_chain} or {b_ires})'
    # print(inter_res)
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


def annot_aatype(seq:str,type_dict:dict={
    "+":"RHK",
    "-":"DE",
    "*":"CGP",
    "H":"STNQ",
    "A":"AVILM",
    "F":"FYW"
    }):
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

    
class AnnotPolarOccupy(BaseScorer):
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings,score_func=annot_polar_occupy)

    @property
    def name(self)->str:
        return 'polar-occupy-annot'
    
    def _init_params(self):
        '''
        not a standard building block in BindCraft
        minimal coupling with settings.
        specify params in `process_batch`/`config_params`
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
    