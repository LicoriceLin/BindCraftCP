import os
from pathlib import Path
import pyrosetta as pr
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.pose import correctly_add_cutpoint_variants
import math
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
# from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, Selection, Polypeptide, PDBIO, Select, Chain, Superimposer
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1

from typing import Tuple

from ..relax import _pose_chain_length, init_pr
from ...utils.design_record import DesignBatch,DesignRecord
from .basescorer import (
    BaseScorer,DesignRecord,DesignBatch,
    GlobalSettings,NEST_SEP)

def calculate_clash_score(pdb_file:str, threshold=2.4, only_ca=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms = []
    atom_info = []  # Detailed atom info for debugging and processing

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':  # Skip hydrogen atoms
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes, only consider clashes between different chains
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)

def hotspot_residues(trajectory_pdb:str, binder_chain="B", atom_distance_cutoff=4.0, target_chain='A'):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0][target_chain], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in protein_letters_3to1:
            aa_single_letter = protein_letters_3to1[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

def score_interface(pdb_file:str, binder_chain="B",target_chain:str='A',cyclize_peptide:bool=False):
    pose = pr.pose_from_pdb(pdb_file)
    if cyclize_peptide:
        # pose.conformation().chains_from_termini()
        prev=1
        chain_lengths=_pose_chain_length(pose)
        for k,v in chain_lengths.items():
            if k!=binder_chain:
                prev+=v
            else:
                b=prev
                e=prev+v-1
                break
        mover=XmlObjects.static_get_mover(
            f'''<PeptideCyclizeMover name="close" >
            <Torsion res1="{e}" res2="{e}" res3="{b}" res4="{b}" atom1="CA" atom2="C" atom3="N" atom4="CA" cst_func="CIRCULARHARMONIC 3.141592654 0.005" />
            <Angle res1="{e}" atom1="CA" res_center="{e}" atom_center="C" res2="{b}" atom2="N" cst_func="HARMONIC 2.01000000 0.01" />
            <Angle res1="{e}" atom1="C" res_center="{b}" atom_center="N" res2="{b}" atom2="CA" cst_func="HARMONIC 2.14675498 0.01" />
            <Distance res1="{e}" res2="{b}" atom1="C" atom2="N" cst_func="HARMONIC 1.32865 0.01" />
            <Bond res1="{e}" res2="{b}" atom1="C" atom2="N" add_termini="true" />
        </PeptideCyclizeMover>'''
        )
        mover.apply(pose)
    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface(f"{target_chain}_{binder_chain}")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # Initialize dictionary with all amino acids
    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    # Initialize list to store PDB residue IDs at the interface
    interface_residues_set = hotspot_residues(pdb_file, binder_chain)
    interface_residues_pdb_ids = []
    
    # Iterate over the interface residues
    for pdb_res_num, aa_type in interface_residues_set.items():
        # Increase the count for this amino acid type
        interface_AA[aa_type] += 1

        # Append the binder_chain and the PDB residue number to the list
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # count interface residues
    interface_nres = len(interface_residues_pdb_ids)

    # Convert the list into a comma-separated string
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)

    # Calculate the percentage of hydrophobic residues at the interface of the binder
    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0

    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value # shape complementarity
    interface_interface_hbonds = interfacescore.interface_hbonds # number of interface H-bonds
    interface_dG = iam.get_interface_dG() # interface dG
    interface_dSASA = iam.get_interface_delta_sasa() # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat() # interface pack stat score
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100 # ratio of dG/dSASA (normalised energy for interface area size)
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100 # Hbonds per interface size percentage
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100 # Unsaturated H-bonds per percentage
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # calculate binder energy score
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # calculate binder SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # calculate surface hydrophobicity
    # if cyclize_peptide:
    #     pose.conformation().chains_from_termini()
    binder_pose = {pose.pdb_info().chain(pose.conformation().chain_begin(i)): p for i, p in zip(range(1, pose.num_chains()+1), pose.split_by_chain())}[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0 
    
    # count apolar and aromatic residues at the surface
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = binder_pose.residue(i)

            # count apolar and aromatic residues as hydrophobic
            if res.is_apolar() == True or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count/total_count

    # output interface score array and amino acid counts at the interface
    interface_scores = {
    'binder_score': binder_score,
    'surface_hydrophobicity': surface_hydrophobicity,
    'interface_sc': interface_sc,
    'interface_packstat': interface_packstat,
    'interface_dG': interface_dG,
    'interface_dSASA': interface_dSASA,
    'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
    'interface_fraction': interface_binder_fraction,
    'interface_hydrophobicity': interface_hydrophobicity,
    'interface_nres': interface_nres,
    'interface_interface_hbonds': interface_interface_hbonds,
    'interface_hbond_percentage': interface_hbond_percentage,
    'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
    'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    # round to two decimal places
    interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items()}
    return interface_scores, interface_AA, interface_residues_pdb_ids_str

def fetch_bfactor(pdb_file:str, chain_id:str="B"):
    '''
    used for plddt fetch
    '''
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model in the structure
    chain = model[chain_id]
    b_list=[]
    for residue in chain:
        b_list.append(round(residue.child_list[0].bfactor,2))
    return b_list

_ss_code_map={
    'H':'HGI',
    'E':'E',
    'C':'BTS-'
    }
_ss_code_map_r={c:k for k,v in _ss_code_map.items() for c in v}

def cal_dssp(pdb_file:str,chain_id="B",dssp_path:str='bindcraft_plus/steps/dssp'):
    '''
    Helix/H: H(alpha-helix), G(3-helix), I(5-helix); 
    Sheet/E: E(extended strand); 
    Loop/C: B(isolated beta-bridge), T(H-bond turn), S(bend),-(None).
    return: ss codes in 8-codes/3-codes alphabet.
    '''
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model in the structure
    chain = model[chain_id]
    dssp = DSSP(model, pdb_file, dssp=dssp_path)

    ss_list=[]
    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_list.append(ss)
    ss_3code_list=[_ss_code_map_r.get(i,'C') for i in ss_list]
    return ss_list,ss_3code_list

def cal_aux_scores(record:DesignRecord,
    pdb_to_take:dict={'pdb_key':'refold:best:relax','binder_chain':'B'},
    # ligand_chain:str='B',
    metrics_prefix:str='',
    cyclize_peptide:bool=False,
    ):
    '''
    Cal essential auxiliary scores from BindCraft. 
    Including: 
        secondary structures; crude hydrophobicity; n_clashes;  
        rosetta: energy scores; shape compatibility; h-bond; 
    Note: 
        derived scores, such as ss_pLDDT, i_pLDDT, secondary_structure%, dG/dSASA are not calculated here. 
        Simply calculate on-the-fly during filteration. 
    '''
    pdb_file=record.pdb_files[pdb_to_take['pdb_key']]
    ligand_chain=pdb_to_take['binder_chain']
    (record.ana_tracks[f'{metrics_prefix}SS8'],record.ana_tracks[f'{metrics_prefix}SS3']
        )=cal_dssp(pdb_file,ligand_chain)
    record.ana_tracks[f'{metrics_prefix}pLDDT']=fetch_bfactor(pdb_file,ligand_chain)
    record.set_metrics(f'{metrics_prefix}aux:relaxed_clashes',calculate_clash_score(pdb_file) )
    interface_score,_1,_2=score_interface(pdb_file, ligand_chain,cyclize_peptide=cyclize_peptide)
    record.set_metrics(f'{metrics_prefix}rst:binder_score',interface_score['binder_score'])
    record.set_metrics(f'{metrics_prefix}rst:packstat',interface_score['interface_packstat'])
    record.set_metrics(f'{metrics_prefix}rst:shape_complementarity',interface_score['interface_sc'])
    record.set_metrics(f'{metrics_prefix}rst:dG',interface_score['interface_dG'])
    record.set_metrics(f'{metrics_prefix}rst:dSASA',interface_score['interface_dSASA'])
    record.set_metrics(f'{metrics_prefix}rst:hbond',interface_score['interface_interface_hbonds'])
    record.set_metrics(f'{metrics_prefix}rst:unsat_hbond',int(interface_score['interface_delta_unsat_hbonds']))
    record.set_metrics(f'{metrics_prefix}aux:surf_hydro',float(interface_score['surface_hydrophobicity']))
    record.set_metrics(f'{metrics_prefix}aux:interface_hydro',float(interface_score['interface_hydrophobicity']))
    return record


class AnnotBCAux(BaseScorer):
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings,score_func=cal_aux_scores)

    @property
    def name(self)->str:
        return 'aux-scores'

    @property
    def _default_pdb_input_key(self):
        return {'pdb_key':'refold:best:relax','binder_chain':'B'}
    
    @property
    def _default_metrics_prefix(self):
        return ''

    def _init_params(self):
        self.params=dict(
            pdb_to_take=self.pdb_to_take,
            # ligand_chain=self.pdb_to_take['binder_chain'],
            metrics_prefix=self.metrics_prefix,
            cyclize_peptide=self.settings.adv.setdefault('cyclize_peptide',False),
            )
        
    @property
    def params_to_take(self)->Tuple[str,...]:
        ret=[f'{self.name}-prefix',f'{self.name}-pdb-input','cyclize_peptide']
        return tuple(ret)

    @property
    def metrics_to_add(self)->Tuple[str,...]:
        metrics_prefix= self.metrics_prefix
        return tuple([
            f'{metrics_prefix}aux:relaxed_clashes',f'{metrics_prefix}rst:binder_score',
            f'{metrics_prefix}rst:packstat',f'{metrics_prefix}rst:shape_complementarity',
            f'{metrics_prefix}rst:dG',f'{metrics_prefix}rst:dSASA',
            f'{metrics_prefix}rst:hbond',f'{metrics_prefix}rst:unsat_hbond',
            f'{metrics_prefix}aux:surf_hydro',f'{metrics_prefix}aux:interface_hydro'
            ])
    
    @property
    def track_to_add(self)->Tuple[str,...]:
        metrics_prefix= self.metrics_prefix
        return tuple([f'{metrics_prefix}SS8',f'{metrics_prefix}SS3',f'{metrics_prefix}pLDDT']) 
