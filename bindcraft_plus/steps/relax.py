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
import tempfile
from .basestep import BaseStep,DesignBatch,DesignRecord,NEST_SEP
from collections import OrderedDict
from typing import Dict

def _pose_chain_length(pose):
    chain_lengths = OrderedDict()
    for i in range(1, pose.total_residue() + 1):
        chain = pose.pdb_info().chain(i)
        if chain not in chain_lengths:
            chain_lengths[chain] = 1
        else:
            chain_lengths[chain] += 1
    return chain_lengths

def pr_relax(pdb_file:str, cyclize_peptide:bool=False,cyclize_chain:str='B')->str:
    # Generate pose
    pose = pr.pose_from_pdb(pdb_file)
    if cyclize_peptide:
        # b,e=len(pose.split_by_chain()[1])+1,len(pose)
        prev=1
        chain_lengths=_pose_chain_length(pose)
        for k,v in chain_lengths.items():
            if k!='B':
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
    start_pose = pose.clone()

    ### Generate movemaps
    mmf = MoveMap()
    mmf.set_chi(True) # enable sidechain movement
    mmf.set_bb(True) # enable backbone movement, can be disabled to increase speed by 30% but makes metrics look worse on average
    mmf.set_jump(False) # disable whole chain movement

    # Run FastRelax
    fastrelax = FastRelax()
    scorefxn = pr.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mmf) # set MoveMap
    fastrelax.max_iter(200) # default iterations is 2500
    fastrelax.min_type("lbfgs_armijo_nonmonotone")
    fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.apply(pose)

    # Align relaxed structure to original trajectory
    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)

    # Copy B factors from start_pose to pose
    for resid in range(1, pose.total_residue() + 1):
        if pose.residue(resid).is_protein():
            # Get the B factor of the first heavy atom in the residue
            bfactor = start_pose.pdb_info().bfactor(resid, 1)
            for atom_id in range(1, pose.residue(resid).natoms() + 1):
                pose.pdb_info().bfactor(resid, atom_id, bfactor)

    # output relaxed and aligned PDB
    with tempfile.TemporaryDirectory() as tdir:
        pose.dump_pdb(f'{tdir}/relaxed.pdb')
        return open(f'{tdir}/relaxed.pdb','r').read()
    
def init_pr(dalphaball_path:str):
    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1')

class Relax(BaseStep):
    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        self.init_pr()

    @property
    def name(self)->str:
        return 'relax'
    
    def init_pr(self):
        dalphaball_path=self.settings.adv.get(
            'dalphaball_path','functions/DAlphaBall.gcc')
        if not dalphaball_path: 
            dalphaball_path='functions/DAlphaBall.gcc'
        init_pr(dalphaball_path)

    @property
    def pdb_to_add(self):
        return tuple([self.pdb_to_take['pdb_key']+':'+self.metrics_prefix.strip(NEST_SEP)])
    
    def process_record(self, input:DesignRecord)->DesignRecord:
        # if not self.check_processed(input):
        with self.record_time(input):
            input.pdb_strs[self.pdb_to_add[0]]=pr_relax(
                input.pdb_files[self.pdb_to_take['pdb_key']],
                cyclize_peptide=self.settings.adv.get('cyclize_peptide',False),
                cyclize_chain=self.pdb_to_take['binder_chain'])
        return input
    
    @property
    def pdb_to_take(self)->Dict[str,str]:
        '''
        pdb_to_take['pdb_key']: pdb key in records.
        pdb_to_take['binder_chain']: only used for cyclic relax. 
            chain to apply cyclization constraint on.
        '''
        return self._pdb_to_take

    def config_pdb_input_key(self, 
        pdb_to_take:str|None = None,binder_chain:str|None =None):

        if pdb_to_take is None:
            _pdb_to_take='halu'
        else:
            _pdb_to_take=pdb_to_take
    
        if binder_chain is None:
            if 'halu' in _pdb_to_take or 'refold' in _pdb_to_take:
                _binder_chain=self.settings.target_settings.new_binder_chain
            else:
                _binder_chain=self.settings.target_settings.full_binder_chain
        self._pdb_to_take={'pdb_key':_pdb_to_take,'binder_chain':_binder_chain}

    

    
