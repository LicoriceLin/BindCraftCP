from functions import *
from ._import import *
from .bc_util import init_task
# import os, time, gc, io
# import contextlib
# import json
from datetime import datetime
from ipywidgets import HTML, VBox
from IPython.display import display

# import numpy as np
# from colabdesign.af.alphafold.common import residue_constants

# from typing import Dict,List,Tuple,Any
# aa_order = residue_constants.restype_order
# order_aa = {b:a for a,b in aa_order.items()}

class Design:
    '''
    Step 1 of original BindCraft.
    Sample Hallucination Trajectories, Genearate Backbones Candidates.
    '''
    def __init__(self,
        outdir:str,
        starting_pdb:str,
        chains:str,
        target_hotspot_residues:str='',
        binder_name:str|None=None,
        advanced_settings_overload:Dict[str,Any]={},
        base_advanced_settings_path:str='settings_advanced/mcmcsampling_multimer_cyc.json',
        filter_settings_path='settings_filters/no_filters.json',
        overwrite_outdir:bool=False,
        ):
        '''
        if `outdir` exists and ~`base_advanced_settings_path`:
            load outdir/"advanced_settings.json" instead
        
        '''
        self._outdir=outdir
        self.starting_pdb=starting_pdb
        self.chains=chains
        self.target_hotspot_residues=target_hotspot_residues

        
        if binder_name is None:
            self.binder_name=Path(starting_pdb).stem
        else:
            self.binder_name=binder_name
        
        _p=Path(outdir)
        if _p.exists():
            if overwrite_outdir:
                _p.rename(_p.with_stem(_p.stem+'_bk'))
                self.load_previous_target_settings=''
            else:
                _=_p/f'{self.binder_name}.json'
                if _.exists():
                    self.load_previous_target_settings=_
                else:
                    self.load_previous_target_settings=''
        else:
            self.load_previous_target_settings=''
        self.outdir=_p
        if base_advanced_settings_path:
            self.base_advanced_settings_path=base_advanced_settings_path
        else:
            _b=_p/'advanced_settings.json'
            assert _b.exists()
            self.base_advanced_settings_path=str(_b)
        self.filter_settings_path=filter_settings_path
        self.advanced_settings_overload=advanced_settings_overload
        self.overwrite_outdir=overwrite_outdir

    def init_task(self):
            (self.target_settings, self.advanced_settings, self.filters,
            self.settings_file,self.filters_file,self.advanced_file,
            self.design_models, self.prediction_models, self.multimer_validation,
            self.design_paths,self.trajectory_csv,self.mpnn_csv,self.final_csv,self.failure_csv,self.trajectory_dirs
            )=init_task(
            load_previous_target_settings=self.load_previous_target_settings,
            advanced_settings_path=self.base_advanced_settings_path,
            filter_settings_path=self.filter_settings_path,
            design_path=self._outdir,
            binder_name=self.binder_name,
            starting_pdb=self.starting_pdb,
            chains=self.chains,
            target_hotspot_residues=self.target_hotspot_residues,
            lengths = "10,20",number_of_final_designs = 1,
            )
            self.advanced_settings.update(self.advanced_settings_overload)

            with open(self.outdir/'advanced_settings.json','w') as f:
                json.dump(self.advanced_settings,f, indent=2)      
            if not self.advanced_settings['omit_AAs']:
                self.advanced_settings['omit_AAs']=None
            
    def sampling(self,prefix:str,seeds:Iterable[int],lengths:Iterable[str],helicity_values:Iterable[int]):
        '''
        please use int for `helicity_value` to generate comfortable `design_id`. They'll be divided by 10 during sampling. 
        '''
        seeds,lengths,helicity_values=list(seeds),list(lengths),list(helicity_values)
        lengths.sort()
        self.trajectory_metrics=[]
        if os.path.exists(self.outdir/'trajectory_stats.csv'):
            _=pd.read_csv(self.outdir/'trajectory_stats.csv')
            if len(_)>0:
                finished=_['Design'].to_list()
            else:
                finished=[]
        else:
            finished=[]
        skipped=self.outdir/'Trajectory/LowConfidence'
        if os.path.exists(skipped):
            finished.extend([i.stem for i in skipped.iterdir()])
        
        for length in lengths:
            af_model=None
            for seed in seeds:#[,11,18,33,62,71,49,81]: [1,8,43,4308,]
                for helicity_value in helicity_values:    
                    # for rep in range(1,3):
                    design_name=f'{prefix}_l{length}_h{helicity_value}_s{seed}'
                    if design_name in finished:
                        print(f'finished: {design_name}')
                    else:
                        print(f'running: {design_name}')
                        trajectory_data,af_model=bindcraft_hallucinate(
                            design_name=design_name,
                            length=length,
                            target_settings=self.target_settings,
                            advanced_settings=self.advanced_settings,
                            design_models=self.design_models,
                            design_paths=self.design_paths,
                            helicity_value=helicity_value/10,
                            seed=seed,
                            af_model=af_model
                            )
                        self.trajectory_metrics.append(trajectory_data)
                        af_model.restart()
        
    
def bindcraft_hallucinate(
    design_name:str,
    length:int,
    target_settings:Dict[str,str|int|List[int]],
    advanced_settings:BasicDict,
    design_models,
    design_paths:Dict[str,str]|None=None,
    helicity_value:float=0.,
    seed:int|None=None,
    binder_chain:str = "B",
    af_model:Optional[mk_afdesign_model]=None,
    settings_file='settings_file', 
    filters_file='filters_file', 
    advanced_file='advanced_file'
    ):
    '''
    to be curated.
    '''
    trajectory_start_time = time.time()
    trajectory_csv:str = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
    failure_csv:str = os.path.join(target_settings["design_path"], 'failure_csv.csv')
    if design_paths is None:
        design_paths=generate_directories(target_settings["design_path"])

    # starting_pdb=target_settings["starting_pdb"], 
    # chain=target_settings["chains"],
    # target_hotspot_residues=target_settings["target_hotspot_residues"], 


    trajectory=binder_hallucination(
        design_name=design_name, 
        starting_pdb=target_settings["starting_pdb"], 
        chain=target_settings["chains"],
        target_hotspot_residues=target_settings["target_hotspot_residues"], 
        length=length, 
        helicity_value=helicity_value,
        design_models=design_models, 
        advanced_settings=advanced_settings, 
        design_paths=design_paths, 
        failure_csv=failure_csv,
        seed=seed, 
        af_model=af_model
        )
    trajectory_time = time.time() - trajectory_start_time
    trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
    trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
    # starting binder sequence
    trajectory_sequence = trajectory.get_seq(get_best=True)[0]
    # return trajectory
    trajectory_metrics:dict = copy_dict(trajectory.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae
    trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

    if os.path.exists(trajectory_pdb):
        # time trajectory
        trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
        pr_relax(trajectory_pdb, trajectory_relaxed,cyclize_peptide=advanced_settings.get('cyclize_peptide',False))
        # Calculate clashes before and after relaxation
        num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
        num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

        # secondary structure content of starting trajectory binder and interface
        trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_i_plddt, trajectory_ss_plddt = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

        # analyze interface scores for relaxed af2 trajectory
        trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain,cyclize_peptide=advanced_settings.get('cyclize_peptide',False))



        # analyze sequence
        traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings)

        # target structure RMSD compared to input PDB
        trajectory_target_rmsd = unaligned_rmsd(target_settings["starting_pdb"], trajectory_pdb, target_settings["chains"], 'A')

        # save trajectory statistics into CSV
        trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], trajectory_sequence, trajectory_interface_residues,
                            trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'], trajectory_metrics['pae'], trajectory_metrics['i_pae'],
                            trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed, trajectory_interface_scores['binder_score'],
                            trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'], trajectory_interface_scores['interface_packstat'],
                            trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'], trajectory_interface_scores['interface_dG_SASA_ratio'],
                            trajectory_interface_scores['interface_fraction'], trajectory_interface_scores['interface_hydrophobicity'], trajectory_interface_scores['interface_nres'], trajectory_interface_scores['interface_interface_hbonds'],
                            trajectory_interface_scores['interface_hbond_percentage'], trajectory_interface_scores['interface_delta_unsat_hbonds'], trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                            trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA, trajectory_target_rmsd,
                            trajectory_time_text, traj_seq_notes, settings_file, filters_file, advanced_file]
        insert_data(trajectory_csv, trajectory_data)
    else:
        trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings["target_hotspot_residues"], trajectory_sequence,'B0,',
                           trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'], trajectory_metrics['pae'], trajectory_metrics['i_pae'],]+['']*31
    return trajectory_data,trajectory




