import os, time, gc, io
import contextlib
import json
from datetime import datetime
from ipywidgets import HTML, VBox
from IPython.display import display
from functions import *

from typing import Dict,List,Tuple

def init_task(
    load_previous_target_settings:str = "",
    advanced_settings_path:str='settings_advanced/default_4stage_multimer_cyc.json',
    filter_settings_path:str='settings_filters/no_filters.json',
    ### new run ### 
    design_path:str = "output/PDL1-curate/",
    binder_name:str = "PDL1",
    starting_pdb:str = "example/PDL1.pdb",
    chains:str = "A" ,
    target_hotspot_residues:str = "",
    lengths:str = "10,20",
    number_of_final_designs:int = 1,
    ):#->Dict[str,str]:
    '''
    creat file system, 
    init arg dict of `settings`, `advanced_settings_path`, `filter_settings_path`
    '''
    if load_previous_target_settings:
        target_settings_path = load_previous_target_settings
    else:
        lengths = [int(x.strip()) for x in lengths.split(',') if len(lengths.split(',')) == 2]

        if len(lengths) != 2:
            raise ValueError("Incorrect specification of binder lengths.")

        settings = {
            "design_path": design_path,
            "binder_name": binder_name,
            "starting_pdb": starting_pdb,
            "chains": chains,
            "target_hotspot_residues": target_hotspot_residues,
            "lengths": lengths,
            "number_of_final_designs": number_of_final_designs
        }

        target_settings_path = os.path.join(design_path, binder_name+".json")
        os.makedirs(design_path, exist_ok=True)

        with open(target_settings_path, 'w') as f:
            json.dump(settings, f, indent=4)
    args = {"settings":target_settings_path,
        "filters":filter_settings_path,
        "advanced":advanced_settings_path}
    
    settings_path, filters_path, advanced_path = (args["settings"], args["filters"], args["advanced"])
    # perform checks of input setting files
    target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)
    ### load settings from JSON
    settings_file = os.path.basename(settings_path).split('.')[0]
    filters_file = os.path.basename(filters_path).split('.')[0]
    advanced_file = os.path.basename(advanced_path).split('.')[0]
    ### load AF2 model settings
    design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])
    ### perform checks on advanced_settings
    bindcraft_folder = "." #TODO bindcraft_folder to args
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)
    ### generate directories, design path names can be found within the function
    design_paths = generate_directories(target_settings["design_path"])
    ### generate dataframes
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv:str = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
    mpnn_csv:str = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
    final_csv:str = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
    failure_csv:str = os.path.join(target_settings["design_path"], 'failure_csv.csv')

    trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]

    create_dataframe(trajectory_csv, trajectory_labels)
    create_dataframe(mpnn_csv, design_labels)
    create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, args["filters"])
    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
    return (
            target_settings, advanced_settings, filters,
            settings_file,filters_file,advanced_file,
            design_models, prediction_models, multimer_validation,
            design_paths,trajectory_csv,mpnn_csv,final_csv,failure_csv,
            trajectory_dirs
        )
    # return args
    
def bindcraft_hallucinate(
    design_name:str,
    length:int,
    target_settings:Dict[str,str|int|List[int]],
    advanced_settings:BasicDict,
    design_models,
    design_paths,
    helicity_value:float=0.,
    seed:int|None=None,
    binder_chain:str = "B"
    ):
    '''
    to be curated.
    '''
    trajectory_start_time = time.time()
    trajectory_csv:str = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
    failure_csv:str = os.path.join(target_settings["design_path"], 'failure_csv.csv')
    if design_paths is None:
        design_paths=generate_directories(target_settings["design_path"])

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
        )
    trajectory_metrics:dict = copy_dict(trajectory.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae
    trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}
    
    trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
    trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
    # time trajectory
    trajectory_time = time.time() - trajectory_start_time
    trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
    pr_relax(trajectory_pdb, trajectory_relaxed)
    # Calculate clashes before and after relaxation
    num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
    num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

    # secondary structure content of starting trajectory binder and interface
    trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_i_plddt, trajectory_ss_plddt = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

    # analyze interface scores for relaxed af2 trajectory
    trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain)

    # starting binder sequence
    trajectory_sequence = trajectory.get_seq(get_best=True)[0]

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
    return trajectory_data,trajectory

def bindcraft_score(
    binder_sequence:str,
    design_name:str,
    # from `init_task`
    target_settings:Dict[str,str|int|List[int]],
    advanced_settings:BasicDict,
    filters:Dict[str,Dict[str,float|bool|None]], #
    prediction_models:List[int], 
    #  
    # extra optional values,  
    trajectory_pdb:str|None=None, # for binder rmsd comparison
    binder_chain:str='B', # binder chain id
    seed:int|None=None, # random seed
    # only for csv logs, place_holders
    settings_file:str='settings_file',
    filters_file:str='filters_file',
    advanced_file:str='advanced_file',
    design_paths:Dict[str,str]|None=None,
    # reuse models to save time slightly
    complex_prediction_model:mk_afdesign_model|None=None,
    binder_prediction_model:mk_afdesign_model|None=None
    ):
    '''
    Must run `init_task` first to:
        - speficify starting_pdb/chain/hotspots,  
        - create file trees / configs vals  
        - start pyrosetta  

    Notable Paramters:
    --- ---
    `design_name`: 
        must be unique! (will be fixed later)
    `filters`: 
        only tested for no_filters
    `prediction_models`: 
        numbers from 0-4, ${model_id-1} in `params`;   
        recommend to use variable created by `init_task`
    `complex_prediction_model`/`binder_prediction_model`:
        reuse previous models to save time slightly;  
        don't `clear_mem()` when reusing.
    `design_paths`:
        recommend to use variable created by `init_task`
        None -> rerun `generate_directories` to gen, extra time.
    '''
    # clear_mem()
    mpnn_csv:str = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
    final_csv:str = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
    failure_csv:str = os.path.join(target_settings["design_path"], 'failure_csv.csv')
    if design_paths is None:
        design_paths=generate_directories(target_settings["design_path"])
    b_time = time.time()
    complex_opt = predict_binder_complex(
            prediction_model=complex_prediction_model,
            binder_sequence=binder_sequence, 
            mpnn_design_name=design_name,
            target_pdb=target_settings['starting_pdb'], 
            chain=target_settings['chains'],
            length=len(binder_sequence), 
            trajectory_pdb=trajectory_pdb, 
            prediction_models=prediction_models, 
            advanced_settings=advanced_settings,
            filters=filters, 
            design_paths=design_paths, 
            failure_csv=failure_csv)
    if complex_prediction_model is None:
        complex_statistics, _,complex_prediction_model=complex_opt 
    else:
        complex_statistics, _=complex_opt 

    mpnn_complex_averages = calculate_averages(complex_statistics, handle_aa=True)

    alone_opt=predict_binder_alone(
        prediction_model=binder_prediction_model, 
        binder_sequence=binder_sequence, 
        mpnn_design_name=design_name, 
        length=len(binder_sequence), 
        trajectory_pdb=trajectory_pdb, 
        binder_chain=binder_chain, 
        prediction_models=prediction_models, 
        advanced_settings=advanced_settings, 
        design_paths=design_paths)
    if binder_prediction_model is None:
        binder_statistics,binder_prediction_model=alone_opt 
    else:
        binder_statistics=alone_opt
    # binder_prediction_model.restart
    binder_averages = calculate_averages(binder_statistics)
    seq_notes = validate_design_sequence(binder_sequence, mpnn_complex_averages.get('Relaxed_Clashes', None), advanced_settings)
    
    end_time = time.time() - b_time
    elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(end_time // 3600), int((end_time % 3600) // 60), int(end_time % 60))}"

    # Insert statistics about MPNN design into CSV, will return None if corresponding model does note exist
    model_numbers = range(1, 6)
    statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                        'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                        'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                        'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

    # Initialize mpnn_data with the non-statistical data
    mpnn_data = [design_name, 'Direct', len(binder_sequence), seed, 0., target_settings["target_hotspot_residues"], binder_sequence, 'mpnn_interface_residues', 'mpnn_score', 'place_holder_id']

    # Add the statistical data for mpnn_complex
    for label in statistics_labels:
        mpnn_data.append(mpnn_complex_averages.get(label, None))
        for model in model_numbers:
            mpnn_data.append(complex_statistics.get(model, {}).get(label, None))

    # Add the statistical data for binder
    for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:  # These are the labels for binder alone
        mpnn_data.append(binder_averages.get(label, None))
        for model in model_numbers:
            mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

    # Add the remaining non-statistical data
    mpnn_data.extend([elapsed_mpnn_text, seq_notes, settings_file, filters_file, advanced_file])

    # insert data into csv
    insert_data(mpnn_csv, mpnn_data)

    # find best model number by pLDDT
    plddt_values = {i: mpnn_data[i] for i in range(11, 15) if mpnn_data[i] is not None}

    # Find the key with the highest value
    highest_plddt_key = int(max(plddt_values, key=plddt_values.get))

    # Output the number part of the key
    best_model_number = highest_plddt_key - 10
    best_model_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{design_name}_model{best_model_number}.pdb")
    shutil.copy(best_model_pdb, design_paths["Accepted"])

    # insert data into final csv
    final_data = [''] + mpnn_data
    insert_data(final_csv, final_data)
    return (mpnn_complex_averages,binder_averages,seq_notes,complex_prediction_model,binder_prediction_model)



if __name__=='__main__':

 
    (target_settings, advanced_settings, filters,
        settings_file,filters_file,advanced_file,
        design_models, prediction_models, multimer_validation,
        design_paths,trajectory_csv,mpnn_csv,final_csv,failure_csv,
        trajectory_dirs
        )=init_task(
        load_previous_target_settings="output/PDL1-curate1/PDL1.json",
        advanced_settings_path='settings_advanced/default_4stage_multimer_cyc.json',
        filter_settings_path='settings_filters/no_filters.json',
        design_path = "output/PDL1-curate1/",
        binder_name = "PDL1",
        starting_pdb = "example/PDL1.pdb",
        chains = "A" ,
        target_hotspot_residues = "",
        lengths = "10,20",
        number_of_final_designs = 1,
        )
    
    trajectory_data,trajectory=bindcraft_hallucinate('poc_cyc_4',16,target_settings,advanced_settings,design_models,design_paths,seed=42)
    (mpnn_complex_averages,binder_averages,seq_notes,
    complex_prediction_model,binder_prediction_model)=bindcraft_score(
        'HAMRTNPTNPKTLEEC',
        design_name='poc_cyc4_doublecheck',
        target_settings=target_settings,
        advanced_settings=advanced_settings,
        filters=filters,
        prediction_models=prediction_models,
        design_paths=design_paths,
        # failure_csv=failure_csv
        )

    # (mpnn_complex_averages,binder_averages,seq_notes,
    #  complex_prediction_model,binder_prediction_model)=bindcraft_score(
    #     'DSDKRGEEIKKWMMEKIAAQM',
    #     design_name='holding-name122',
    #     target_settings=target_settings,
    #     advanced_settings=advanced_settings,
    #     filters=filters,
    #     prediction_models=prediction_models,
    #     complex_prediction_model=complex_prediction_model,
    #     binder_prediction_model=binder_prediction_model
    # )
    # args=init_task(
    #     load_previous_target_settings="output/PDL1-curate/PDL1.json",
    #     advanced_settings_path='settings_advanced/default_4stage_multimer_cyc.json',
    #     filter_settings_path='settings_filters/no_filters.json',
    #     )
    # (target_settings, advanced_settings, filters,
    # settings_file,filters_file,advanced_file,
    # design_models, prediction_models, multimer_validation,
    # design_paths,trajectory_csv,mpnn_csv,final_csv,failure_csv,
    # trajectory_dirs
    # )=parse_args(args)
    # seed = 42
    # length=15
    # design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
    # helicity_value=0.2

    # trajectory:mk_afdesign_model = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
    #                                         target_settings["target_hotspot_residues"], length, seed, helicity_value,
    #                                         design_models, advanced_settings, design_paths, failure_csv)
    
    '''
    pdb cmds:
    n:next line;
    s:next line & enter func;
    r: until current func returns; 
    unt: execution until the line 
    w: where I am
    '''    
    
    '''
    af_model._pdb: `prep_pdb`;`make_fixed_size`
    af_model._pdb['idx']['chain']: (L,)
    af_model._pdb['idx']['residue']: (L,)
    af_model._pdb["batch"]["aatype"]: (L+Lb,)
    af_model._pdb["batch"]["all_atom_mask"]: (L+Lb,37)
    af_model._pdb["batch"]["all_atom_positions"]: (L+Lb,37,3)

    af_model._inputs: 
    # see `prep_input_features` for default values
    
    {
    'aatype': (L+Lb,) #zero_init, 
    'target_feat': (L+Lb,20), #all zeros? 
    'msa_feat':(1, 130, 49), # cluster_msa[20aa+unk&gap&mask]+has_del&del_val&del_mean+cluster_profile[23]
    'seq_mask':(L+Lb,), #ones 
    'msa_mask':(L+Lb,), #ones
    'msa_row_mask', 
    'atom14_atom_exists', 
    'atom37_atom_exists', 
    'residx_atom14_to_atom37', 
    'residx_atom37_to_atom14', 
    'residue_index':(L+Lb,), # 50 indent between chain
    'extra_deletion_value', 
    'extra_has_deletion', 
    'extra_msa', 
    'extra_msa_mask', 
    'extra_msa_row_mask', 
    'template_aatype', 
    'template_all_atom_mask', 
    'template_all_atom_positions'(1, L+Lb, 37, 3), 
    'template_mask', 
    'template_pseudo_beta', 
    'template_pseudo_beta_mask', 
    'asym_id', 
    'sym_id', 
    'entity_id', 
    'all_atom_positions':(L+Lb, 37, 3), #zero_init
    'batch':
        {
        'aatype', 
        'all_atom_mask', 
        'all_atom_positions':(L+Lb, 37, 3)
        }, 
    'rm_template', 
    'rm_template_seq', 
    'rm_template_sc', 
    'bias', 
    'offset':(L+Lb,L+Lb),
    }

    '''

# def parse_args(args:Dict[str,str]):
#     '''
#     `args` from `init_task`
#     '''
#     settings_path, filters_path, advanced_path = (args["settings"], args["filters"], args["advanced"])
#     # perform checks of input setting files
#     target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)
#     ### load settings from JSON
#     settings_file = os.path.basename(settings_path).split('.')[0]
#     filters_file = os.path.basename(filters_path).split('.')[0]
#     advanced_file = os.path.basename(advanced_path).split('.')[0]
#     ### load AF2 model settings
#     design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])
#     ### perform checks on advanced_settings
#     bindcraft_folder = "." #TODO bindcraft_folder to args
#     advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)
#     ### generate directories, design path names can be found within the function
#     design_paths = generate_directories(target_settings["design_path"])
#     ### generate dataframes
#     trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

#     trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
#     mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
#     final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
#     failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

#     trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]

#     create_dataframe(trajectory_csv, trajectory_labels)
#     create_dataframe(mpnn_csv, design_labels)
#     create_dataframe(final_csv, final_labels)
#     generate_filter_pass_csv(failure_csv, args["filters"])
#     pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
#     return (
#             target_settings, advanced_settings, filters,
#             settings_file,filters_file,advanced_file,
#             design_models, prediction_models, multimer_validation,
#             design_paths,trajectory_csv,mpnn_csv,final_csv,failure_csv,
#             trajectory_dirs
#         )

# def init_pyrosetta(advanced_settings):
#     pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
