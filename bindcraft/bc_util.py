from functions import *
from ._import import *

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
    