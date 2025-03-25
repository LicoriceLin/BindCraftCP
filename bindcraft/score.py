from ._import import *
from functions import *
from .bc_util import init_task
# from .post_design import Metrics

from .util import _RFD_benchmark_filters as _default_mpnn_filter
from .util import filters_type,check_filters

class Score:
    '''
    Step 2 of original BindCraft. Note: Selection of Backbones / 
    Take MPNN.
    Sample Hallucination Trajectories, Genearate Backbones Candidates.
    '''
    def __init__(self,
        outdir:str,
        binder_name:str,
        post_ana_dir='Post',
        mode:Literal['direct','grafted']='grafted',
        mpnn_rescore_path:str='mpnn_rescore',
        rescore_path:str='rescore',
        filters:filters_type=_default_mpnn_filter
        ):
        '''
        Prefilter (maintained in Metrics Core)
        --- ---
        None:
            load_metrics `Post` & filter 
        Direct:
            pdbfile, rescore_chains: `rescore/log.csv`
        Templated:
            grafted & rescore on full structures & filter: `rescore/log.csv`
        --- ---

        MPNN_Sampler
        --- ---
        MPNN modes: See .mpnn.MPNNSampler: `Post/MPNN.csv`
        None:
            No extra Steps, mpnn_df['filt']=='All_Good' 
        Direct:
            Direct refold & filter: `mpnn_rescore/log.csv`
        Templated:
            Refold with template from `rescore/{design_id}.pdb` & filter: `mpnn_rescore/log.csv`
        --- ---

        Validator (Full Rescorer)
        --- ---
        init from Metrics. take local target_settings / advanced_settings.

        None/Direct:
        read-in Post/MPNN.csv, choose 'All_Good', take target pdbs/chains, refold
        Templated:
        read-in Post/MPNN.csv, choose 'All_Good', refold with template from `rescore/{design_id}.pdb` 
        TODO load refold results directly, no need to fold the same thing twice.
        --- ---
        
        TODO:
        a design_env class,
        maintain Designer, Augmentor(sub-class of Metrics, with filter/rescorer/mpnn_sampler), Scorer, Collector (sub-class of Metrics)

        TODO:
        Highlight in Lab Meeting:
        1.CycPep Designs;
        2.Surf-Only Design for Hard-Target/Accelerations: Preliminary Positive results;
        3.Retro-Spective Ana for old metrics (RFD-Cyc; MPOP1); 
        4.New Metrics: PI/PTM/Stability/Interactions;
        5.Smarter biased MPNN;
        6.Automation of my pipelines.
        '''
        self.outdir=outdir
        self.binder_name=binder_name
        self.post_ana_dir=post_ana_dir
        self.mpnn_rescore_path=mpnn_rescore_path
        self.rescore_path=rescore_path
        self.filters=filters
        self.mode=mode


        self._p=Path(outdir)
        setting_file=self._p/f'{binder_name}.json'
        self.target_settings_file=str(setting_file)
        with open(setting_file,'r') as f:
            target_settings:Dict[str,Any]=json.load(f)
        self.starting_pdb=target_settings['starting_pdb']
        self.chains=target_settings['chains']
        self.target_hotspot_residues=target_settings['target_hotspot_residues']
        self.advanced_settings_path=str(self._p/'advanced_settings.json')
        self.filter_settings_path='settings_filters/no_filters.json'
        
        self.init_task()
        self.load_mpnn_df()


    def load_mpnn_df(self):
        if self.mode=='grafted':
            mpnn_df=pd.read_csv(f'{self.outdir}/Trajectory/{self.post_ana_dir}/MPNN.csv').set_index('Design')
            mpnn_rescore_df=pd.read_csv(f'{self.outdir}/Trajectory/{self.rescore_path}/log.csv').set_index('Design')
            mpnn_rescore_df.columns=[ 'r:'+i if i !='r_pdb' else i for i in mpnn_rescore_df.columns ]
            self.mpnn_df=pd.merge(left=mpnn_df,right=mpnn_rescore_df,left_index=True,right_index=True)
            self.mpnn_df=check_filters(self.mpnn_df,self.filters).sort_values(by='seq',key=lambda x:x.str.len())
            self.rescore_df=pd.read_csv(f'{self.outdir}/Trajectory/{self.rescore_path}/log.csv').set_index('Design')
            # rescore df would be used to load templates
            return self.mpnn_df
        elif self.mode=='direct':
            raise NotImplementedError
        else:
            raise ValueError

    def init_task(self):
        (self.target_settings, self.advanced_settings, self.filters,
            self.settings_file,self.filters_file,self.advanced_file,
            self.design_models, self.prediction_models, self.multimer_validation,
            self.design_paths,self.trajectory_csv,self.mpnn_csv,self.final_csv,self.failure_csv,self.trajectory_dirs
            )=init_task(
            load_previous_target_settings=self.target_settings_file,
            advanced_settings_path=self.advanced_settings_path,
            filter_settings_path=self.filter_settings_path,
            design_path=self.outdir,
            binder_name=self.binder_name,
            starting_pdb=self.starting_pdb,
            chains=self.chains,
            target_hotspot_residues=self.target_hotspot_residues,
            lengths = "10,20",number_of_final_designs = 1,
            )
        self._cyclic=self.advanced_settings['cyclize_peptide']

    def score(self):
        raise NotImplementedError
        prev_seq_len=-1
        good_mpnn_df=self.mpnn_df[self.mpnn_df['filt']=='All_Good']
        for design_id,sub_df in good_mpnn_df.groupby('design_id'):

            seq=s['seq']
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
    # try:
    # import pdb;pdb.set_trace()
    seq_notes = validate_design_sequence(binder_sequence, mpnn_complex_averages.get('Relaxed_Clashes', 0), advanced_settings)
        # assert 
    # except:
        # import pdb;pdb.set_trace()
    end_time = time.time() - b_time
    elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(end_time // 3600), int((end_time % 3600) // 60), int(end_time % 60))}"

    # Insert statistics about MPNN design into CSV, will return None if corresponding model does note exist
    model_numbers = range(1, 6)
    statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                        'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                        'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                        'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

    # Initialize mpnn_data with the non-statistical data
    # try:
    if 1 in complex_statistics:
        c_=complex_statistics[1] 
    else:
        k=list(complex_statistics.keys())[0]
        complex_statistics[k]
    # except:
    #     return complex_statistics
    mpnn_data = [design_name, 'Direct', len(binder_sequence), seed, 0., c_.get('target_interface_residues',''), 
            binder_sequence, c_.get('mpnn_interface_residues',''), 'mpnn_score', 'place_holder_id']
    # WARNING! In original bindcraft, it seems that only the last mpnn_interface_residues are saved.
    # should propose this bug for further modification
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
