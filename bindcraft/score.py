from ._import import *
from functions import *
from .bc_util import init_task
# from .post_design import Metrics

from .util import _RFD_benchmark_filters as _default_mpnn_filter
from .util import filters_type,check_filters
# from functions import hotspot_residues,mk_afdesign_model,add_cyclic_offset

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

class Score:
    '''
    Step 2 of original BindCraft. 
    Direct read `MPNN.df` & Run bindcraft_score.
    `filt` in `MPNN.df`: only run 'All_Good'
    `design_id` in `MPNN.df` & `templated_refold`: refold with traj as template.
    Note: if `templated_refold` & ~`self.multimer_validation`, only models #0 & #1 would be available. 
    '''
    def __init__(self,
        outdir:str,
        binder_name:str,
        # used for rescore from mpnn results
        post_stem='Post',
        mpnn_stem:str='MPNN',
        # used for rescore anything
        rescore_csv:str='',
        target_settings_args={
            'starting_pdb':'',
            'chains':'',
            'target_hotspot_residues':''
            },
        # templated_refold are only available for "rescore from mpnn results"
        templated_refold:bool=False,
        refold_stem:str='rescore', # for refold templates 
        # no need for "rescore from mpnn results". overided by outdir/'advanced_settings.json'
        default_advanced_settings_path:str='settings_advanced/mcmcsampling_multimer_cyc.json', 
        filters:filters_type|None=None, #None: use 'filt' col in mpnn df or skip filter. 
        advanced_settings_overload:Dict[str,Any]={}, # will always override config in jsons.
        
        ):
        '''
        `binder_name`:
            used to fetch `target_settings_file`
                as `load_previous_target_settings` in `init_task`
            for fresh start rescores, please use `target_settings_args`

        `post_stem`
        target_settings_args:
            for fresh created ourdirs (e.g. score designs from other methods.)
            used to create the `target_settings`
            overided by self.outdir/f'{binder_name}.json'
        
        '''
        self._outdir=outdir
        self.outdir=Path(outdir)
        self.binder_name=binder_name
        self._post_stem=post_stem
        self._mpnn_stem=mpnn_stem
        self._refold_stem=refold_stem
        self.templated=templated_refold
        self.filters=filters
        self.advanced_settings_overload=advanced_settings_overload

        self.post_dir=Path(outdir)/'Trajectory'/post_stem
        self.mpnn_csv=self.post_dir/f'{mpnn_stem}.csv'
        self.refold_dir=Path(outdir)/'Trajectory'/refold_stem

        setting_file=self.outdir/f'{binder_name}.json'

        if not rescore_csv:
            self.target_settings_file=str(setting_file)
            with open(setting_file,'r') as f:
                target_settings:Dict[str,Any]=json.load(f)
            for k in ['starting_pdb','chains','target_hotspot_residues']:
                setattr(self,k,target_settings[k])
        else:
            self.target_settings_file=''
            for k in ['starting_pdb','chains','target_hotspot_residues']:
                setattr(self,k,target_settings_args[k])
            
            # self.starting_pdb=target_settings[]
            # self.chains=target_settings[]
            # self.target_hotspot_residues=target_settings[]
        
        if (self.outdir/'advanced_settings.json').exists():
            self.advanced_settings_path=str(self.outdir/'advanced_settings.json')
        else:
            self.advanced_settings_path=default_advanced_settings_path
        self.filter_settings_path='settings_filters/no_filters.json'
        if self.target_settings_file:
            self.load_mpnn_df()
        else:
            assert rescore_csv
            assert not templated_refold

            self.load_any_df(rescore_csv)
        self.init_task()
        
    def load_any_df(self,csvfile:str):
        '''
        for convenience, still named as mpnn_df.
        conserved cols: Design,seq.
        '''
        self.mpnn_df=pd.read_csv(csvfile).set_index('Design').sort_values(by='seq',key=lambda x:x.str.len())
    
    def load_mpnn_df(self):
        '''
        filter conducted in `Metrics.mpnn_sample`
        '''
        mpnn_df=pd.read_csv(self.mpnn_csv).set_index('Design')
        if self.filters is not None:
            if 'filt' in mpnn_df.columns:
                print('re-filter MPNN df by given `filters` dict')
            mpnn_df=check_filters(mpnn_df,self.filters)
        
        if 'filt' in mpnn_df.columns:
            print(f'MPNN df are filtered. {len(mpnn_df)} designs left.')
            mpnn_df=mpnn_df[mpnn_df['filt']=='All_Good'].drop(columns='filt')

        if self.templated:
            assert 'design_id' in mpnn_df.columns
            mpnn_df['template_pdb']=mpnn_df['design_id'].apply(lambda x: str(self.refold_dir/f'{x}.pdb'))

        mpnn_df['seq_len'] = mpnn_df['seq'].str.len()
        mpnn_df = mpnn_df.sort_values(by=['seq_len', 'design_id'],
            ascending=[True, True]).drop(columns='seq_len')
        
        self.mpnn_df=mpnn_df

        # if self.mode=='grafted':
        #     mpnn_df=pd.read_csv(f'{self.outdir}/Trajectory/{self.post_ana_dir}/MPNN.csv').set_index('Design')
        #     mpnn_rescore_df=pd.read_csv(f'{self.outdir}/Trajectory/{self.rescore_path}/log.csv').set_index('Design')
        #     mpnn_rescore_df.columns=[ 'r:'+i if i !='r_pdb' else i for i in mpnn_rescore_df.columns ]
        #     self.mpnn_df=pd.merge(left=mpnn_df,right=mpnn_rescore_df,left_index=True,right_index=True)
        #     self.mpnn_df=check_filters(self.mpnn_df,self.filters).sort_values(by='seq',key=lambda x:x.str.len())
        #     self.rescore_df=pd.read_csv(f'{self.outdir}/Trajectory/{self.rescore_path}/log.csv').set_index('Design')
        #     # rescore df would be used to load templates
        #     return self.mpnn_df
        # elif self.mode=='direct':
        #     raise NotImplementedError
        # else:
        #     raise ValueError

    def init_task(self):
        (self.target_settings, self.advanced_settings, self.filters,
            self.settings_file,self.filters_file,self.advanced_file,
            self.design_models, self.prediction_models, self.multimer_validation,
            self.design_paths,self.trajectory_csv,self.mpnn_csv,self.final_csv,self.failure_csv,self.trajectory_dirs
            )=init_task(
            load_previous_target_settings=self.target_settings_file,
            advanced_settings_path=self.advanced_settings_path,
            filter_settings_path=self.filter_settings_path,
            design_path=self._outdir,
            binder_name=self.binder_name,
            starting_pdb=self.starting_pdb,
            chains=self.chains,
            target_hotspot_residues=self.target_hotspot_residues,
            lengths = "10,20",number_of_final_designs = 1,
            )
        self._cyclic=self.advanced_settings['cyclize_peptide']
        if len(self.advanced_settings_overload)>0:
            print('advanced_settings are overloaded.')
            self.advanced_settings.update(self.advanced_settings_overload)
            with open(self.outdir/'rescore_advanced_settings.json','w') as f:
                json.dump(self.advanced_settings,f)
        if not self.advanced_settings['omit_AAs']:
            self.advanced_settings['omit_AAs']=None

    def score(self,seed:int=42):
        m_model,b_model=self.complex_prediction_model,self.binder_prediction_model
        scored_designs=self._scored_designs()
        mpnn_df=self.mpnn_df.loc[[i for i in self.mpnn_df.index if i not in scored_designs]]
        target_chain=self.target_settings['chains']
        prev_seq_len=-1
        if self.templated:
            for template_pdb,sub_df in tqdm(mpnn_df.groupby(by='template_pdb')):
                seq_len=len(sub_df['seq'].iloc[0])
                m_model.prep_inputs(
                    pdb_filename=template_pdb, 
                    chain=target_chain, binder_chain='B', binder_len=seq_len, 
                    use_binder_template=True, 
                    rm_target_seq=self.advanced_settings["rm_template_seq_predict"],
                    rm_target_sc=self.advanced_settings["rm_template_sc_predict"], 
                    rm_template_ic=True,seed=seed)
                if seq_len!=prev_seq_len:
                    prev_seq_len=seq_len
                    b_model.prep_inputs(length=seq_len,seed=seed)
                if self._cyclic:
                    add_cyclic_offset(m_model, offset_type=2)
                    add_cyclic_offset(b_model, offset_type=2)
                    
                for mpnn_id,seq in sub_df['seq'].items():
                    print(f"run {mpnn_id}")
                    (mpnn_complex_averages,binder_averages,seq_notes,
                    complex_prediction_model,binder_prediction_model
                    )=bindcraft_score(
                        binder_sequence=seq,design_name=mpnn_id,trajectory_pdb=template_pdb,
                        target_settings=self.target_settings,advanced_settings=self.advanced_settings,filters=self.filters,
                        prediction_models=self.prediction_models,
                        settings_file=self.settings_file,filters_file='null',advanced_file=self.advanced_file,
                        complex_prediction_model=m_model,binder_prediction_model=b_model,
                        design_paths=self.design_paths, seed=seed)

        else:
            pdb_file=self.target_settings['starting_pdb']
            for mpnn_id,s in mpnn_df.iterrows():
                seq=s['seq']
                seq_len=len(seq)
                if prev_seq_len!=seq_len:
                    m_model.prep_inputs(pdb_filename=pdb_file, 
                        chain=target_chain, 
                        binder_len=seq_len, 
                        rm_target_seq=self.advanced_settings["rm_template_seq_predict"],
                        rm_target_sc=self.advanced_settings["rm_template_sc_predict"],
                        seed=seed)
                    b_model.prep_inputs(length=seq_len)
                    if self._cyclic:
                        add_cyclic_offset(m_model, offset_type=2)
                        add_cyclic_offset(b_model, offset_type=2)
                    prev_seq_len=seq_len
                print(f"run {mpnn_id}")
                (mpnn_complex_averages,binder_averages,seq_notes,
                    complex_prediction_model,binder_prediction_model)=bindcraft_score(
                    binder_sequence=seq,design_name=mpnn_id,trajectory_pdb=None,
                        target_settings=self.target_settings,advanced_settings=self.advanced_settings,filters=self.filters,
                        prediction_models=self.prediction_models,
                        settings_file=self.settings_file,filters_file='null',advanced_file=self.advanced_file,
                        complex_prediction_model=m_model,binder_prediction_model=b_model,
                        design_paths=self.design_paths, seed=seed,)
                
    def _scored_designs(self)->List[str]:
        if os.path.exists(self.outdir/'final_design_stats.csv'):
            _=pd.read_csv(self.outdir/'final_design_stats.csv')
            if len(_)>0:
                finished=_['Design'].to_list()
            else:
                finished=[]
        else:
            finished=[]
        return finished
        # prev_seq_len=-1
        # good_mpnn_df=self.mpnn_df[self.mpnn_df['filt']=='All_Good']
        # for design_id,sub_df in good_mpnn_df.groupby('design_id'):

        #     seq=s['seq']

    @property
    def complex_prediction_model(self):
        advanced_settings=self.advanced_settings
        if getattr(self,'_complex_prediction_model',None) is None:
            if self.templated:
                use_initial_guess,use_initial_atom_pos=True,True
            else:
                use_initial_guess,use_initial_atom_pos=False,False
            self._complex_prediction_model = mk_afdesign_model(protocol="binder", 
                num_recycles=advanced_settings["num_recycles_validation"], 
                data_dir=advanced_settings["af_params_dir"], 
                use_multimer=self.multimer_validation, 
                use_initial_guess=use_initial_guess, 
                use_initial_atom_pos=use_initial_atom_pos)
        return self._complex_prediction_model
    
    @property
    def binder_prediction_model(self):
        advanced_settings=self.advanced_settings
        if getattr(self,'_binder_prediction_model',None) is None:
            self._binder_prediction_model=mk_afdesign_model(
                protocol="hallucination", use_templates=False, initial_guess=False, 
                use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"], 
                data_dir=advanced_settings["af_params_dir"], use_multimer=self.multimer_validation)
        return self._binder_prediction_model
    
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
