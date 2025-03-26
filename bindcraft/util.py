from ._import import *

#%% pdb
def write_out(strcture:Entity,file:str='tmp.pdb',write_end:bool=True, preserve_atom_numbering:bool=False)->None:
    '''
    write out an Entity (from the whole structure to a single atom) to pdb file.
    use the `write_end` to write an END line, 
    the `preserve_atom_numbering` to renumber atom from 1.
    will be compatible with all `allowed_residue_source` in the future
    '''
    io = BP.PDBIO()
    io.set_structure(strcture)
    io.save(file,write_end=write_end,preserve_atom_numbering=preserve_atom_numbering)

# %% metrics IO
def id2pdbfile(design_id:str,outdir:str,mode:Literal['relax','mpnn','design','single','design_relax']='relax'):
    if mode not in ['design','design_relax']:
        f=glob.glob(f'{outdir}/Accepted/{design_id}_model*.pdb')[0]
    if mode == 'relax':
        return f    
    elif mode == 'mpnn':
        return f.replace('/Accepted','/MPNN')
    elif mode == 'design':
        return f'{outdir}/Trajectory/{design_id}.pdb'
    elif mode == 'single':
        return f.replace('/Accepted','/MPNN/Binder')
    elif mode == 'design_relax':
        return f'{outdir}/Trajectory/Relaxed/{design_id}.pdb'
    else:
        raise NotImplementedError

outdir_from_metrics=lambda metrics:str(Path(metrics.iloc[0]['pdbfile']).parents[1])

def _avg_InterfaceAAs(aa_dicts:List[Dict[str,int|float]]):
    o=aa_dicts[0].copy()
    l=len(aa_dicts)
    for aa_dict in aa_dicts[1:]:
        for k,v in aa_dict.items():
            o[k]+=v
    o={k:v/l for k,v in o.items()}
    return o

_meta_cols=[
    'Design','Length','Target_Hotspot','Sequence','InterfaceResidues']
_metrics_col=['pLDDT','pTM','i_pTM','pAE','i_pAE','i_pLDDT','ss_pLDDT',
                'Unrelaxed_Clashes','Relaxed_Clashes','ShapeComplementarity','PackStat',
                'Binder_Energy_Score','dG','dSASA','dG/dSASA','Interface_SASA_%',
                'Interface_Hydrophobicity','Surface_Hydrophobicity',
                'n_InterfaceResidues','n_InterfaceHbonds','n_InterfaceUnsatHbonds',
                'Interface_Helix%','Interface_BetaSheet%','Interface_Loop%',
                'Binder_Helix%','Binder_BetaSheet%','Binder_Loop%',
                'Binder_pLDDT','Binder_pTM','Binder_pAE']
_aa_col='InterfaceAAs'

def read_bc_metrics(outdir:str,first_two_only:bool=False):
    '''
    read `final_design_stats`
    fixed rules: `Design` should always be the index col. 
    '''
    final_score_df=pd.read_csv(f'{outdir}/final_design_stats.csv'
        ).drop_duplicates(subset=['Design'], keep='last')
    if not first_two_only:
        used_cols=_meta_cols+[f'Average_{i}' for i in _metrics_col+[_aa_col]]
        bc_metrics=final_score_df[used_cols].copy()
        bc_metrics.columns=[i.replace('Average_','') for i in bc_metrics.columns]
        bc_metrics[f'{_aa_col}']=bc_metrics[f'{_aa_col}'].apply(lambda x:literal_eval(x))
    else:
        aa_col_=[f'{i}_{_aa_col}' for i in [1,2]]
        arr_1=final_score_df[[f'1_{i}' for i in _metrics_col]].to_numpy()
        arr_2=final_score_df[[f'2_{i}' for i in _metrics_col]].to_numpy()
        arr_avg=(arr_1+arr_2)/2
        df_avg=pd.DataFrame(arr_avg,columns=[f'{i}' for i in _metrics_col])
        
        df_aas=final_score_df[aa_col_]
        df_avg[f'{_aa_col}']=df_aas.apply(lambda s:_avg_InterfaceAAs([literal_eval(s[i]) 
                        for i in aa_col_]),axis=1)
        
        bc_metrics=pd.concat([final_score_df[_meta_cols],df_avg],axis=1)
        
    bc_metrics=bc_metrics.set_index('Design')
    bc_metrics['pdbfile']=[id2pdbfile(i,outdir) for i in bc_metrics.index]
    return bc_metrics

def read_design_metrics(outdir:str):
    design_metrics=pd.read_csv(f'{outdir}/trajectory_stats.csv'
        ).drop_duplicates(subset=['Design'], keep='last').set_index('Design')
    design_metrics['InterfaceAAs']=design_metrics['InterfaceAAs'].apply(literal_eval)
    # design_metrics['InterfaceResidues'].fillna('',inplace=True)
    design_metrics.fillna({'InterfaceResidues':''},inplace=True)
    design_metrics['pdbfile']=[id2pdbfile(i,outdir=outdir,mode='design') for i in design_metrics.index]
    return design_metrics

def dump_metrics(df:pd.DataFrame,file:str):
    '''
    syntactic sugar
    .to_csv(...,index_label='Design')
    '''
    df.to_csv(file,index_label='Design')

# %% metrics filters
def _load_default_filters(bindcraft_benchmark:bool=False):
    '''
    Default default_filters,  
    remove "Average_" for compatibility with `design_stat`  
    adjust several thresholds (pLDDT, ShapeComplementarity, Surface_Hydrophobicity, Binder_Loop%)
    '''
    filter:dict=json.load(open('settings_filters/default_filters.json','r'))
    design_stat_col=['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot',
       'Sequence', 'InterfaceResidues', 'pLDDT', 'pTM', 'i_pTM', 'pAE',
       'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes',
       'Binder_Energy_Score', 'Surface_Hydrophobicity', 'ShapeComplementarity',
       'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%',
       'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds',
       'InterfaceHbondsPercentage', 'n_InterfaceUnsatHbonds',
       'InterfaceUnsatHbondsPercentage', 'Interface_Helix%',
       'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
       'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Target_RMSD',
       'TrajectoryTime', 'Notes', 'TargetSettings', 'Filters',
       'AdvancedSettings']
    
    filter={k.replace("Average_",""):v for k,v in filter.items() if "Average_" in k and (v.get("threshold",None) is not None or k =='Average_InterfaceAAs') }
    filter={i:filter[i] for i in filter if i in design_stat_col}
    if bindcraft_benchmark:
        return filter
    else:
        filter.pop('InterfaceAAs')
        # filter['InterfaceAAs']['C']={'threshold': 1, 'higher': False}
        filter['pLDDT']={'threshold': 0.75, 'higher': True}
        filter['ShapeComplementarity']={'threshold': 0.55, 'higher': True}
        filter['Surface_Hydrophobicity']={'threshold': 0.55, 'higher': False}
        filter['Binder_Loop%']={'threshold': 35, 'higher': False}
    return filter
    # filter['n_InterfaceUnsatHbonds']={'threshold': 40, 'higher': False}
filters_type=Dict[str,Dict[str,float|int|bool]]
_bc_benchmark_filters=_load_default_filters(True)
_RFD_benchmark_filters={
    'r:plddt':{'threshold': 0.80, 'higher': True},
    'r:i_pae':{'threshold': 10/31, 'higher': False}, # 31 is the normalization scale in colabdesign.
    'r:rmsd':{'threshold': 1, 'higher': False},
    }
_refold_filter_strict=_RFD_benchmark_filters
_refold_filter_loose={'r:plddt':{'threshold': 0.7, 'higher': True},
    'r:i_pae':{'threshold': 0.4, 'higher': False},
    'r:rmsd':{'threshold': 2, 'higher': False}
    }

_bc_design_benchmark_filters={
    'pLDDT':{'threshold': 0.70, 'higher': True},
    'n_InterfaceResidues':{'threshold': 3, 'higher': True},
    'Relaxed_Clashes':{'threshold': 0, 'higher': False},
}

_default_filters=_load_default_filters()
_simpliest_filter={'pLDDT':{'threshold': 0.70, 'higher': True}}

def _check_filters(entry:pd.Series, filters:dict=_default_filters):
    '''
    only check consensus cols in `entry` and `filters`
    '''
    unmet_conditions = []

    # check filters against thresholds
    for label, conditions in filters.items():
        # special conditions for interface amino acid counts
        if label == 'InterfaceAAs':
            for aa, aa_conditions in conditions.items():
                if entry.get(label) is None:
                    continue
                value = entry.get(label).get(aa)
                if value is None or aa_conditions["threshold"] is None:
                    continue
                if aa_conditions["higher"]:
                    if value < aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
                else:
                    if value > aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
        elif label in ['InterfaceResidues','Target_Hotspot']:
            # conditions should be a string like "B1,B7,..."
            expected_res:List[str]= conditions.split(",")
            res:List[str]= entry.get(label).split(",")
            for r in expected_res:
                if res not in expected_res:
                    unmet_conditions.append(label)
                    break
               
        else:
            # if no threshold, then skip
            value = entry.get(label)
            if value is None or conditions["threshold"] is None:
                continue
            if conditions["higher"]:
                if value < conditions["threshold"]:
                    unmet_conditions.append(label)
            else:
                if value > conditions["threshold"]:
                    unmet_conditions.append(label)

    # if all filters are passed then return True
    if len(unmet_conditions) == 0:
        return "All_Good"
    # if some filters were unmet, print them out
    else:
        return ';'.join(unmet_conditions)

def check_filters(metrics:pd.DataFrame,filters:dict=_default_filters):
    o=[]
    for _,s in metrics.iterrows():
        o.append(_check_filters(s,filters))
    metrics['filt']=o
    return metrics

def show_filter(metrics:pd.DataFrame)->Tuple[plt.Figure,plt.Axes]: #,filters:dict=_default_filters
    # check_filters(stat,filters)
    assert 'filt' in metrics.columns,'run `check_filters` first!'
    o=metrics['filt'].to_list()

    failed_labels=[]
    for i in o:
        failed_labels.extend(i.split(';'))
    fig,ax=plt.subplots(1,1)
    s=pd.Series(failed_labels).value_counts().sort_values()/len(metrics)
    ax=sns.barplot(s,ax=ax,order=[i for i in s.index if i!='All_Good']+['All_Good'])
    ax.set_ylim([0,1])
    ax.bar_label(ax.containers[0], fontsize=10,fmt=lambda x:f'{x*100:.1f}')
    ax.set_ylabel('Failed Rate')
    ax.set_xlabel('Failed Criteria')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_verticalalignment('top')
        tick.set_horizontalalignment('right')
        if tick.get_text()=='All_Good':
            tick.set_color('red')
            # tick.set_text('All Good')

    return fig,ax


def is_pickleable(obj) -> bool:
    try:
        pkl.dumps(obj)
        return True
    except Exception:
        return False