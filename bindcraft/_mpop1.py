from ._import import *
from .plt_util import plot_protein_features
from .biophy_metrics import (
    single_mutsite_parse,cal_sasa,write_out,
    PDBFile,pdbfixer,propka_single,musite_parse_recipe)

def peel_pdbfile(pdbfile:str):
    '''
    two possible inputs:
        - pdbfile: output/*/Accepted/*_model*.pdb
        - id: shorter_6c0b_500x5_52_dldesign_3_af2pred,pae=5.628-TcdB
    '''
    if pdbfile.endswith('.pdb'):
        pdbfile=Path(pdbfile).stem[:-7]
    if '-' in pdbfile[-10:]:
        return pdbfile
    else:
        return pdbfile+'-Skp1'

id2pdbfile=lambda x:glob.glob(f'output/MPOP1-benchmark*/Accepted/{x.replace("-Skp1","")}*.pdb')[0]


def plot_feat_MPOP1(df:pd.DataFrame,outpdf:str,cols:List[str]|None=None):
    '''
    hue, order, palette & statistic pairs are hard-coded for MPOP1 system.
    '''
    if cols is None:
        cols=[i for i in df.columns if i not in ['binder','assay_target', 'tag']]

    with PdfPages(outpdf) as pdf:
        for y in tqdm(cols):#['pi-fold', 'pi-unfold', 'pH-opt', 'dG-opt']:
            fig,ax=plt.subplots(1,1,figsize=(6,6))
            order=['NonHit', 'Hit','Promiscuous']
            hue_order=['Skp1', 'FBXW7', 'TcdB']
            palette=sns.color_palette(['tab:blue','tab:orange','tab:purple'])
            sns.boxplot(df,y=y,x='tag',hue='assay_target',fliersize=0,boxprops={'alpha': 0.4},order=order,ax=ax,hue_order=hue_order,palette=palette)
            # sns.violinplot(df,y=y,x='tag',hue='assay_target',alpha=0.4,order=order,ax=ax,hue_order=hue_order,palette=palette)
            sns.stripplot(df,y=y,x='tag',hue='assay_target',ax=ax,order=order,dodge=True,hue_order=hue_order,palette=palette)
            pairs=[('Promiscuous', 'Hit'), ('Hit', 'NonHit')]
            annotator = Annotator(ax, pairs, data=df, x='tag', y=y, order=order,plot='violinplot')
            # pairs=result
            # annotator = Annotator(ax, pairs, data=df, x='tag', y=y, order=order,hue='assay_target',hue_order=hue_order)
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
            annotator.apply_and_annotate()

            ax.set_xticklabels([i.get_text().replace('-','\n') for i in ax.get_xticklabels()])
            ax.set_xlabel('')

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=[(handles[i], handles[i+3]) for i in range(3)],
                labels=hue_order)
            fig.suptitle(y)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def plot_ptm_tracks_MPOP1(ptm_tracks:dict,propka_metrics:pd.DataFrame):
    '''
    ptm_tracks: from `gen_ptm_tracks_MPOP1`
    '''
    feature_names=['ppi', 'sasa', 'MeR','MeK', 'AcK','SuK', 'PiX', 'HoX', 'UbK','Gly'] #'PrQ',
    colors=['navy','tab:blue']+['cyan','teal']+['limegreen']+['crimson','coral']+['rosybrown','slategray','darkseagreen']
    with PdfPages('view_ptm_2.pdf') as pdf:
        for n,o in tqdm(ptm_tracks.items()):
            # o=parse_ptm_track(s,0.5)
            # n=o['name']
            # ptm_tracks[n]=o
            fig,axes=plot_protein_features(
                seq=o['seq'], 
                features=[o[i] for i in feature_names], 
                feature_names=feature_names, 
                colors=colors,
                chunk_size=25,width=8,height_single=2,
                exclude_annot=['ppi'])
            fig.set_dpi(400)
            
            fig.suptitle(o['tag']+'-'+f"{propka_metrics.loc[n]['pi-fold']:.1f}"+'\n'+propka_metrics.loc[n]['binder'])
            pdf.savefig(fig)
            plt.close(fig)

def read_bc_metrics_MPOP1()->pd.DataFrame:
    dfs=[]
    for i in glob.glob('output/MPOP1-benchmark*/final_design_stats.csv'):
        df=pd.read_csv(i)
        df['Design']=df['Design'].apply(peel_pdbfile)
        dfs.append(df)
    used_cols=[
        'Design','Length','Target_Hotspot','Sequence','InterfaceResidues',
        'Average_pLDDT','Average_pTM','Average_i_pTM','Average_pAE','Average_i_pAE','Average_i_pLDDT','Average_ss_pLDDT',
        'Average_Unrelaxed_Clashes','Average_Relaxed_Clashes',
        'Average_Binder_Energy_Score',
        'Average_ShapeComplementarity','Average_PackStat',
        'Average_dG','Average_dSASA','Average_dG/dSASA','Average_Interface_SASA_%',
        'Average_Interface_Hydrophobicity','Average_Surface_Hydrophobicity',
        'Average_n_InterfaceResidues','Average_n_InterfaceHbonds','Average_n_InterfaceUnsatHbonds',
        # 'Average_InterfaceUnsatHbondsPercentage','Average_InterfaceHbondsPercentage',
        'Average_Interface_Helix%','Average_Interface_BetaSheet%','Average_Interface_Loop%',
        'Average_Binder_Helix%','Average_Binder_BetaSheet%','Average_Binder_Loop%',
        'Average_InterfaceAAs',
        'Average_Binder_pLDDT','Average_Binder_pTM','Average_Binder_pAE','Average_Binder_RMSD',
        ]
    ori_bc_metrics=pd.concat(dfs,ignore_index=True)[used_cols]
    ori_bc_metrics.columns=[i.replace('Average_','') for i in ori_bc_metrics.columns]
    ori_bc_metrics=ori_bc_metrics.set_index('Design')
    ori_bc_metrics['pdbfile']=[id2pdbfile(i) for i in ori_bc_metrics.index]
    return ori_bc_metrics

def patch_feats(df:pd.DataFrame,ref_csv:str='MPOP1-2.csv')->pd.DataFrame:
    '''
    inplace operations to pad df with 'binder','assay_target' & 'tag'
    '''
    ref_df=pd.read_csv(ref_csv).set_index('Design')
    df['binder']=[ref_df.loc[i]['binder'] for i in df.index]
    df['assay_target']=[i.split('-')[-1] for i in df.index]
    df['tag']=[ref_df.loc[i]['tag'] for i in df.index]

    tag_dtype = CategoricalDtype(categories=['NonHit','Hit','Promiscuous'], ordered=True)
    assay_dtype = CategoricalDtype(categories=['Skp1', 'FBXW7', 'TcdB'], ordered=True)
    df['tag'] = df['tag'].astype(tag_dtype)
    df['assay_target'] = df['assay_target'].astype(assay_dtype)
    df = df.sort_values(by=['tag', 'assay_target'])
    return df

def gen_ptm_tracks_MPOP1(
    bc_metrics:pd.DataFrame,ptms:Dict[str,Tuple[single_mutsite_parse,str]],
    sasa_threshold:float=0.4,
    ):
    '''
    bc_metrics: from `read_bc_metrics`. future: keep the format, make it generalizable.
    ptms: from `parse_musite_dir`
    '''
    output={}
    for i,s in tqdm(bc_metrics.iterrows()):
        o={}
        o['name']=i
        o['seq']=s['Sequence']
        o['tag']=f"{s['tag']}-{s['assay_target']}"

        o['sasa']=cal_sasa(s['pdbfile'])

        o['ppi']=np.zeros(len(o['seq']),dtype=int)
        o['ppi'][[int(i[1:])-1 for i in s['InterfaceResidues'].split(',')]]=1
        o['ppi'] = o['ppi'] & (o['sasa']>sasa_threshold)
        o['surf']=(o['sasa']>sasa_threshold) & (~o['ppi'])
        o['core']=(o['sasa']<=sasa_threshold).astype(int)

        for k,v in ptms.items():
            o[k]=np.zeros(len(o['seq']),dtype=float)
            ptm=v[0].get(o['name'],[])
            for i in ptm:
                o[k][i[0]-1]=i[2]

        o['PiX']=o['PiST']+o['PiY']
        o['Gly']=o['GlyO']+o['GlyN']
        o['HoX']=o['HoP']+o['HoK']

        output[o['name']]=o

    return output

def ptm_propka(pdbfile:str,
    ptms:Dict[str,Tuple[Dict[str,Tuple[int,str,float]],str]],
    ptm_threshold:float=0.5,
    mut_recipe:Dict[str,Tuple[str,str]]=musite_parse_recipe,
    sasa_threshold:float=0.,
    MPOP1:bool=False):
    '''
    ptms: from `parse_musite_dir`

    '''
    # ptms=ptms.copy()
    mut_recipe_={v[0]:v[1] for v in mut_recipe.keys()}
    for k in ptms.keys():
        ptms[k]=(ptms[k][0],mut_recipe_.get(k,''))
    
    with TemporaryDirectory() as tmpdir:
        chain='B'
        pdb=PDBParser(QUIET=True).get_structure('tmp',pdbfile)[0][chain]
        write_out(pdb,f'{tmpdir}/{Path(pdbfile).stem}.pdb')
        if sasa_threshold>0:
            sasas=cal_sasa(f'{tmpdir}/{Path(pdbfile).stem}.pdb')
        
        mutstrs={}
        for k,v in ptms.items():
            # no need to overwrite 
            if v[1]:
                if MPOP1:
                    des=peel_pdbfile(pdbfile)
                else:
                    des=Path(pdbfile).stem[:-7]
                for l in v[0].get(des,[]):
                    t=l[2]>ptm_threshold and l[2]>=mutstrs.get(l[0],('',0.))[1]
                    if sasa_threshold>0:
                        t= t and sasas[l[0]-1]>sasa_threshold
                    if t:
                        # if v[1]:
                        mutstrs[l[0]]=(f"{PDBData.protein_letters_1to3[l[1]]}-{l[0]}-{v[1]}",l[2])
        # return mutstrs
                        # else:
                        #     if l[0] in mutstrs:
                        #         mutstrs.pop(l[0])

        if len(mutstrs)>0:    
            fixer = pdbfixer.PDBFixer(filename=f'{tmpdir}/{Path(pdbfile).stem}.pdb')
            fixer.applyMutations([i[0] for i in mutstrs.values()], 'B')
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            PDBFile.writeFile(fixer.topology, fixer.positions, open(f'{tmpdir}/{Path(pdbfile).stem}.pdb', 'w'))
            # return PDBParser(QUIET=True).get_structure('tmp',f'{tmpdir}/{Path(pdbfile).stem}.pdb')[0]
            pis=propka_single(f'{tmpdir}/{Path(pdbfile).stem}.pdb',optargs=[f'-c=A'])
        else:
            pis=propka_single(f'{tmpdir}/{Path(pdbfile).stem}.pdb',optargs=[f'-c=B'])
        return pis