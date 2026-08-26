from .basestep import *
from epitopecraft.steps.scorer.pymol_utils import *
from pymol import cmd
import re
import yaml
from subprocess import run
from warnings import warn
from glob import glob


def _load_chain_residue_spec(spec)->Dict[str,List[str]|str]:
    if spec in (None,''):
        return {}

    if isinstance(spec, dict):
        ret={}
        for chain, residues in spec.items():
            if residues == 'all':
                ret[str(chain)]='all'
            elif isinstance(residues, str):
                ret[str(chain)]=[i for i in re.split(r'[\s,]+', residues.strip()) if i]
            else:
                ret[str(chain)]=[str(i) for i in residues if str(i)]
        return ret

    if isinstance(spec, (list,tuple,set)):
        tokens=[str(i).strip() for i in spec if str(i).strip()]
    elif isinstance(spec, str):
        spec_path=Path(spec)
        if spec_path.exists():
            text=spec_path.read_text()
        else:
            text=spec
        tokens=[i for i in re.split(r'[\s,]+', text.strip()) if i]
    else:
        raise TypeError(f'unsupported residue spec type: {type(spec)}')

    ret={}
    for token in tokens:
        if token.lower() == 'all':
            raise ValueError('ambiguous residue token `all`; use a dict like {"A":"all"} to specify per-chain masks.')
        chain,resi=token[0],token[1:]
        if not chain or not resi:
            raise ValueError(f'invalid residue token: {token}')
        _=ret.setdefault(chain,[])
        _.append(resi)
    return ret


def _map_chain_residue_spec(
    residue_spec:Dict[str,List[str]|str],
    id_map:Dict[str,Dict[str,str]],
    label:str,
    )->Dict[str,List[str]|str]:
    mapped={}
    missing=[]
    for chain,residues in residue_spec.items():
        if chain not in id_map:
            missing.append(f'{chain}:<chain missing>')
            continue
        if residues == 'all':
            mapped[chain]='all'
            continue
        for resi in residues:
            mapped_resi=id_map[chain].get(resi)
            if mapped_resi is None:
                missing.append(f'{chain}{resi}')
                continue
            _=mapped.setdefault(chain,[])
            _.append(mapped_resi)
    if missing:
        raise ValueError(f'failed to map {label} residues onto target indices: {missing}')
    return mapped


class BolzGenSampler(BaseStep):
    def __init__(self,
        settings:GlobalSettings,
        ):
        '''
        Note: difference compared to ColabDesign:
        1. range provided by min/max value of `binder_settings.binder_lengths`
        2. use `full_target_pdb` instead of `starting_pdb`
        3. num_designs=len(bs.binder_lengths) * len(bs.helix_values) * len(bs.global_seed)
        '''
        super().__init__(settings)
        self._used_columns=(
            'design_ptm','filter_rmsd','interaction_pae','iptm','ptm',
            'plip_hbonds_refolded','delta_sasa_refolded','design_chain_hydrophobicity',
            'loop','helix','sheet',
            'liability_score','liability_violations_summary',
            )

    def _gen_sub(self):
        adv=self.settings.adv
        full_pdb=self.settings.target_settings.full_target_pdb
        target_chain=self.settings.target_settings.full_target_chain

        objects=['ori']
        cmd.load(full_pdb,'ori')
        cmd.remove('not (alt "" or alt A)')
        no_organic_purify('ori')

        id_map={}
        for c in target_chain.split(','):
            l_=[]
            cmd.iterate(f'ori and name CA and chain {c} ','l_.append(resi)',space={'l_':l_})
            id_map[c]={i:str(j+1) for j,i in enumerate(l_)}

        # hotspot params
        hotspot_str=self.settings.target_settings.target_hotspot_residues
        not_binding_spec=adv.get('not_binding',None)
        epitope_range:str=adv.setdefault('epitope_range','full')
        epitope_strategy:str=adv.setdefault('epitope_strategy','none') 
        binding_types=[]
        if not_binding_spec not in (None,''):
            not_binding_dict=_map_chain_residue_spec(
                _load_chain_residue_spec(not_binding_spec),
                id_map,
                'not_binding',
                )
            for k,v in not_binding_dict.items():
                if v == 'all':
                    binding_types.append({'chain':{'id':k,'not_binding':'all'}})
                else:
                    binding_types.append({'chain':{'id':k,'not_binding':','.join(v)}})
        if hotspot_str:
            hotspots:List[Tuple[str,str]]=[(i[0],i[1:]) for i in hotspot_str.split(',')]
            hsp_dict={}
            for i in hotspots:
                _=hsp_dict.setdefault(i[0],[])
                _.append(id_map[i[0]][i[1]])
            # Keep binding entries after not_binding so explicit hotspot sites win on overlap.
            binding_types.extend([{'chain':{'id':k,'binding':','.join(v)}} for k,v in hsp_dict.items()])
        else:
            assert epitope_range=='full', 'epitope-only strategy attempted without hotspots specified.'

        # binder params
        cyclic=adv.setdefault('cyclize_peptide',False)
        l_=self.settings.binder_settings.binder_lengths
        binder_length_range=f'{min(l_)}..{max(l_)}'
        binder_d= {'id': self.settings.target_settings.new_binder_chain,
         'sequence': binder_length_range, 'cyclic': cyclic}
        
        # epitope params
        if epitope_range =='full':
            chain=[{'chain':{'id':c}} for c in target_chain.split(',')]
        else:
            try:
                epitope_range=int(epitope_range)
            except:
                raise ValueError(f'invalid epitope range: {epitope_range}')
            if epitope_strategy == 'top-k':
                cmd.create('target',f'ori and (chain {target_chain})')
                other_res_sorted_list=sort_distance_to_hotspots('target',hotspots)
                cmd.delete('target')
                top_k_epitope('ori',hotspots,other_res_sorted_list,k=epitope_range) # select {obj}_top{k}
                chain=[]
                for c in target_chain.split(','):
                    r=[]
                    cmd.iterate(f'chain {c} and name CA and ori_top{epitope_range}','r.append(resi)',space={'r':r})
                    chain.append({'chain':{'id':c,'res_index':','.join([id_map[c][j] for j in r])}})

            elif epitope_strategy == 'dist-range':
                hotspots_to_seg_surf('ori',hotspots,
                    opt_obj=f'seg_{epitope_range}',vicinity=epitope_range)
                objects.append(f'seg_{epitope_range}')
                chain=[]
                for c in target_chain.split(','):
                    r=[]
                    cmd.iterate(f'chain {c} and name CA and seg_{epitope_range}','r.append(resi)',space={'r':r})
                    chain.append({'chain':{'id':c,'res_index':','.join([id_map[c][j] for j in r])}})
            else:
                raise ValueError(f'invalid `epitope_strategy`:{epitope_strategy}')

        file_d={'path': str(Path(full_pdb).absolute()),
                'include': chain,
                'structure_groups': 'all',
                }
        if binding_types:
            file_d['binding_types']=binding_types
        yaml_opt={'entities': 
            [{'file': file_d},
            {'protein': binder_d
            }]}
        Path(self.design_path).mkdir(exist_ok=True,parents=True)
        yaml_recipe=f'{self.design_path}/{self.design_name}-boltz.yaml'
        with open(yaml_recipe,'w') as f:
            f.write(yaml.safe_dump(yaml_opt,indent=2,sort_keys=False))
        cmd.save(yaml_recipe.replace('.yaml','.pse'))
        for i in objects:
            cmd.delete(i)

    def _run(self,overwrite:bool=False):
        '''
        Note: overwrite is internally controlled by 
        '''
        adv=self.settings.adv
        bs=self.settings.binder_settings
        boltzdesign_stem=adv.setdefault('boltzdesign_stem','boltzdesign')
        env=adv.setdefault('boltzgen_env','boltzgen')
        if '/' in env:
            flag='-p'
        else:
            flag='-n'
        if max(bs.binder_lengths) >=40:
            protocol_='protein-anything'
        else:
            protocol_='peptide-anything'
        num_designs=len(bs.binder_lengths) * len(bs.helix_values) * len(bs.random_seeds)
        yaml_recipe=f'{self.design_path}/{self.design_name}-boltz.yaml'
        cmds=['conda','run',flag,env, 
            'boltzgen','run',yaml_recipe, 
            '--output',bs.design_path+f'/{boltzdesign_stem}',
            '--protocol',protocol_, '--num_designs', str(num_designs), '--budget', str(num_designs)]
        if not overwrite:
            cmds.append('--reuse')
        run(cmds)
        self.settings.target_settings.starting_pdb=f'{bs.design_path}/{boltzdesign_stem}/{bs.binder_name}-boltz.cif'

    def _collect_results(self,batch:DesignBatch,overwrite:bool=False):
        assert self.metrics is not None
        bs=self.settings.binder_settings
        adv=self.settings.adv
        boltzdesign_stem=adv.setdefault('boltzdesign_stem','boltzdesign')
        for i,s in self.metrics.iterrows():
            if s['id'] not in batch.records or overwrite:
                design_pdb_=glob(f'{bs.design_path}/{boltzdesign_stem}/final_ranked_designs/final_*_designs/*{s["id"]}.cif')
                if len(design_pdb_)>0:
                    design_pdb=design_pdb_[0]
                    record=DesignRecord(id=s['id'],sequence=s['designed_chain_sequence'])
                    record.pdb_files[f'{self.metrics_prefix}design']=design_pdb
                    for _c in self._used_columns:
                        record.set_metrics(f'{self.metrics_prefix}{_c}',s[_c])
                    #TODO read per-res pLDDT from cif.
                    batch.add_record(record)
                    batch.save_record(s['id'])
        return batch
    
    def process_batch(self,overwrite:bool=False,metrics_prefix:str|None=None,
        pdb_purge_stem=None,pdb_to_take=None, # no use
        ):
        bs=self.settings.binder_settings
        metrics_stem:str=self.settings.adv.setdefault('metrics_stem','metrics')
        if metrics_prefix is not None:
            self.config_metrics_prefix(metrics_prefix)

        metrics_dir=Path(bs.design_path)/metrics_stem
        if not overwrite and metrics_dir.exists():
            batch=DesignBatch.from_cache(metrics_dir).filter(lambda x:bs.binder_name in x.id)
        else:
            batch=DesignBatch(Path(bs.design_path)/metrics_stem)

        yaml_recipe=Path(f'{self.design_path}/{self.design_name}-boltz.yaml')
        if not yaml_recipe.exists() or overwrite:
            self._gen_sub()

        if self.metrics is None:
            self._run(overwrite)
        
        batch=self._collect_results(batch,overwrite)
        return batch
        
    @property
    def params_to_take(self)->Tuple[str,...]:
        '''
        epitope_range: "full" or a int number
        epitope_strategy: "none" or "top-k" or "dist-range" 
        '''
        ret=['epitope_range','epitope_strategy']
        return tuple(ret)
        
    @property
    def name(self):
        return 'boltzgen'
    
    def process_record(self):
        raise NotImplementedError(f'Run {self.name} in Batches!')

    @property
    def design_path(self):
        return self.settings.binder_settings.design_path

    @property
    def pdb_to_add(self):
        return tuple([f'{self.metrics_prefix}:design'])
    
    @property
    def design_name(self):
        return self.settings.binder_settings.binder_name
    
    @property
    def metrics_to_add(self):
        """
        BioPhysical Annotations: plip_hbonds_refolded,delta_sasa_refolded, design_chain_hydrophobicity
        SS-Struct Annotations: loop, helix, sheet
        Liability score components assessing sequence-level developability risks:
            - ProtTryp: Proteolytic trypsin cleavage liability; indicates susceptibility to trypsin-mediated proteolysis at Lys/Arg sites, potentially reducing in vivo and in vitro stability.
            - DPP4: Dipeptidyl peptidase-4 cleavage liability; reflects risk of N-terminal dipeptide truncation by DPP4, often leading to rapid degradation and shortened half-life of peptides.
            - MetOx: Methionine oxidation liability; captures the propensity for oxidative modification of methionine residues, which can impair stability, activity, and batch consistency.
            - TrpOxNTCycl: Tryptophan oxidation and N-terminal cyclization liability; represents risks of Trp oxidation and N-terminal cyclization (e.g., pyroglutamate formation), contributing to chemical heterogeneity.
            - AspCleave: Aspartate-mediated backbone cleavage liability; indicates susceptibility to non-enzymatic peptide bond cleavage near Asp residues under stress or storage conditions.
            - AspBridge: Aspartate isomerization/bridging liability; reflects risk of Asp/isoAsp formation or abnormal intramolecular rearrangements, potentially altering backbone geometry and bioactivity.
            - HydroPatch: Hydrophobic surface patch liability; measures exposed hydrophobic clustering that can promote aggregation, poor solubility, and reduced manufacturability.
        """
        return tuple([f'{self.metrics_prefix}{i}' for i in self._used_columns])
        
    @property
    def metrics(self):
        '''
        trick: del `self._metrics` to reload.
        '''
        if getattr(self,'_metrics', None) is None:
            boltzdesign_stem=self.settings.adv.setdefault('boltzdesign_stem','boltzdesign')
            metrics_path=f'{self.design_path}/{boltzdesign_stem}/final_ranked_designs/all_designs_metrics.csv'
            if Path(metrics_path).exists():
                self._metrics=pd.read_csv(metrics_path)
            else:
                warn(f'metrics not found at {metrics_path}')
                self._metrics = None
        return self._metrics

    @property
    def _default_metrics_prefix(self):
        return f'boltzgen{NEST_SEP}'
    
    @property
    def params_to_take(self):
        ret=[f'{self.name}-prefix','boltzdesign_stem','metrics_stem','boltzgen_env',
             'epitope_range','epitope_strategy','cyclize_peptide','not_binding',]
        return tuple(ret)
            
        
