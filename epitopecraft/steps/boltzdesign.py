from .basestep import *
from epitopecraft.steps.scorer.pymol_utils import *
from pymol import cmd
import yaml
from subprocess import run
from warnings import warn
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
        epitope_range:str=adv.setdefault('epitope_range','full')
        epitope_strategy:str=adv.setdefault('epitope_strategy','none') 
        if hotspot_str:
            hotspots:List[Tuple[str,str]]=[(i[0],i[1:]) for i in hotspot_str.split(',')]
            hsp_dict={}
            for i in hotspots:
                _=hsp_dict.setdefault(i[0],[])
                _.append(id_map[i[0]][i[1]])
            binding_types=[{'chain':{'id':k,'binding':','.join(v)}} for k,v in hsp_dict.items()]
        else:
            binding_types=[]
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

    def _run(self):
        adv=self.settings.adv
        bs=self.settings.binder_settings
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
        
        run(['conda','run',flag,env, 
            'boltzgen','run',yaml_recipe, 
            '--output',bs.design_path+'/boltzdesign',
            '--protocol',protocol_, '--num_designs', str(num_designs), '--budget', str(num_designs), '--reuse'])
        self.settings.target_settings.starting_pdb=f'{bs.design_path}/boltzdesign/{bs.binder_name}-boltz.cif'

    def collect_results(self):
        assert self.metrics is not None
        

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
    def design_name(self):
        return self.settings.binder_settings.binder_name

    @property
    def metrics(self):
        '''
        trick: del `self._metrics` to reload.
        '''
        if getattr(self,'_metrics', None) is None:
            metrics_path=f'{self.design_path}/boltzdesign/final_ranked_designs/all_designs_metrics.csv'
            if Path(metrics_path).exists():
                self._metrics=pd.read_csv(metrics_path)
            else:
                warn(f'metrics not found at {metrics_path}')
                self._metrics = None
        return self._metrics

            
        


        
