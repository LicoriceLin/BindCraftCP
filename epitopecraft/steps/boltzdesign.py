from .basestep import *
from epitopecraft.steps.scorer.pymol_utils import *
from pymol import cmd
import yaml
from subprocess import run

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
        design_path=self.settings.binder_settings.design_path
        design_name=self.settings.binder_settings.binder_name
        Path(design_path).mkdir(exist_ok=True,parents=True)
        self.yaml_recipe=f'{design_path}/{design_name}-boltz.yaml'

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
        hotspots:List[Tuple[str,str]]=[(i[0],i[1:]) for i in hotspot_str.split(',')]
        hsp_dict={}
        for i in hotspots:
            _=hsp_dict.setdefault(i[0],[])
            _.append(id_map[i[0]][i[1]])
        binding_types=[{'chain':{'id':k,'binding':','.join(v)}} for k,v in hsp_dict.items()]

        # binder params
        cyclic=adv.setdefault('cyclize_peptide',False)
        l_=self.settings.binder_settings.binder_lengths
        binder_length_range=f'{min(l_)}..{max(l_)}'
        binder_d= {'id': self.settings.target_settings.new_binder_chain,
         'sequence': binder_length_range, 'cyclic': cyclic}
        
        # epitope params
        epitope_range:str=adv.setdefault('epitope_range','full')
        if epitope_range =='full':
            chain=[{'chain':{'id':c}} for c in target_chain.split(',')]
        else:
            try:
                epitope_range=int(epitope_range)
            except:
                raise ValueError(f'invalid epitope range: {epitope_range}')
            epitope_strategy:str=adv.setdefault('epitope_strategy','top-k') 
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

        yaml_opt={'entities': 
            [{'file': 
                {'path': str(Path(full_pdb).absolute()),
                'include': chain,
                'structure_groups': 'all',
                'binding_types':binding_types
                }
                },
            {'protein': binder_d
            }]
                }
        design_path=self.settings.binder_settings.design_path
        design_name=self.settings.binder_settings.binder_name
        Path(design_path).mkdir(exist_ok=True,parents=True)
        with open(self.yaml_recipe,'w') as f:
            f.write(yaml.safe_dump(yaml_opt,indent=2,sort_keys=False))
        cmd.save(self.yaml_recipe.replace('.yaml','.pse'))
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

        
        run(['conda','run',flag,env, 
            'boltzgen','run',self.yaml_recipe, 
            '--output',bs.design_path+'/boltzdesign',
            '--protocol',protocol_, '--num_designs', str(num_designs), '--budget', str(num_designs), '--reuse'])


    def collect_results(self):
        pass

    @property
    def params_to_take(self)->Tuple[str,...]:
        '''
        epitope_range: "full" or a int number
        epitope_strategy: top-k or dist-range
        '''
        ret=['epitope_range','epitope_strategy']
        return tuple(ret)
        
    @property
    def name(self):
        return 'boltzgen'
    
    def process_record(self):
        pass

        


        
