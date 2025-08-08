from .basestep import *
from colabdesign import mk_afdesign_model
from pymol import cmd
import tempfile
from string import ascii_uppercase
from tqdm import tqdm
class Refold(BaseStep):
    def __init__(self,settings:GlobalSettings):
        self.templated=settings.adv.get('templated',False)
        super().__init__(settings)
        
    
    @property
    def name(self):
        return 'refold'
    
    @property
    def metrics_to_add(self):
        prefix=self.metrics_prefix
        ret=[f'{prefix}multimer-{n+1}' for n in self.prediction_models]
        ret.extend([f'{prefix}monomer-{n+1}' for n in self.prediction_models])
        return tuple(ret)
    
    @property
    def pdb_to_add(self):
        return self.metrics_to_add
    
    def config_pdb_input_key(self, pdb_to_take = None):
        if pdb_to_take is None:
            default='template' if self.templated else 'halu'
            self._pdb_to_take=self.settings.adv.get('template-pdb-key',default)
        else:
            self._pdb_to_take=pdb_to_take
    
    def config_complex_model(self,record:DesignRecord):
        '''
        '''
        c_model=self.complex_prediction_model
        advanced_settings,s=self.settings.adv,self.settings
        binder_len=len(record.sequence)
        if self.templated:
            try:
                template_pdb=record.pdb_files[self.pdb_to_take]
            except:
                raise ValueError(
                    f'template-pdb-key `{self.pdb_to_take}` '
                    'not found in `record.pdb_files`')
            
            if getattr(self,'current_template_pdb','')!=template_pdb:
                self.current_template_pdb=template_pdb
                c_model.prep_inputs(
                    pdb_filename=template_pdb, 
                    chain=s.target_settings.full_target_chain, 
                    binder_chain=s.target_settings.new_binder_chain, 
                    binder_len=binder_len, 
                    use_binder_template=True, 
                    rm_target_seq=advanced_settings["rm_template_seq_predict"],
                    rm_target_sc=advanced_settings["rm_template_sc_predict"], 
                    rm_template_ic=advanced_settings.get("rm_template_ic_predict",True),
                    seed=s.binder_settings.global_seed)
        else:
            if getattr(c_model,'_binder_len',0) !=binder_len:
                c_model.prep_inputs(pdb_filename=s.target_settings.starting_pdb, 
                        chain=s.target_settings.chains, 
                        binder_len=binder_len, 
                        rm_target_seq=advanced_settings["rm_template_seq_predict"],
                        rm_target_sc=advanced_settings["rm_template_sc_predict"],
                        seed=s.binder_settings.global_seed)
        if advanced_settings.get('cyclize_peptide',False):
            add_cyclic_offset(c_model, offset_type=2)
            
    def config_monomer_model(self,record:DesignRecord):
        m_model=self.binder_prediction_model
        binder_len=len(record.sequence)
        if getattr(m_model,'_binder_len',0) !=binder_len:
            m_model.prep_inputs(length=binder_len)
        
    @property
    def prediction_models(self):
        return [0,1] if self.settings.adv['use_multimer_design'] else [0,1,2,3,4]
    
    def refold(self,record:DesignRecord)->DesignRecord:
        prefix=self.metrics_prefix
        c_model,m_model=self.complex_prediction_model,self.binder_prediction_model
        binder_sequence=record.sequence
        advanced_settings,s=self.settings.adv,self.settings
        
        for model_num in self.prediction_models:
            refold_id_c=f'{prefix}multimer-{model_num+1}'
            if refold_id_c not in record.metrics:
                c_model.predict(seq=binder_sequence, models=[model_num], 
                    num_recycles=advanced_settings["num_recycles_validation"], 
                    verbose=False,seed=s.binder_settings.global_seed)
                record.pdb_strs[refold_id_c]=c_model.save_pdb(None,get_best=False)
                metrics={refold_id_c+':'+k:c_model.aux['log'][v] for k,v in 
                    {'pLDDT':'plddt','pTM':'ptm',
                     'i-pTM':'i_ptm','pAE':'pae',
                     'i-pAE':'i_pae'}.items()}
                record.update_metrics(metrics)
            
        for model_num in self.prediction_models:
            refold_id_m=f'{prefix}monomer-{model_num+1}'
            if refold_id_m not in metrics:
                m_model.predict(models=[model_num], 
                    num_recycles=advanced_settings["num_recycles_validation"], verbose=False,
                    seed=s.binder_settings.global_seed)
                record.pdb_strs[refold_id_m]=c_model.save_pdb(None,get_best=False)
                metrics={refold_id_m+':'+k:m_model.aux['log'][v] for k,v in 
                    {'pLDDT':'plddt','pTM':'ptm',
                     'pAE':'pae'}.items()}
                record.update_metrics(metrics)
        
        return record
    
    def purge_record(self,record:DesignRecord):
        '''
        create self.pdb_purge_dir/{multimer,monomer}
        purge refold structure from different af2 model to it.
        '''
        if self.pdb_purge_dir is not None:
            prefix = self.metrics_prefix
            for model_num in self.prediction_models:
                refold_id_c=f'{prefix}multimer-{model_num+1}'
                if refold_id_c in record.pdb_strs:
                    record.purge_pdb(refold_id_c,
                        self.pdb_purge_dir/'multimer'/f'{record.id}-{model_num+1}.pdb')

                refold_id_m=f'{prefix}monomer-{model_num+1}'
                if refold_id_m in record.pdb_strs:
                    record.purge_pdb(refold_id_m,
                        self.pdb_purge_dir/'monomer'/f'{record.id}-{model_num+1}.pdb')

    def config_pdb_purge(self, pdb_purge_stem = None):
        super().config_pdb_purge(pdb_purge_stem)
        if self.pdb_purge_dir is not None:
            cdir,mdir=self.pdb_purge_dir/'multimer',self.pdb_purge_dir/'monomer'
            cdir.mkdir(parents=True,exist_ok=True)
            mdir.mkdir(parents=True,exist_ok=True)

    def sort_batch(self,input:DesignBatch)->List[str]:
        if not self.templated:
            d={k:len(v.sequence) for k,v in input.records.items()}
            
        else:
            d={k:(len(v.sequence),v.pdb_files[self.pdb_to_take]) for k,v in input.records.items()}
        return sorted(d,key=lambda k:d[k])

    def check_processed(self,input: DesignRecord)->bool:
        prefix=self.metrics_prefix
        for model_num in self.prediction_models:
            refold_id_c=f'{prefix}multimer-{model_num+1}'
            if not input.has_pdb(refold_id_c):
                return False
            refold_id_m=f'{prefix}monomer-{model_num+1}'
            if not input.has_pdb(refold_id_m):
                return False
        return True
    
    def process_record(self, input: DesignRecord)->DesignRecord:
        with self.record_time(input):
            self.config_complex_model(input)
            self.config_monomer_model(input)
            self.refold(input)
        return input    
    
    def process_batch(self, input:DesignBatch,
        pdb_purge_stem:Optional[str]=None,
        metrics_prefix:str|None=None,
        pdb_to_take:str=None,
        ):
        '''
        advanced settings:
            refold-template-pdb-key:str ='graft' 
            num_recycles_validation: int
            use_multimer_design: bool
            cyclize_peptide: bool
            rm_template_seq_predict: bool
            rm_template_sc_predict: bool
            rm_template_ic_predict: bool
            num_recycles_validation: int

        '''
        if pdb_purge_stem is not None:
            self.config_pdb_purge(pdb_purge_stem)
        if metrics_prefix is not None:
            self.config_metrics_prefix(metrics_prefix)
        if pdb_to_take is not None:
            self.config_pdb_input_key(pdb_to_take)
        for design_id in tqdm(self.sort_batch(input),desc=f'{self.name}:'):
            record=input.records[design_id]
            if not self.check_processed(record):
                self.process_record(record)
                self.purge_record(record)
                input.save_record(design_id)
        return input

        # return super().__call__(input)

    @property
    def complex_prediction_model(self):
        advanced_settings=self.settings.adv
        if getattr(self,'_complex_prediction_model',None) is None:
            use_initial_guess=advanced_settings.get("predict_initial_guess",False)
            use_initial_atom_pos=advanced_settings.get("predict_bigbang",False)
            self._complex_prediction_model = mk_afdesign_model(protocol="binder", 
                num_recycles=advanced_settings["num_recycles_validation"], 
                data_dir=advanced_settings["af_params_dir"], 
                use_multimer= not advanced_settings['use_multimer_design'], 
                use_initial_guess=use_initial_guess, 
                use_initial_atom_pos=use_initial_atom_pos)
        return self._complex_prediction_model
    
    @property
    def binder_prediction_model(self):
        advanced_settings=self.settings.adv
        if getattr(self,'_binder_prediction_model',None) is None:
            self._binder_prediction_model=mk_afdesign_model(
                protocol="hallucination", use_templates=False, initial_guess=False, 
                use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"], 
                data_dir=advanced_settings["af_params_dir"], use_multimer=not advanced_settings['use_multimer_design'])
        return self._binder_prediction_model
 
class Graft(BaseStep):
    def __init__(self, settings:GlobalSettings):
        super().__init__(settings)
        self.graft_target=self.settings.target_settings.full_target_pdb
        self.graft_chain=self.settings.target_settings.full_target_chain
        self.ori_binder_chain=self.settings.target_settings.full_binder_chain
        self.new_binder_chain=self.settings.target_settings.new_binder_chain
    @property
    def name(self):
        return 'graft'
    
    @property
    def _default_metrics_prefix(self):
        return 'template:'
    
    @property
    def pdb_to_add(self):
        return tuple([self.metrics_prefix.strip(NEST_SEP)])
    
    def config_pdb_input_key(self, pdb_to_take = None):
        if pdb_to_take is None:
            self._pdb_to_take=self.settings.adv.get('graft-ori-key','halu')
        else:
            self._pdb_to_take=pdb_to_take

    def graft_binder(self,record:DesignRecord,):
        ori_key=self.pdb_to_take
        prefix=self.metrics_prefix.strip(NEST_SEP)
        with tempfile.TemporaryDirectory() as tdir:
            if ori_key in record.pdb_files:
                ori_pdb=record.pdb_files[ori_key]
            elif ori_key in record.pdb_strs:
                record.cache_pdb(ori_key,Path(tdir)/'ori.pdb')
                ori_pdb=str(Path(tdir)/'ori.pdb')
            else:
                raise ValueError(f'no ori_pdb_key: {ori_key}')
            
            out_pdb=Path(tdir)/f'{record.id}.pdb'
            self.new_graft_chain=_graft_binder(ori_pdb,self.ori_binder_chain,
                self.graft_target,self.graft_chain,out_pdb,self.new_binder_chain)
            record.pdb_strs[prefix]=open(out_pdb,'r').read()
        return record

    def process_record(self,input: DesignRecord)->DesignRecord:
        self.graft_binder(input)
        # self.purge_record(input)
        return input

                     
def _graft_binder(ori_pdb:str,ori_binder_chain:str,
    graft_target:str,graft_chain:str,out_pdb:str,new_binder_chain:str):
    '''
    make sure traj/rescore pdb are pre-aligned.
    '''
    cmd.load(ori_pdb,'ori_pdb')
    cmd.load(graft_target,'graft_pdb')
    if new_binder_chain != ori_binder_chain:
        cmd.alter(f'ori_pdb and chain {ori_binder_chain}',f'chain="{new_binder_chain}"')
    cmd.create('to_write',f" (ori_pdb and (chain {new_binder_chain})) or (graft_pdb and (chain {graft_chain}))")
    cmd.save(out_pdb,'to_write')
    cmd.delete('ori_pdb')
    cmd.delete('graft_pdb')
    cmd.delete('to_write')