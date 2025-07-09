from .basestep import *
from colabdesign import mk_afdesign_model
from pymol import cmd
import tempfile

class Refold(BaseStep):
    def __init__(self,settings:GlobalSettings):
        super().__init__(settings)
        self.templated=settings.adv.get('templated',False)
    
    # def _templated_sanity_check(self):
    #     target_setting=self.settings.target_settings
    #     assert target_setting.full_target_pdb and target_setting.full_target_chain
    #     advanced_settings=self.settings.adv
    #     use_initial_guess=advanced_settings.get("predict_initial_guess",False)
    #     use_initial_atom_pos=advanced_settings.get("predict_bigbang",False)
    #     assert use_initial_atom_pos or use_initial_guess
    # def refold()
    # @property
    # def templated(self):
    #     c_model= self.complex_prediction_model
    #     if c_model._args['use_initial_guess'] or c_model._args['use_initial_atom_pos']:
    #         return True
    #     else:
    #         return False
    
    def graft_full_target(self,record:DesignRecord):
        raise NotImplementedError
    
    def config_complex_model(self,record:DesignRecord):
        '''
        record.pdb_files['template'] is conserved for path to template pdb
        '''
        c_model=self.complex_prediction_model
        advanced_settings,s=self.settings.adv,self.settings
        binder_len=len(record.sequence)
        if self.templated:
            try:
                template_pdb=record.pdb_files['template']
            except:
                raise ValueError('assign template pdb to `record.pdb_files["template"]`')
            
            if getattr(self,'current_template_pdb','')!=template_pdb:
                self.current_template_pdb=template_pdb
                c_model.prep_inputs(
                    pdb_filename=template_pdb, 
                    chain=s.target_settings.full_target_chain, 
                    binder_chain=s.target_settings.full_binder_chain, 
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
        
    def refold(self,record:DesignRecord,prefix='refold-')->DesignRecord:
        c_model,m_model=self.complex_prediction_model,self.binder_prediction_model
        binder_sequence=record.sequence
        advanced_settings,s=self.settings.adv,self.settings
        prediction_models = [0,1] if advanced_settings['use_multimer_design'] else [0,1,2,3,4]

        for model_num in prediction_models:
            refold_id_c=f'{prefix}multimer-{model_num+1}'
            if refold_id_c not in record.metrics:
                c_model.predict(seq=binder_sequence, models=[model_num], 
                    num_recycles=advanced_settings["num_recycles_validation"], 
                    verbose=False,seed=s.binder_settings.global_seed)
                record.pdb_strs[refold_id_c]=c_model.save_pdb(None,get_best=False)
                metrics={k:c_model.aux['log'][v] for k,v in 
                    {'pLDDT':'plddt','pTM':'ptm',
                     'i-pTM':'i_ptm','pAE':'pae',
                     'i-pAE':'i_pae'}.items()}
                record.metrics[refold_id_c]=metrics
            
        for model_num in prediction_models:
            refold_id_m=f'{prefix}monomer-{model_num+1}'
            if refold_id_m not in metrics:
                m_model.predict(models=[model_num], 
                    num_recycles=advanced_settings["num_recycles_validation"], verbose=False,
                    seed=s.binder_settings.global_seed)
                record.pdb_strs[refold_id_m]=c_model.save_pdb(None,get_best=False)
                metrics={k:m_model.aux['log'][v] for k,v in 
                    {'pLDDT':'plddt','pTM':'ptm',
                     'pAE':'pae'}.items()}
                record.metrics[refold_id_m]=metrics
        
        return record
    
    def purge_record(self,record:DesignRecord,prefix:str='refold-'):
        assert self.pdb_purge_dir is not None
        advanced_settings,s=self.settings.adv,self.settings
        prediction_models = [0,1] if advanced_settings['use_multimer_design'] else [0,1,2,3,4]
        for model_num in prediction_models:
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
            d={k:(len(v.sequence),v.pdb_files['template']) for k,v in input.records.items()}
        return sorted(d,key=lambda k:d[k])

    def __call__(self, input:DesignBatch,
        prefix:str='refold-',pdb_purge_stem:Optional[str]=None):
        self.config_pdb_purge(pdb_purge_stem)
        for design_id in self.sort_batch(input):
            record=input.records[design_id]
            self.config_complex_model(record)
            self.config_monomer_model(record)
            self.refold(record,prefix)
            if self.pdb_purge_dir is not None:
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

    def graft_binder(self,record:DesignRecord,ori_key:str='halu',graft_key='template'):
        with tempfile.TemporaryDirectory() as tdir:
            if ori_key in record.pdb_files:
                ori_pdb=record.pdb_files[ori_key]
            elif ori_key in record.pdb_strs:
                record.cache_pdb(ori_key,Path(tdir)/'ori.pdb')
                ori_pdb=str(Path(tdir)/'ori.pdb')
            else:
                raise ValueError(f'no ori_pdb_key: {ori_key}')
            
            if self.pdb_purge_dir is not None:
                out_pdb=self.pdb_purge_dir/f'{record.id}.pdb'
            else:
                out_pdb=Path(tdir)/f'{record.id}.pdb'

            if not out_pdb.exists():
                _graft_binder(ori_pdb,self.ori_binder_chain,
                    self.graft_target,self.graft_chain,out_pdb)
            
            if self.pdb_purge_dir is not None:
                record.pdb_files[graft_key]=str(out_pdb)
            else:
                record.pdb_strs[graft_key]=open(out_pdb,'r').read()
        return record
    
    def __call__(self, input:DesignBatch, pdb_purge_stem:Optional[str]=None,
            ori_key:str='halu',graft_key='template'):
        self.config_pdb_purge(pdb_purge_stem)
        for records_id,record in input.records.items():
            self.graft_binder(record,ori_key,graft_key)
        input.save_records()
        return input
    
        # return super().__call__(input)
                     
def _graft_binder(ori_pdb:str,ori_binder_chain:str, graft_target:str,graft_chain:str,out_pdb:str):
    '''
    make sure traj/rescore pdb are pre-aligned.
    '''
    cmd.load(ori_pdb,'ori_pdb')
    cmd.load(graft_target,'graft_pdb')
    cmd.select('to_write',f" (ori_pdb and (chain {ori_binder_chain})) or (graft_pdb and (chain {graft_chain}))")
    cmd.save(out_pdb,'to_write')
    cmd.delete('ori_pdb')
    cmd.delete('rescorepdb')