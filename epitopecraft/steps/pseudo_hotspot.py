from .basestep import *
from epitopecraft.steps.scorer.pymol_utils import *
from pymol import cmd

class AnnotHotspot(BaseStep):
    def __init__(self,
        settings:GlobalSettings,
        ):
        super().__init__(settings)


    
    @property
    def name(self)->str:
        'pseudo-hotspot'

    @property
    def _default_pdb_input_key(self)->Dict[str,str]:
        ts=self.settings.target_settings
        # {'pdb_key':'refold','binder_chain':'B','target_chain':'A'}
        return {'pdb_key':'template','binder_chain':ts.new_binder_chain,'target_chain':ts.full_target_chain}
    
    def config_pdb_input_key(self,pdb_key:str|None=None,binder_chain:str|None=None,target_chain:str|None=None,
            pdb_to_take:Dict[str,str]|None=None):
        '''
        '''
        if pdb_to_take is None:
            pdb_to_take=self._default_pdb_input_key
        if pdb_key is not None:
            pdb_to_take['pdb_key']=pdb_key
        if binder_chain is not None:
            pdb_to_take['binder_chain']=binder_chain
        if target_chain is not None:
            pdb_to_take['target_chain']=target_chain
        super().config_pdb_input_key(pdb_to_take)

    @property
    def pdb_to_take(self)->Dict[str,str]:
        '''
        Hotspots is usually defined on the full target pdb
        default: {'pdb_key':'template','binder_chain':ts.new_binder_chain,'target_chain':ts.full_target_chain}
        '''
        if not hasattr(self,'_pdb_to_take'):
            self.config_pdb_input_key()
        return self._pdb_to_take
    
    def process_record(self, input: DesignRecord):
        t_=self.pdb_to_take
        cmd.load(input.pdb_files[t_['pdb_key']],input.id)
        hotspots:List[Tuple[str,str]]=hotspots_by_ligand(input.id,t_['target_chain'],t_['binder_chain'])['hotspots']
        input.set_metrics(f'{self.metrics_prefix}hotspots',hotspots)
        cmd.delete(input.id)
        cmd.delete('complex')
        cmd.delete('target')

    def merge_hotspots(self,batch:DesignBatch):
        consensus_ratio=self.settings.adv.setdefault('hotspot:consensus_ratio',0.8)
        threshold=int(len(batch)*consensus_ratio)
        hotspots_count={}
        for x in batch:
            for h_ in x.get_metrics('graft:hotspots',[]):
                p_=hotspots_count.get(h_,0)
                hotspots_count[h_]=p_+1
        selected_hotspots=[k for k,v in hotspots_count.items() if v>=threshold]
        batch.log({'selected_hotspots':selected_hotspots})
        return {'hotspots_count':hotspots_count,'selected_hotspots':selected_hotspots}
    
    @property
    def _default_metrics_prefix(self)->str:
        return f'{self.pdb_to_take["pdb_key"]}{NEST_SEP}'
    
    @property
    def metrics_to_add(self):
        return tuple([f'{self.metrics_prefix}hotspots'])