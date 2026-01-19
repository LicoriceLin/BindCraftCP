from ..steps import (Hallucinate,Filter,Refold,Graft,AnnotRMSD,AnnotGyration,
    AnnotSurf,MPNN,AnnotPolarOccupy,AnnotPTM,AnnotBCAux,AnnotPI,Relax)

from ..utils import (
    TargetSettings,GlobalSettings,AdvancedSettings,
    FilterSettings,BinderSettings,NEST_SEP,
    DesignRecord,DesignBatch
    )
from ..utils.settings import BaseSettings,dataclass
from .base_pipeline import BasePipeline,_dir_path
import json
import sys
from pathlib import Path
from typing import Dict,Any,Tuple
from functools import partial


class RemoveC(BasePipeline):
    def _init_steps(self):
        settings=self.settings
        settings.target_settings.full_target_chain='A'
        settings.target_settings.full_target_chain='B'

        self.filter=Filter(settings)
        self.refold=Refold(settings)       
        self.mpnn=MPNN(settings)
        self.annot_rmsd=AnnotRMSD(settings) 
        self.annot_surf=AnnotSurf(settings)
        self.annot_polar=AnnotPolarOccupy(settings)
        self.annot_gyr=AnnotGyration(settings)
        self.relax=Relax(self.settings)
        self.annot_aux=AnnotBCAux(self.settings)
        self.annot_pi=AnnotPI(self.settings)

        self._config_steps()
        self._save_settings('rmC')

    def _config_steps(self):
        adv=self.settings.adv

        adv['mpnn_binder_chain'],adv['mpnn_target_chain']='B','A'
        adv['mpnn_bias_recipe']='epitopecraft/selfs/config/mpnn-rmC-recipe.json'
        adv['mpnn-prefix']='rmC:'
        adv['max_mpnn_sequences']=1
        adv['mpnn-pdb-input']='refold:best'

        self.refold.config_pdb_input_key('template')
        self.refold.config_pdb_purge(adv.setdefault('refold_stem','refold'))

        self.annot_rmsd.config_params({'target_sel': 'chain A',
        'target_rms_sel': 'chain B'})
        self.annot_surf.config_pdb_input_key({'pdb_key': 'template', 'binder_chain': 'B'})

    def run(self,batch:DesignBatch):

        batch=self.mpnn.process_batch(batch)
        def rename_temp(x:DesignRecord):
            x.pdb_files['template']=x.pdb_files.pop('refold:best')
        batch.apply(lambda x: rename_temp(x))
        batch.save_records()

        batch=self.refold.process_batch(batch)

        self.annot_rmsd.process_batch(batch)
        self.annot_gyr.process_batch(batch)
        self.filter.set_recipe("after:refold").process_batch(batch) # no removal
        self.annot_surf.process_batch(batch)
        self.relax.process_batch(batch)
        self.annot_polar.process_batch(batch)
        self.annot_aux.process_batch(batch)
        self.annot_pi.process_batch(batch)
        self.filter.set_recipe("final").process_batch(batch)
        self._save_settings('rmC')
        return batch

