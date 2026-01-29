from ..steps import (Filter,RefoldOnly,AnnotRMSD,Relax)
from ..utils import (NEST_SEP,DesignRecord,DesignBatch)
from ..utils.settings import BaseSettings
from .base_pipeline import BasePipeline,_dir_path

from dataclasses import dataclass
from typing import Optional

##################################################################
##################################################################

@dataclass
class RefoldKeys:
    template: str = "template"
    relaxed: Optional[str] = None


class RefoldValidation(BasePipeline):

    '''
    Refolding only, prepared for large scale validation
    + RMSD
    - AF monomer refolding (NOT done yet)
    Write refold PDBs to <design_path>/<refold_stem>/ and metrics JSON to the DesignBatch cache_dir
    '''

    def _init_steps(self):
        settings = self.settings
        self.refold = RefoldOnly(settings)
        self.annot_rmsd = AnnotRMSD(settings)

        #self.annot_surf = AnnotSurf(settings) ## ?
        #self.annot_gyr=AnnotGyration(settings) ## ?
        #self.relax=Relax(settings)
        #self.annot_polar = AnnotPolarOccupy(settings)
        #self.annot_aux=AnnotBCAux(settings)
        #self.annot_pi=AnnotPI(settings)

        self._keys = RefoldKeys()
        self._config_steps()
        self._save_settings()

    def _config_steps(self):
        adv = self.settings.adv

        self.refold.config_pdb_input_key(self._keys.template) # 'template'
        self.refold.config_pdb_purge(adv.setdefault('refold_stem','refold')) ## TODO: maybe remove extra refold folder for storage
        best_refold_pdb = self.refold.metrics_prefix+"best"

        # RMSD: best refold vs. template
        self.annot_rmsd.config_pdb_input_key(
            pdb_to_take={"mobile": best_refold_pdb, "target":self._keys.template})
        self.annot_rmsd.config_metrics_prefix(best_refold_pdb+NEST_SEP) # 'refold:best:'

    def run(self, batch: DesignBatch) -> DesignBatch:

        self.refold.process_batch(batch)
        breakpoint()
        self.annot_rmsd.process_batch(batch)

        return batch





