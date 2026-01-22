from .basestep import *
import jax
import jax.numpy as jnp
from colabdesign import mk_afdesign_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from tqdm import tqdm
class Hallucinate(BaseStep):
    def __init__(self,
        settings:GlobalSettings,
        af_model:mk_afdesign_model|None=None,
        ):
        super().__init__(settings)
        self.init_afdesign_model(af_model)
        
    @property
    def name(self):
        return 'hallucinate'
    
    @property
    def _default_metrics_prefix(self):
        return f'halu{NEST_SEP}'
    
    @property
    def metrics_to_add(self):
        ret=[f'config:{i}' for i in ["helix",'length','seed']]
        prefix=self.metrics_prefix
        ret.extend([f'{prefix}pLDDT',f'{prefix}pTM',f'{prefix}i-pTM',f'{prefix}pAE',f'{prefix}i-pAE'])
        if self.settings.adv.setdefault('hallu_save_loss',False):
            ret.extend([f'{self.metrics_prefix}loss'])
        return tuple(ret)
    
    @property
    def pdb_to_add(self):
        return tuple([self.metrics_prefix.strip(NEST_SEP)])
    
    @property
    def params_to_take(self)->Tuple[str,...]:
        ret=[
            'cyclize_peptide',"4stage-mcmc",
            "af_params_dir","use_multimer_design","num_recycles_design","sample_models","hallu_save_loss",
            "omit_AAs","rm_template_seq_design","rm_template_sc_design",
            "weights_pae_intra","weights_plddt","weights_pae_inter","weights_con_intra","weights_con_inter",
            "intra_contact_number","inter_contact_number","intra_contact_distance","inter_contact_distance",
            "use_rg_loss","weights_rg","use_i_ptm_loss","weights_iptm","use_termini_distance_loss","weights_termini_loss",
            "soft_iterations","temporary_iterations","hard_iterations","greedy_iterations",
            "4stage-use-pssm","4stage_keep_best",
            "mcmc_half_life_ratio","t_init_mcmc",f'{self.name}-prefix','verb',
            ]
        return tuple(ret)

    def init_afdesign_model(
        self,
        af_model:mk_afdesign_model|None=None,
        )->mk_afdesign_model:
        advanced_settings=self.settings.adv
        if af_model is None:
            af_model = mk_afdesign_model(protocol="binder", debug=False, 
                data_dir=advanced_settings.setdefault('af_params_dir',''), 
                use_multimer=advanced_settings.setdefault("use_multimer_design",True), 
                num_recycles=advanced_settings.setdefault("num_recycles_design",1),
                best_metric='loss')
        else:
            af_model.restart(seed=self.settings.binder_settings.global_seed)
        self.af_model=af_model

    def config_afdesign_model(self,length:int,seed:int,helicity_value:float):
        advanced_settings=self.settings.adv
        target_settings=self.settings.target_settings
        rm_aa = advanced_settings.setdefault("omit_AAs",'') 
        if not rm_aa:
            rm_aa=None
        af_model=self.af_model
        af_model.prep_inputs(
            pdb_filename=target_settings.starting_pdb, 
            chain=target_settings.chains, 
            binder_len=length, 
            hotspot=target_settings.target_hotspot_residues, 
            seed=seed, 
            rm_aa=rm_aa,
            rm_target_seq=advanced_settings.setdefault("rm_template_seq_design",False), 
            rm_target_sc=advanced_settings.setdefault("rm_template_sc_design",False))
        self._current_design_seed=seed
        if advanced_settings.setdefault('cyclize_peptide',False):
            # To implement: multi-chain target, scaffolding.
            add_cyclic_offset(af_model, offset_type=2)
        ### Update weights based on specified settings
        af_model.opt["weights"].update({
            "pae":advanced_settings.setdefault("weights_pae_intra",0.4),
            "plddt":advanced_settings.setdefault("weights_plddt",0.1),
            "i_pae":advanced_settings.setdefault("weights_pae_inter",0.1),
            "con":advanced_settings.setdefault("weights_con_intra",1.0),
            "i_con":advanced_settings.setdefault("weights_con_inter",1.0),
            })
        af_model.opt["con"].update({
            "num":advanced_settings.setdefault("intra_contact_number",2),
            "cutoff":advanced_settings.setdefault("intra_contact_distance",14),
            "binary":False,"seqsep":9})
        af_model.opt["i_con"].update({
            "num":advanced_settings.setdefault("inter_contact_number",2),
            "cutoff":advanced_settings.setdefault("inter_contact_distance",20),
            "binary":False})

        ### additional loss functions
        if advanced_settings.setdefault("use_rg_loss",False):
            add_rg_loss(af_model, advanced_settings.setdefault("weights_rg",0.3))

        if advanced_settings.setdefault("use_i_ptm_loss",False):
            add_i_ptm_loss(af_model, advanced_settings.setdefault("weights_iptm",0.05))

        if advanced_settings.setdefault("use_termini_distance_loss",False):
            add_termini_distance_loss(af_model, advanced_settings.setdefault("weights_termini_loss",1.))
        add_helix_loss(af_model, helicity_value)

    def sample_trajectory(self,input:DesignRecord)->DesignRecord:
        prefix=self.metrics_prefix
        af_model=self.af_model
        advanced_settings=self.settings.adv
        design_models = [0,1,2,3,4] if advanced_settings.setdefault("use_multimer_design",True) else [0,1]
        verb=advanced_settings.setdefault('verb',1)
        if verb: 
            print("Running SGD-based hallucination...")
        af_model.design_logits(iters=advanced_settings.setdefault("soft_iterations",75), e_soft=1, models=design_models, num_models=1, 
            sample_models=advanced_settings.setdefault("sample_models",True), ramp_recycles=False, save_best=True,verbose=verb)
        af_model.design_soft(advanced_settings.setdefault("temporary_iterations",45), e_temp=1e-2, models=design_models, num_models=1,
            sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True,verbose=verb)
        af_model.design_hard(advanced_settings.setdefault("hard_iterations",5), temp=1e-2, models=design_models, num_models=1,
            sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True,verbose=verb)
        if advanced_settings.setdefault("greedy_iterations",15)>0:
            greedy_tries = math.ceil(af_model._binder_len * (advanced_settings.setdefault("greedy_percentage",5) / 100))
            af_model.clear_best()
            if advanced_settings.setdefault('4stage-use-pssm',False):
                seq_logits = None
            else:
                seq_logits = af_model.aux["seq"]["pssm"][0]
            if not advanced_settings.setdefault('4stage-mcmc',False):
                af_model.design_pssm_semigreedy(
                    soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                    num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True,
                    seq_logits=seq_logits,verbose=verb)
            else:
                af_model._design_mcmc(
                    steps=advanced_settings["greedy_iterations"], 
                    half_life=int(advanced_settings["greedy_iterations"]*advanced_settings.setdefault('mcmc_half_life_ratio',0.2)), 
                    seq_logits=seq_logits,T_init=advanced_settings.setdefault('t_init_mcmc',0.01), 
                    mutation_rate=greedy_tries, num_models=1, 
                    models=design_models,
                    sample_models=advanced_settings["sample_models"], save_best=True,verbose=verb)
            if advanced_settings.setdefault('4stage_keep_best',False):
                best = af_model._tmp["best"]
                af_model.aux, seq = best["aux"], jnp.array(best["aux"]["seq"]['input'])
                af_model.set_seq(seq=seq, bias=af_model._inputs["bias"])
                af_model._save_results(save_best=True, verbose=False)
        
        metrics={k:af_model.aux['log'][v] for k,v in {f'{prefix}pLDDT':'plddt',f'{prefix}pTM':'ptm',f'{prefix}i-pTM':'i_ptm',f'{prefix}pAE':'pae',f'{prefix}i-pAE':'i_pae'}.items()}
        # metrics.update({"helix":af_model.opt["weights"]["helix"],'length':af_model._binder_len,'seed':self._current_design_seed})
        if advanced_settings.setdefault('hallu_save_loss',False):
            metrics[f'{prefix}loss'] = af_model.aux['log']['loss']
        pdbstr=af_model.save_pdb()
        input.sequence=af_model.get_seq(get_best=False)[0]
        input.pdb_strs.update({prefix.strip(NEST_SEP):pdbstr})
        input.update_metrics(metrics)
        return input
        # return DesignRecord(id=design_id,sequence=af_model.get_seq(get_best=False)[0],
        #         pdb_strs={prefix.strip('-'):pdbstr},metrics=metrics)
    
    def log_trajectory(self):
        raise NotImplementedError

    def config_logger(self):
        raise NotImplementedError
    
    def process_record(self,input:DesignRecord)->DesignRecord:
        with self.record_time(input):
            m=input.metrics
            # breakpoint()
            self.config_afdesign_model(
                length=m['config']['length'],seed=m['config']['seed'],helicity_value=m['config']['helix'])
            record=self.sample_trajectory(input)
        return record

    def process_batch(self,
        batch_cache_stem:str='metrics',
        pdb_purge_stem:Optional[str]=None,
        metrics_prefix:str|None=None, 
        overwrite:bool=False,input=None)->DesignBatchSlice:
        '''
        cache_stem: path to cache output `DesignBatch`
        pdb_purge_stem: None: save with batch; else: purge pdb to this dir
        metrics_prefix: metrics_prefix > adv[f'{self.name}-prefix'] > self._default_metrics_prefix
        '''
        if pdb_purge_stem is not None:
            self.config_pdb_purge(pdb_purge_stem)
        if metrics_prefix is not None:
            self.config_metrics_prefix(metrics_prefix)
        binder_settings=self.settings.binder_settings
        batch=DesignBatch(Path(binder_settings.design_path)/batch_cache_stem)
        batch.set_overwrite(overwrite)
        if not overwrite:
            batch.load_records()

        global_id=0
        tot=len(binder_settings.binder_lengths)*len(binder_settings.random_seeds)*len(binder_settings.helix_values)
        to_fill=int(math.log10(tot)+1)
        pbar = tqdm(total=tot,desc=f'{self.name}: prefix={self.metrics_prefix}')
        ids=[]
        for length in binder_settings.binder_lengths:
            for seed in binder_settings.random_seeds:
                for helicity_value in binder_settings.helix_values:
                    design_id = binder_settings.binder_name+'-'+str(global_id).zfill(to_fill)
                    if batch.overwrite or design_id not in batch.records:
                        record=DesignRecord(id=design_id,sequence='')
                        record.update_metrics({"config:helix":helicity_value,'config:length':length,'config:seed':seed})
                        breakpoint() # break_m1
                        batch.add_record(self.process_record(record))
                        self.purge_record(record)
                        batch.save_record(design_id)
                    global_id+=1
                    ids.append(design_id)
                    pbar.update(1)
        # batch_slice=batch.filter(lambda x: x.id.startswith(binder_settings.binder_name))
        batch_slice=batch[ids]
        return batch_slice # batch #
    
# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)

# Define radius of gyration loss for colabdesign
def add_rg_loss(self:mk_afdesign_model, weight=0.1):
    '''add radius of gyration loss'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight

# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self:mk_afdesign_model, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}
    
    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

# add helicity loss
def add_helix_loss(self:mk_afdesign_model, weight=0):
    def binder_helicity(inputs, outputs):  
      if "offset" in inputs:
        offset = inputs["offset"]
      else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:,None] - idx[None,:]

      # define distogram
      dgram = outputs["distogram"]["logits"]
      dgram_bins = get_dgram_bins(outputs)
      mask_2d = np.outer(np.append(np.zeros(self._target_len), 
            np.ones(self._binder_len)), np.append(np.zeros(self._target_len), np.ones(self._binder_len)))

      x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
      if offset is None:
        if mask_2d is None:
          helix_loss = jnp.diagonal(x,3).mean()
        else:
          helix_loss = jnp.diagonal(x * mask_2d,3).sum() + (jnp.diagonal(mask_2d,3).sum() + 1e-8)
      else:
        mask = offset == 3
        if mask_2d is not None:
          mask = jnp.where(mask_2d,mask,0)
        helix_loss = jnp.where(mask,x,0.0).sum() / (mask.sum() + 1e-8)

      return {"helix":helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

# add N- and C-terminus distance loss
def add_termini_distance_loss(self:mk_afdesign_model, weight=0.1, threshold_distance=7.0):
    '''Add loss penalizing the distance between N and C termini'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        # termini_distance_loss = jax.nn.relu(deviation)
        termini_distance_loss = deviation+1
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

# plot design trajectory losses
def plot_trajectory(af_model:mk_afdesign_model, design_name, design_paths):
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'mpnn']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # Create a new figure for each metric
            plt.figure()

            loss = af_model.get_loss(metric)
            # Create an x axis for iterations
            iterations = range(1, len(loss) + 1)

            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])

            # Add labels and a legend
            plt.xlabel('Iterations')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)

            # Save the plot
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], design_name+"_"+metric+".png"), dpi=150)
            
            # Close the figure
            plt.close()

