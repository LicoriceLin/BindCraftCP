from .basestep import *
import jax
import jax.numpy as jnp
from colabdesign import mk_afdesign_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss

class Hallucinate(BaseStep):
    def __init__(self,
        settings:GlobalSettings,
        af_model:mk_afdesign_model|None=None,
        ):
        super().__init__(settings)
        self.init_afdesign_model(af_model)

    def init_afdesign_model(
        self,
        af_model:mk_afdesign_model|None=None,
        )->mk_afdesign_model:
        advanced_settings=self.settings.adv
        if af_model is None:
            af_model = mk_afdesign_model(protocol="binder", debug=False, 
                data_dir=advanced_settings['af_params_dir'], 
                use_multimer=advanced_settings["use_multimer_design"], 
                num_recycles=advanced_settings["num_recycles_design"],
                best_metric='loss')
        else:
            af_model.restart(seed=self.settings.binder_settings.global_seed)
        self.af_model=af_model

    def config_afdesign_model(self,length:int,seed:int,helicity_value:float):
        advanced_settings=self.settings.adv
        target_settings=self.settings.target_settings
        af_model=self.af_model
        af_model.prep_inputs(
            pdb_filename=target_settings.starting_pdb, 
            chain=target_settings.chains, 
            binder_len=length, 
            hotspot=target_settings.target_hotspot_residues, 
            seed=seed, 
            rm_aa=advanced_settings["omit_AAs"],
            rm_target_seq=advanced_settings["rm_template_seq_design"], 
            rm_target_sc=advanced_settings["rm_template_sc_design"])
        self._current_design_seed=seed
        if advanced_settings.get('cyclize_peptide',False):
            # To implement: multi-chain target, scaffolding.
            add_cyclic_offset(af_model, offset_type=2)
        ### Update weights based on specified settings
        af_model.opt["weights"].update({"pae":advanced_settings["weights_pae_intra"],
                                        "plddt":advanced_settings["weights_plddt"],
                                        "i_pae":advanced_settings["weights_pae_inter"],
                                        "con":advanced_settings["weights_con_intra"],
                                        "i_con":advanced_settings["weights_con_inter"],
                                        })
        af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"],"cutoff":advanced_settings["intra_contact_distance"],"binary":False,"seqsep":9})
        af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"],"cutoff":advanced_settings["inter_contact_distance"],"binary":False})

        ### additional loss functions
        if advanced_settings["use_rg_loss"]:
            add_rg_loss(af_model, advanced_settings["weights_rg"])

        if advanced_settings["use_i_ptm_loss"]:
            add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

        if advanced_settings["use_termini_distance_loss"]:
            add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])
        add_helix_loss(af_model, helicity_value)

    def sample_trajectory(self,design_id:str,prefix:str='halu-')->DesignRecord:
        af_model=self.af_model
        advanced_settings=self.settings.adv
        design_models = [0,1,2,3,4] if advanced_settings["use_multimer_design"] else [0,1]
        verb=advanced_settings.get('verb',1)
        # af_model.design_logits(iters=50, e_soft=0.9, models=design_models,
        #      num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)
        af_model.design_logits(iters=advanced_settings["soft_iterations"], e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
            ramp_recycles=False, save_best=True,verbose=verb)
        af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
            sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True,verbose=verb)
        af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
            sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True,verbose=verb)
        greedy_tries = math.ceil(af_model._binder_len * (advanced_settings["greedy_percentage"] / 100))
        af_model.clear_best()
        if not advanced_settings.get('4stage-mcmc',False):
            af_model.design_pssm_semigreedy(
                soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True,
                seq_logits=af_model.aux["seq"]["pssm"][0],verbose=verb)
        else:
            af_model._design_mcmc(
                steps=advanced_settings["greedy_iterations"], 
                half_life=int(advanced_settings["greedy_iterations"]*advanced_settings.get('mcmc_half_life_ratio',0.2)), 
                T_init=advanced_settings.get('t_init_mcmc',0.01), 
                mutation_rate=greedy_tries, num_models=1, 
                models=design_models,
                sample_models=advanced_settings["sample_models"], save_best=True,verbose=verb)
        if advanced_settings.get('4stage_keep_best',False):
            best = af_model._tmp["best"]
            af_model.aux, seq = best["aux"], jnp.array(best["aux"]["seq"]['input'])
            af_model.set_seq(seq=seq, bias=af_model._inputs["bias"])
            af_model._save_results(save_best=True, verbose=False)
        
        metrics={k:af_model.aux['log'][v] for k,v in {f'{prefix}pLDDT':'plddt',f'{prefix}pTM':'ptm',f'{prefix}i-pTM':'i_ptm',f'{prefix}pAE':'pae',f'{prefix}i-pAE':'i_pae'}.items()}
        metrics.update({"helix":af_model.opt["weights"]["helix"],'length':af_model._binder_len,'seed':self._current_design_seed})
        pdbstr=af_model.save_pdb()
        return DesignRecord(id=design_id,sequence=af_model.get_seq(get_best=False)[0],
                pdb_strs={prefix.strip('-'):pdbstr},metrics=metrics)
    
    def log_trajectory(self):
        raise NotImplementedError

    def config_logger(self):
        raise NotImplementedError
    
    def __call__(self,prefix:str='halu-',cache_stem:str='hallucination',
        pdb_purge_stem:Optional[str]=None, input=None)->DesignBatch:
        binder_settings=self.settings.binder_settings
        batch=DesignBatch(Path(binder_settings.design_path)/cache_stem)
        batch.load_records()
        self.config_pdb_purge(pdb_purge_stem)

        global_id=0
        tot=len(binder_settings.binder_lengths)*len(binder_settings.random_seeds)*len(binder_settings.helix_values)
        to_fill=int(math.log10(tot)+1)
        for length in binder_settings.binder_lengths:
            for seed in binder_settings.random_seeds:
                for helicity_value in binder_settings.helix_values:
                    design_id = binder_settings.binder_name+'-'+str(global_id).zfill(to_fill)
                    if design_id not in batch.records:
                        self.config_afdesign_model(length,seed,helicity_value)
                        record=self.sample_trajectory(design_id,prefix=prefix)
                        if self.pdb_purge_dir is not None:
                            record.purge_pdb(prefix.strip('-'),self.pdb_purge_dir/f'{record.id}.pdb')
                        batch.add_record(record)
                        batch.save_record(design_id)
                    global_id+=1
        return batch
    
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

