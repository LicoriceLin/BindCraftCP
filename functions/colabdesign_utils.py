####################################
############## ColabDesign functions
####################################
### Import dependencies
import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import List,Dict,Tuple,Optional
import jax
import jax.numpy as jnp
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.shared.utils import copy_dict
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages,target_pdb_rmsd
from .pyrosetta_utils import pr_relax, align_pdbs,unaligned_rmsd,score_interface
from .generic_utils import update_failures,BasicDict,backup_if_exists,backuppdb_if_multiframe

def add_cyclic_offset(self:mk_afdesign_model, offset_type=2):
  '''add cyclic offset to connect N and C term'''
  def cyclic_offset(L):
    i = np.arange(L)
    ij = np.stack([i,i+L],-1)
    offset = i[:,None] - i[None,:]
    c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
    if offset_type == 1:
      c_offset = c_offset
    elif offset_type >= 2:
      a = c_offset < np.abs(offset)
      c_offset[a] = -c_offset[a]
    if offset_type == 3:
      idx = np.abs(c_offset) > 2
      c_offset[idx] = (32 * c_offset[idx] )/  abs(c_offset[idx])
    return c_offset * np.sign(offset)
  idx = self._inputs["residue_index"]
  offset = np.array(idx[:,None] - idx[None,:])

  if self.protocol == "binder":
    c_offset = cyclic_offset(self._binder_len)
    offset[self._target_len:,self._target_len:] = c_offset

  elif self.protocol in ["fixbb","hallucination"]:
    Ln = 0
    for L in self._lengths:
      offset[Ln:Ln+L,Ln:Ln+L] = cyclic_offset(L)
      Ln += L
  
  elif self.protocol=="partial":
    print("Under Construction")
    raise NotImplementedError
  else:
    raise ValueError
  
  self._inputs["offset"] = offset

# def add_rg_loss(self:mk_afdesign_model, weight=0.1):
#   '''add radius of gyration loss'''
#   def loss_fn(inputs, outputs):
#     xyz = outputs["structure_module"]
#     ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
#     rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
#     rg_th = 2.38 * ca.shape[0] ** 0.365
#     rg = jax.nn.elu(rg - rg_th)
#     return {"rg":rg}
#   self._callbacks["model"]["loss"].append(loss_fn)
#   self.opt["weights"]["rg"] = weight

# hallucinate a binder
def binder_hallucination(design_name:str, starting_pdb:str, chain:str, 
                    target_hotspot_residues:str, length:int, seed:int, helicity_value:float, 
                    design_models:List[int], advanced_settings:BasicDict, 
                    design_paths:Dict[str,str], failure_csv:str, af_model:Optional[mk_afdesign_model]=None,
                    ):
    # from pdb import set_trace; set_trace()
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")
    if af_model is None:
        # clear GPU memory for new trajectory
        clear_mem()

        # initialise binder hallucination model
        af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"], 
                                    use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                    best_metric='loss')

        # sanity check for hotspots
        if target_hotspot_residues == "":
            target_hotspot_residues = None
        
        af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length, hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"],
                            rm_target_seq=advanced_settings["rm_template_seq_design"], rm_target_sc=advanced_settings["rm_template_sc_design"])
    else:
        af_model.restart(seed=seed)

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

    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"],"cutoff":advanced_settings["intra_contact_distance"],"binary":False,"seqsep":9})
    af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"],"cutoff":advanced_settings["inter_contact_distance"],"binary":False})
        

    ### additional loss functions
    if advanced_settings["use_rg_loss"]:
        # radius of gyration loss
        add_rg_loss(af_model, advanced_settings["weights_rg"])

    if advanced_settings["use_i_ptm_loss"]:
        # interface pTM loss
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        # termini distance loss
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])

    # add the helicity loss
    add_helix_loss(af_model, helicity_value)

    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### start design algorithm based on selection
    if advanced_settings["design_algorithm"] == '2stage':
        # uses gradient descend to get a PSSM profile and then uses PSSM to bias the sampling of random mutations to decrease loss
        af_model.design_pssm_semigreedy(soft_iters=advanced_settings["soft_iterations"], hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

    elif advanced_settings["design_algorithm"] == '3stage':
        # 3 stage design using logits, softmax, and one hot encoding
        af_model.design_3stage(soft_iters=advanced_settings["soft_iterations"], temp_iters=advanced_settings["temporary_iterations"], hard_iters=advanced_settings["hard_iterations"], 
                                num_models=1, models=design_models, sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'greedy':
        # design by using random mutations that decrease loss
        af_model.design_semigreedy(advanced_settings["greedy_iterations"], tries=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'mcmc':
        # design by using random mutations that decrease loss
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)
    elif advanced_settings["design_algorithm"] == 'mcmc_sampling':
        '''
        for quick bunch sampling.
        At Stage 1, 50 steps of design_logits (e_soft=0., num_models=3) are conducted as annealing.
        At Stage 2, step-by-steps of design_logits (e_soft=1) are conducted until:
            1) maximum of `soft_iterations` is achieved. Or
            2) or np.random.rand() > 2**((pLDDT-0.65)*10)-1 (0.75: forced exit)
        At Stage 3, step-by-steps of design_soft (e_temp=1e-2) are conducted until:
            1) maximum of `temp_iterations` achieved. Or
            2) or np.random.rand() > 2**((pLDDT-0.75)*10)-1 (0.85: forced exit)
            Then we are having af_model.aux["seq"]["logits"] as profile:
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"] (or 'pssm'?)
            af_model._inputs/self.aux are kept in af_model._tmp.
        At Stage 4, design_semigreedy(tries=greedy_tries,seq_logits=af_model._tmp["seq_logits"][0] for x rounds until 'pssm_sampling_num'/'pssm_max_round' achieved.
            for each round, if pLDDT > `day0_final_plddt` and no same seq hit before (kept in af_model._tmp['prev_hit']),
                a full output would be generated / saved with model_pdb_path.replace('.pdb','b{i}.pdb')
            Then self._inputs/self.aux would be resumed from af_model._tmp
        '''
        print("Stage 1: Annealing")
        af_model.design_logits(iters=50, e_soft=0., models=design_models, num_models=advanced_settings.get('annealing_num_models',1),  #3
            sample_models=advanced_settings["sample_models"], save_best=True)
        plddt = get_best_plddt(af_model, length)
        print(f"Annealing Finished with init pLDDT {plddt:.2f}, Continue design_logits")
        af_model.clear_best()
        hit_count=0
        moveon_hitcount=advanced_settings.get('moveon_hitcount',3)
        while af_model._k<advanced_settings["soft_iterations"]:
            soft=(af_model._k-49)/(advanced_settings["soft_iterations"]-49)
            af_model.design_logits(iters=1, soft=soft, models=design_models, num_models=1, 
                sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
            plddt=np.mean(af_model.aux["plddt"][-length:])
            if  plddt>0.65 and np.random.rand() < 2**((plddt-0.65)*10)-1:
                hit_count+=1
                if hit_count>=moveon_hitcount:
                    break
        action_str= "Early Stopped" if af_model._k<advanced_settings["soft_iterations"] else "Terminated"
        print(f"design_logits {action_str} at {af_model._k} step, with pLDDT {plddt:.2f}. Begin design_soft")

        design_soft_step=0
        hit_count=0
        while design_soft_step<advanced_settings["temporary_iterations"]:
            temp=1e-2+(1-1e-2)*(1-(design_soft_step+1)/advanced_settings["temporary_iterations"])**2
            af_model.design(1, soft=soft,temp=temp, models=design_models, num_models=1,
                    sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
            plddt=np.mean(af_model.aux["plddt"][-length:])
            design_soft_step+=1
            if  plddt>0.75 and np.random.rand() < 2**((plddt-0.75)*10)-1:
                hit_count+=1
                if hit_count>=moveon_hitcount:
                    break
        # af_model.set_seq
        action_str= "Early Stopped" if design_soft_step<advanced_settings["temporary_iterations"] else "Terminated"
        print(f"design_soft {action_str} at {af_model._k} step, with pLDDT {plddt:.2f}. Begin `pssm-mcmc`")
        print(f'seq: {af_model.get_seq()[0]}')
        def load_settings():    
            if "save_" in af_model._tmp:
                [af_model.opt, af_model._args, af_model._params, af_model._inputs,af_model._tmp["best"]] = af_model._tmp["save_"]#af_model._tmp.pop("save_")
        def save_settings():
            # load_settings()
            af_model._tmp["save_"] = [copy_dict(x) for x in [
                af_model.opt, af_model._args, af_model._params, af_model._inputs,af_model._tmp["best"]]]
        save_settings()
        # return af_model
        af_model._tmp["seq_logits"] = seq_logits = af_model.aux["seq"]["logits"][0] # or logits/soft?
        design_greedy_step=0
        mcmc_inner_iter=advanced_settings.get("mcmc_inner_iter",100)
        af_model._tmp['outputs']=[]
        half_life = round(mcmc_inner_iter, 0)
        t_mcmc = 0.01
        save_branch=advanced_settings.get('save_branch',False)
        while design_greedy_step<advanced_settings["greedy_iterations"]:
            af_model.clear_best()
            af_model._design_mcmc(
                steps=mcmc_inner_iter,seq_logits=seq_logits,
                half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, 
                num_models=advanced_settings.get('mcmc_num_models',1), 
                models=design_models,sample_models=advanced_settings["sample_models"], 
                save_best=True)
            af_model._tmp['outputs'].append(af_model._tmp["best"])
            # af_model._k-=mcmc_inner_iter
            if save_branch:
                af_model.save_pdb(model_pdb_path.replace('.pdb',f'-b{design_greedy_step+1}.pdb'))
            print(f"sample id {design_greedy_step+1} finished with pLDDT {get_best_plddt(af_model, length):.2f}")
            print(f'seq: {af_model.get_seq()[0]}')
            # final_plddt = get_best_plddt(af_model, length)
            # todo: discard poor results and don't count them
            load_settings()
            # save_settings()
            design_greedy_step+=1
        print("tmp: choose the best sample to continue.")
        losses = [x["aux"]["loss"] for x in af_model._tmp['outputs']]
        best = af_model._tmp['outputs'][np.argmin(losses)]
        af_model.aux, seq = best["aux"], jnp.array(best["aux"]["seq"]['input'])
        af_model.set_seq(seq=seq, bias=af_model._inputs["bias"])
        af_model._save_results(save_best=True, verbose=False)
        # return af_model

        # print(f"Soft Design Early Stopped at {af_model._k} step, with pLDDT {plddt:.2f}, Anneal temp with decreasing steps")
        # remain_iter=advanced_settings["temporary_iterations"]-design_soft_step
        # b_temp=1e-2+(1-1e-2)*(1-(design_soft_step+1)/advanced_settings["temporary_iterations"])**2
        # af_model.design_soft(
        #     iters=remain_iter, tmp=b_temp,e_temp=1e-2, 
        #     e_step=0.1,models=design_models, num_models=1,
        #     sample_models=advanced_settings["sample_models"], 
        #     ramp_recycles=False, save_best=True)


    elif advanced_settings["design_algorithm"] == '4stage':
        # initial logits to prescreen trajectory
        print("Stage 1: Test Logits")
        af_model.design_logits(iters=50, e_soft=0.9, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)

        # determine pLDDT of best iteration according to lowest 'loss' value
        initial_plddt = get_best_plddt(af_model, length)
        
        # if best iteration has high enough confidence then continue
        if initial_plddt > advanced_settings.get('day0_initial_plddt',0.65):
            print("Initial trajectory pLDDT good, continuing: "+str(initial_plddt))
            if advanced_settings["optimise_beta"]:
                # temporarily dump model to assess secondary structure
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
                os.remove(model_pdb_path)

                # if beta sheeted trajectory is detected then choose to optimise
                if float(beta) > 15:
                    advanced_settings["soft_iterations"] = advanced_settings["soft_iterations"] + advanced_settings["optimise_beta_extra_soft"]
                    advanced_settings["temporary_iterations"] = advanced_settings["temporary_iterations"] + advanced_settings["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                    print("Beta sheeted trajectory detected, optimising settings")

            # how many logit iterations left
            logits_iter = advanced_settings["soft_iterations"] - 50
            if logits_iter > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
                                    ramp_recycles=False, save_best=True)
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
                print("Optimised logit trajectory pLDDT: "+str(logit_plddt))
            else:
                logit_plddt = initial_plddt

            # perform softmax trajectory design
            if advanced_settings["temporary_iterations"] > 0:
                print("Stage 2: Softmax Optimisation")
                af_model.clear_best()
                af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
                softmax_plddt = get_best_plddt(af_model, length)
            else:
                softmax_plddt = logit_plddt

            # perform one hot encoding
            if softmax_plddt > advanced_settings.get('day0_softmax_plddt',0.65):
                print("Softmax trajectory pLDDT good, continuing: "+str(softmax_plddt))
                if advanced_settings["hard_iterations"] > 0:
                    af_model.clear_best()
                    print("Stage 3: One-hot Optimisation")
                    af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True)
                    onehot_plddt = get_best_plddt(af_model, length)

                if onehot_plddt > advanced_settings.get('day0_onehot_plddt',0.65):
                    # perform greedy mutation optimisation
                    print("One-hot trajectory pLDDT good, continuing: "+str(onehot_plddt))
                    if advanced_settings["greedy_iterations"] > 0:
                        print("Stage 4: PSSM Semigreedy Optimisation")
                        af_model.clear_best()
                        af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True,
                                                        seq_logits=af_model.aux["seq"]["pssm"][0])

                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    print("One-hot trajectory pLDDT too low to continue: "+str(onehot_plddt))

            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax trajectory pLDDT too low to continue: "+str(softmax_plddt))

        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial trajectory pLDDT too low to continue: "+str(initial_plddt))

    else:
        print("ERROR: No valid design model selected")
        exit()
        return

    ### save trajectory PDB
    # final_plddt = get_best_plddt(af_model, length)
    final_plddt=np.mean(af_model.aux["plddt"][-length:])
    backup_if_exists(model_pdb_path)
    af_model.save_pdb(model_pdb_path)
    backuppdb_if_multiframe(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    # let's check whether the trajectory is worth optimising by checking confidence, clashes, and contacts
    # check clashes
    #clash_interface = calculate_clash_score(model_pdb_path, 2.4)
    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

    #if clash_interface > 25 (legacy) or ca_clashes > 0:
    if ca_clashes > advanced_settings.get('day0_ca_clashes',0):
        af_model.aux["log"]["terminate"] = "Clashing"
        update_failures(failure_csv, 'Trajectory_Clashes')
        print("Severe clashes detected, skipping analysis and MPNN optimisation")
        print("")
    else:
        # check if low quality prediction
        if final_plddt < advanced_settings.get('day0_final_plddt',0.7):
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_final_pLDDT')
            print("Trajectory starting confidence low, skipping analysis and MPNN optimisation")
            print("")
        else:
            # does it have enough contacts to consider?
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            # if less than 3 contacts then protein is floating above and is not binder
            if binder_contacts_n < advanced_settings.get('day0_binder_contacts_n',3):
                af_model.aux["log"]["terminate"] = "LowConfidence"
                update_failures(failure_csv, 'Trajectory_Contacts')
                print("Too few contacts at the interface, skipping analysis and MPNN optimisation")
                print("")
            else:
                # phew, trajectory is okay! We can continue
                af_model.aux["log"]["terminate"] = ""
                print("Trajectory successful, final pLDDT: "+str(final_plddt))

    # move low quality prediction:
    if af_model.aux["log"]["terminate"] != "":
        _=design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"]+f'/{design_name}.pdb'
        if os.path.exists(_):
            os.remove(_)
        shutil.move(model_pdb_path, _)

    ### get the sampled sequence for plotting
    af_model.get_seqs()
    if advanced_settings["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    ### save the hallucination trajectory animation
    if advanced_settings["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    if advanced_settings["save_trajectory_pickle"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name+".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return af_model

# run prediction for binder with masked template target
def predict_binder_complex(
        prediction_model:mk_afdesign_model|None, binder_sequence:str, mpnn_design_name:str, 
        target_pdb:str|None, chain:str|None, length:int|None, trajectory_pdb:str|None, prediction_models:list[int], 
        advanced_settings:BasicDict, filters:Dict[str,Dict[str,bool|None]], 
        design_paths:Dict[str,str], failure_csv:str, seed:int|None=None,
        binder_chain:str = "B",re_prep:bool=False,
        )->Tuple[Dict[str,BasicDict],bool]|Tuple[Dict[str,BasicDict],bool,mk_afdesign_model]:
    '''
    In House Modification:
    Now init prediction_model inside the function and return it 
    if it's None as input.  

    (low priority) 
    MPNN & MPNN/Relaxed paths are hardcoded. try make it flexible.
    '''
    length=len(binder_sequence) if length is None else length
    if prediction_model is None:
        prediction_model = mk_afdesign_model(
            protocol="binder", 
            num_recycles=advanced_settings["num_recycles_validation"],
            data_dir=advanced_settings["af_params_dir"],
            use_multimer= (not advanced_settings["use_multimer_design"]))
        ret_m=True
        re_prep=True
    else:
        ret_m=False
    if re_prep:
        prediction_model.prep_inputs(
                pdb_filename=target_pdb, 
                chain=chain, 
                binder_len=length, 
                rm_target_seq=advanced_settings["rm_template_seq_predict"],
                rm_target_sc=advanced_settings["rm_template_sc_predict"],
                seed=seed)
    if advanced_settings.get('cyclize_peptide',False):
        # To implement: multi-chain target, scaffolding.
        add_cyclic_offset(prediction_model, offset_type=2)
    prediction_stats:Dict[str,BasicDict] = {}

    # clean sequence
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())

    # reset filtering conditionals
    pass_af2_filters = True
    filter_failures = {}

    # start prediction per AF2 model, 2 are used by default due to masked templates
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        # if not os.path.exists(complex_pdb):
        # predict model
        prediction_model.predict(seq=binder_sequence, models=[model_num], 
            num_recycles=advanced_settings["num_recycles_validation"], verbose=False,seed=seed)
        prediction_model.save_pdb(complex_pdb,get_best=False)
        prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

        # extract the statistics for the model
        stats = {
            'pLDDT': round(prediction_metrics['plddt'], 2), 
            'pTM': round(prediction_metrics['ptm'], 2), 
            'i_pTM': round(prediction_metrics['i_ptm'], 2), 
            'pAE': round(prediction_metrics['pae'], 2), 
            'i_pAE': round(prediction_metrics['i_pae'], 2)
        }
        prediction_stats[model_num+1] = stats

        # List of filter conditions and corresponding keys
        filter_conditions = [
            (f"{model_num+1}_pLDDT", 'plddt', '>='),
            (f"{model_num+1}_pTM", 'ptm', '>='),
            (f"{model_num+1}_i_pTM", 'i_ptm', '>='),
            (f"{model_num+1}_pAE", 'pae', '<='),
            (f"{model_num+1}_i_pAE", 'i_pae', '<='),
        ]

        # perform initial AF2 values filtering to determine whether to skip relaxation and interface scoring
        for filter_name, metric_key, comparison in filter_conditions:
            threshold = filters.get(filter_name, {}).get("threshold")
            if threshold is not None:
                if comparison == '>=' and prediction_metrics[metric_key] < threshold:
                    pass_af2_filters = False
                    filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                elif comparison == '<=' and prediction_metrics[metric_key] > threshold:
                    pass_af2_filters = False
                    filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1

        if not pass_af2_filters:
            break

    # Update the CSV file with the failure counts
    if filter_failures:
        update_failures(failure_csv, filter_failures)

    # AF2 filters passed, contuing with relaxation
    for model_num in prediction_models:
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if pass_af2_filters and model_num+1 in prediction_stats and os.path.exists(complex_pdb):
            mpnn_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            pr_relax(complex_pdb, mpnn_relaxed,advanced_settings.get('cyclize_peptide',False))
            
            # integrated from BindCraft Main Workflow
            num_clashes_mpnn = calculate_clash_score(complex_pdb)
            num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_relaxed)
            # analyze interface scores for relaxed af2 trajectory
            mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_relaxed, binder_chain,cyclize_peptide=advanced_settings.get('cyclize_peptide',False))
            target_interface_residues = ','.join([f'A{i}' for i in hotspot_residues(mpnn_relaxed,'A',target_chain='B').keys()])
            # secondary structure content of starting trajectory binder
            (mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface, 
             mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt, mpnn_ss_plddt 
             )= calc_ss_percentage(complex_pdb, advanced_settings, binder_chain)
            # unaligned RMSD calculate to determine if binder is in the designed binding site
            rmsd_site = -1. if trajectory_pdb is None else unaligned_rmsd(trajectory_pdb, complex_pdb, binder_chain, binder_chain) 
            # calculate RMSD of target compared to input PDB
            target_rmsd = target_pdb_rmsd(complex_pdb, target_pdb, chain)

            prediction_stats[model_num+1].update({
                'i_pLDDT': mpnn_i_plddt,
                'ss_pLDDT': mpnn_ss_plddt,
                'Unrelaxed_Clashes': num_clashes_mpnn,
                'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                'ShapeComplementarity': mpnn_interface_scores['interface_sc'],
                'PackStat': mpnn_interface_scores['interface_packstat'],
                'dG': mpnn_interface_scores['interface_dG'],
                'dSASA': mpnn_interface_scores['interface_dSASA'],
                'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],
                'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],
                'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                'InterfaceAAs': mpnn_interface_AA,
                'Interface_Helix%': mpnn_alpha_interface,
                'Interface_BetaSheet%': mpnn_beta_interface,
                'Interface_Loop%': mpnn_loops_interface,
                'Binder_Helix%': mpnn_alpha,
                'Binder_BetaSheet%': mpnn_beta,
                'Binder_Loop%': mpnn_loops,
                'Hotspot_RMSD': rmsd_site,
                'Target_RMSD': target_rmsd,
                'mpnn_interface_residues':mpnn_interface_residues,
                "target_interface_residues":target_interface_residues
            })
        else:
            if os.path.exists(complex_pdb):
                os.remove(complex_pdb)
    ret=(prediction_stats, pass_af2_filters,prediction_model) if ret_m else (prediction_stats, pass_af2_filters)
    return ret

# run prediction for binder alone
def predict_binder_alone(
        prediction_model:mk_afdesign_model|None, binder_sequence:str, mpnn_design_name:str, 
        length:int|None, trajectory_pdb:str|None, binder_chain:str|None, prediction_models:List[int], 
        advanced_settings:BasicDict, design_paths:str, seed:int|None=None,re_prep:bool=False,
        )->Dict[str,BasicDict]|Tuple[Dict[str,BasicDict],mk_afdesign_model]:
    '''
    In House Modification:
    Now init prediction_model inside the function and return it 
    if it's None as input.  
    '''
    length=len(binder_sequence) if length is None else length
    if prediction_model is None:
        prediction_model=mk_afdesign_model( 
            protocol="hallucination", use_templates=False, initial_guess=False,
            use_initial_atom_pos=False, num_recycles=advanced_settings["num_recycles_validation"],
            data_dir=advanced_settings["af_params_dir"], use_multimer=advanced_settings["use_multimer_design"])
        
        ret_m=True
        re_prep=True
    else:
        ret_m=False
    if re_prep:
        prediction_model.prep_inputs(length=length,seed=seed)
    if advanced_settings.get('cyclize_peptide',False):
        # To implement: multi-chain target, scaffolding.
        add_cyclic_offset(prediction_model, offset_type=2)
    binder_stats = {}

    # prepare sequence for prediction
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    prediction_model.set_seq(binder_sequence)

    # predict each model separately
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        binder_alone_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(binder_alone_pdb):
            # predict model
            prediction_model.predict(models=[model_num], 
                    num_recycles=advanced_settings["num_recycles_validation"], verbose=False,seed=seed)
            prediction_model.save_pdb(binder_alone_pdb,get_best=False)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, pae

            
            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2),
            }
            
            # integrated from BindCraft Main Workflow
            # align binder model to trajectory binder
            if trajectory_pdb is not None and binder_chain is not None:
                align_pdbs(trajectory_pdb, binder_alone_pdb, binder_chain, "A")
                rmsd_binder = unaligned_rmsd(trajectory_pdb, binder_alone_pdb, binder_chain, "A")
                stats.update({'Binder_RMSD': rmsd_binder})
            else:
                stats.update({'Binder_RMSD': None})
            binder_stats[model_num+1] = stats
    ret = (binder_stats,prediction_model) if ret_m else binder_stats
    return ret

# run MPNN to generate sequences for binders
def mpnn_gen_sequence(trajectory_pdb:str, binder_chain:str, trajectory_interface_residues:str, 
    advanced_settings:BasicDict):
    # clear GPU memory
    clear_mem()

    # initialise MPNN model
    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"], model_name=advanced_settings["model_path"], weights=advanced_settings["mpnn_weights"])

    # check whether keep the interface generated by the trajectory or whether to redesign with MPNN
    design_chains = 'A,' + binder_chain

    if advanced_settings["mpnn_fix_interface"]:
        fixed_positions = 'A,' + trajectory_interface_residues
        print("Fixing interface residues: "+trajectory_interface_residues)
    else:
        fixed_positions = 'A'

    # prepare inputs for MPNN
    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])

    # sample MPNN sequences in parallel
    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"], num=advanced_settings["num_seqs"], batch=advanced_settings["num_seqs"])

    return mpnn_sequences

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