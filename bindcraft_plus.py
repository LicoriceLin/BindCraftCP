from bindcraft.design import Design
from bindcraft._import import *
from bindcraft.post_design import Metrics
from bindcraft.score import Score
from bindcraft.util import _bindcraft_4stage_midfilter,_bc_benchmark_filters

if __name__=='__main__':
    import argparse
    import json
    # outdir='output/cd33_demo'
    # target_pdb="input/cd33_demo/5ihb_cd33.pdb"
    # chains='A'
    # target_hotspot_residues=''

    parser=argparse.ArgumentParser(
        prog='BindCraft+.',
        description=('Reimplementation of BindCraft. Try to keep interface/behavior identical to original version.\n\n'
                     'NOTE: No random sampling of length/helix values. Sample Iteratively for Seed * Length * Helix.'),

        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out-dir','-o',required=True,help='directory for design outputs.')
    parser.add_argument('--target-pdb','-i',required=True,help='path to target pdb.')
    parser.add_argument('--chains','-ch',default='A',help='Target chains. Sep by ",".')
    parser.add_argument('--target-hotspot-residues','-hot',default='',help='Target chains. Sep by ",".')
    parser.add_argument('--prefix','-p',default='4s',help='prefix for designs ids.')

    parser.add_argument('--random-seeds','-r',nargs='*',default='0 42 3407'.split(),
        help='seeds to be iterated for each length/helix combination. \nArg Format:`-r 0 1 2 ...` ')
    parser.add_argument('--lengths','-l',nargs='*',default='50 75 100'.split(),
        help='lengths to be iterated for designs.')
    parser.add_argument('--helices','-x',nargs='*',default='-0.2 0. 0.2'.split(),
        help='helix penalty values to be iterated for designs.')
    
    parser.add_argument('--filter-setting','-filt',default='settings_filters/default_filters.json',
        help='path to filters.json')
    parser.add_argument('--base-advanced-setting','-set',default='settings_advanced/default_4stage_multimer.json',
        help='path to base advanced settings. NOTE: Only Set Sampling/Refolding configs here.')
    parser.add_argument('--patch-advanced-setting','-set1',default='',
        help='You may write another .json to override `base advanced setting`.')
    
    args=parser.parse_args()
    outdir:str=args.out_dir
    target_pdb:str=args.target_pdb
    advanced_settings_overload = {
            "day0_initial_plddt":0.0,
            "day0_softmax_plddt":0.0,
            "day0_onehot_plddt":0.0,
            "day0_final_plddt":0.0,
            "day0_ca_clashes":99,
            "day0_binder_contacts_n":1,
        } # disable half-way abortion of sampling.
    if args.patch_advanced_setting:
        advanced_settings_overload.update(json.load(open(args.patch_advanced_setting,'r')))

    design=Design(
    outdir=outdir,
    starting_pdb = target_pdb,
    chains = args.chains ,
    target_hotspot_residues = args.target_hotspot_residues,
    base_advanced_settings_path=args.base_advanced_setting,
    advanced_settings_overload=advanced_settings_overload)

    design.init_task()
    design.sampling(args.prefix,
                    seeds=[int(i) for i in args.random_seeds],
                    lengths=[int(i) for i in args.lengths],
                    helicity_values=[int(float(i)*10) for i in args.helices])
    ### ###
    print('ColabDesign Sampling Finished! Pre-filter & MPNN')
    metrics=Metrics(outdir=outdir)
    metrics.post_process('minimal')
    metrics.filter(
        refold_mode='none',
        filters=_bindcraft_4stage_midfilter,
        cyclic=False)
    metrics.mpnn_sample(
        mode='non_ppi',
        num_sample=design.advanced_settings['num_seqs'],
        max_mpnn_sequences=design.advanced_settings['max_mpnn_sequences'],
        seed=88
        )
    ### ###
    print('MPNN Finished! Refold/Filter')
    score=Score(outdir=outdir,binder_name=Path(target_pdb).stem)
    score.score()

    metrics_m=Metrics(outdir=outdir,mode='mpnn')
    metrics_m.filter(
        refold_mode='none',
        filters=json.load(open(args.filter_setting,'r')),
        cyclic=False)
    metrics_m.metrics.to_csv(outdir+'/annot.csv',index_label='Design')