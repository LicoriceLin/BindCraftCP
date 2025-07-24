from bindcraft_plus.steps import *
from bindcraft_plus.utils import TargetSettings,GlobalSettings,AdvancedSettings,FilterSettings,BinderSettings
from pathlib import Path

def _sampling(settings:GlobalSettings):
    hallu=Hallucinate(settings)
    filter=Filter(settings)
    refold=Refold(settings)
    graft=Graft(settings)
    mpnn=MPNN(settings)
    relax=Relax(settings)
    annot_rmsd=AnnotRMSD(settings)
    annot_surf=AnnotSurf(settings)
    annot_ptm=AnnotPTM(settings)
    annot_occ=AnnotPolarOccupy(settings)
    annot_pi=AnnotPI(settings)

    # step 1 
    batch=hallu.process_batch(pdb_purge_stem='hallu')

    batch=filter.set_recipe("after:hallucinate").process_batch(batch)
    if settings.adv.get('templated',False):
        graft.process_batch(batch,pdb_purge_stem='graft')
    refold.process_batch(batch,pdb_purge_stem='refold')
    
    # step 2
    batch=filter.set_recipe("after:refold").process_batch(batch)
    annot_surf.process_batch(batch)
    for i in refold.pdb_to_add:
        if 'multimer' in i:
            annot_occ.process_batch(batch,pdb_to_take=i)
            annot_rmsd.process_batch(batch,pdb_to_take=i)

    if not settings.adv['cyclize_peptide']:
        for i in ['O-linked_glycosylation','N-linked_glycosylation']:
            annot_ptm.process_batch(batch,model=i)
        
    # step 3
    mpnn.process_batch(batch)
    if settings.adv.get('templated',False):
        graft.process_batch(batch,pdb_purge_stem='graft')
    refold.process_batch(batch,pdb_purge_stem='refold')

    # step 4
    batch=filter.set_recipe("after:refold").process_batch(batch)
    annot_surf.process_batch(batch)
    for i in refold.pdb_to_add:
        if 'multimer' in i:
            annot_occ.process_batch(batch,pdb_to_take=i)
            annot_rmsd.process_batch(batch,pdb_to_take=i)

    if not settings.adv['cyclize_peptide']:
        for i in ['O-linked_glycosylation','N-linked_glycosylation']:
            annot_ptm.process_batch(batch,model=i)
    relax.process_batch(batch,pdb_purge_stem='relax')
    annot_pi.process_batch(batch,pdb_to_take=relax.pdb_to_add[0],optargs=['--quiet'])
    return batch


def test_templated():
    odir='output/test_cyc'
    global_setting=GlobalSettings(
            target_settings=TargetSettings(
                starting_pdb='example/WDR5-seg_8.pdb',chains='A',
                full_target_pdb='example/WDR5-full.pdb',full_target_chain='A'),
            binder_settings=BinderSettings(design_path=odir,binder_name='test_surf',
                binder_lengths=[12,16],random_seeds=[42,43],helix_values=[0.,-0.5]),
            advanced_settings=AdvancedSettings(
                advanced_paths=['config/base_adv_setting.json',
                'config/patch_4s_mc.json',
                'config/patch_cyc.json',
                "config/patch_templated.json"]),
            filter_settings=FilterSettings(
                filters_path='config/default_filter.json')
            )
    Path(odir).mkdir(exist_ok=True,parents=True)
    global_setting.save(f'{odir}/settings.json')
    _sampling(global_setting)

def test_routine():
    odir='output/test_mini'
    global_setting=GlobalSettings(
            target_settings=TargetSettings(
                starting_pdb='example/PDL1.pdb',chains='A'),
            binder_settings=BinderSettings(design_path=odir,binder_name='test',
                binder_lengths=[55,75],random_seeds=[42,43],helix_values=[0.,-0.5]),
            advanced_settings=AdvancedSettings(
                advanced_paths=[
                'config/base_adv_setting.json',
                'config/patch_4s_mc.json']),
            filter_settings=FilterSettings(
                filters_path='config/default_filter.json')
            )
    Path(odir).mkdir(exist_ok=True,parents=True)
    global_setting.save(f'{odir}/settings.json')
    _sampling(global_setting)