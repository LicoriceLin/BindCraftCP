from bindcraft_plus.steps.scorer.pymol_utils import *
import json
target_metadata={
    "TBLR1":
        {"pdb":'5NAF',
         'target_chain':'B',
         'ligand_chain':'F',
         "pre_process":'no_organic',
         "cyc":True,
         "miniprot":False},
    "HA":
        {
         "pdb":'5VLI',
         'target_chain':'A,B',
         'ligand_chain':'C',
         "pre_process":'no_organic',
         "cyc":False,
         'miniprot':True
        },
    "WDR5":
        {
         "pdb":'2G99',
         'target_chain':'B',
         'ligand_chain':'C',
         "pre_process":'no_inorganic',
         "cyc":True,
         'miniprot':False
        },
    "TcdB":
        {
         "pdb":'6C0B',
         'target_chain':'A',
         'ligand_chain':'B',
         "pre_process":'no_inorganic',
         "cyc":False,
         'miniprot':True,
        },
    "ALS3":
        {
         "pdb":'4LEB',
         'target_chain':'A',
         'ligand_chain':'B',
         "pre_process":'no_inorganic',
         "cyc":True,
         'miniprot':False,
        },
    "SpCas9":
        {
         "pdb":'4ZT0',
         'target_chain':'A',
         'ligand_chain':'B',
         "pre_process":'manual:trunc no_inorganic B 27-43',
         "cyc":False,
         'miniprot':True,
        },
    'ClpP':
        {
         "pdb":'6BBA',
         'target_chain':'A,B',
         'ligand_chain':'L',
         "pre_process":'manual: AF3 refold, no_inorganic',
         "cyc":True,
         'miniprot':False,
        },
    }

target_metadata_plus={
    "Stx2a":
        {"pdb":'4m1u',
         'target_chain':'A,B,C,D,E,F',
         'ligand_chain':'G',
         "pre_process":'none',
         "cyc":False,
         "miniprot":True},
    }

def process_target(name:str,meta:dict):
    cmd.delete('all')
    if 'AF' not in meta['pre_process']:
        file=f'input/Jul_benchmark/{meta["pdb"].lower()}.pdb'
    else:
        file=f'input/Jul_benchmark/{meta["pdb"].lower()}-af.pdb'
    cmd.load(file,'ori')

    if 'trunc' in meta['pre_process']:
        *_,c,be=meta['pre_process'].split()
        cmd.remove(f'chain {c} and not resi {be}')

    if 'no_inorganic' in meta['pre_process']:
        no_inorganic_purify('ori')
    elif 'no_organic' in meta['pre_process']:
        no_organic_purify('ori')
    
    hotspots=hotspots_by_ligand('ori',
        meta['target_chain'],meta['ligand_chain'])
    cmd.save(f'input/Jul_benchmark/segs/{name}-full.pdb','target')
    with open(f'input/Jul_benchmark/segs/{name}.json','w') as f:
        json.dump({'hotspots':hotspots['hotspots']},f,indent=2)
    hotspots_to_seg_continue('ori',hotspots['hotspots'],'seg_con')
    for i in [4,6,8,10,12,14]:
        hotspots_to_seg_surf('ori',hotspots['hotspots'],
            opt_obj=f'seg_{i}',vicinity=i)
    cmd.save(f'input/Jul_benchmark/segs/{name}.pse')
    for obj in [f'seg_{i}' for i in [4,6,8,10,12,14]]+['seg_con']:
        cmd.save(f'input/Jul_benchmark/segs/{name}-{obj}.pdb',obj)

if __name__=='__main__':
    for k,v in target_metadata.items():
        process_target(k,v)