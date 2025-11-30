from ..steps.scorer.pymol_utils import *
from bindcraft_plus.steps.scorer.pymol_utils import _reduce_hotspot_list

import json
from typing import Literal,Callable
from pathlib import Path

### tmp ### 
from matplotlib import colors as mcolors
palette={'4':'lightcoral','6':'sandybrown','8':'gold','10':'lawngreen','12':'aquamarine', "full":'silver'} 
palette_hex={k:mcolors.to_hex(v).replace('#','0x') for k,v in palette.items()}
palette_hex={'epitope_'+k if k!='full' else k:v for k,v in palette_hex.items()}
###    ###

_pre_process_dict:Dict[str,Callable[[str],None]]={
    'only_organic':no_inorganic_purify,
    'only_polymer':no_organic_purify,
    }

def known_binder_motifs(
    pdb:str,
    target_chain:str,
    ligand_chain:str,
    pre_process:Literal['only_organic','only_polymer']='only_polymer',
    output_dir:str='motif',
    epitope_ranges:List[int]=[4,6,8,10,12,14]
    ):
    stem=Path(pdb).stem
    Path(output_dir).mkdir(exist_ok=True,parents=True)
    cmd.load(pdb,stem+'_ori')
    _pre_process_dict[pre_process](stem+'_ori')
    hotspots=hotspots_by_ligand(stem+'_ori',
        target_chain,ligand_chain)
    cmd.save(f'{output_dir}/{stem}-full.pdb','target')
    with open(f'{output_dir}/{stem}-hotspot.json','w') as f:
        json.dump({'hotspots':hotspots['hotspots']},f,indent=2)
    for i in epitope_ranges:
        hotspots_to_seg_surf(stem+'_ori',hotspots['hotspots'],
            opt_obj=f'seg_{i}',vicinity=i)
    for obj in [f'seg_{i}' for i in epitope_ranges]:
        cmd.save(f'{output_dir}/{stem}-{obj}.pdb',obj)

    res=_reduce_hotspot_list(hotspots['hotspots'])
    res_sel=[]
    for k,v in res.items():
        res_sel.append(f'(chain {k} and resi {",".join([str(i) for i in v])})')
    cmd.select('hotspots',f'target and ( {" or ".join(res_sel) } )')
    target_chains=cmd.get_chains('target')
    binder_chain=[i for i in cmd.get_chains('complex') if i not in target_chains]
    cmd.disable(stem+'_ori')
    cmd.disable('complex')
    cmd.remove('hydro')
    cmd.show('licorice','hotspots')
    cmd.color('warmpink','hotspots and element C')
    for i in epitope_ranges:
        k = f'epitope_{i}'
        cmd.select(k,f'byres ( (target and chain {",".join(target_chains)}) within {i} of hotspots)')
    cmd.save(f'{output_dir}/{stem}.pse')