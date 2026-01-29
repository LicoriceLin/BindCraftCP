import click
from epitopecraft.pipelines.refold_val import RefoldValidation
from epitopecraft.utils import (
    TargetSettings,GlobalSettings,AdvancedSettings,
    FilterSettings,BinderSettings,
    DesignBatch, DesignRecord, flatten_dict
)

from pathlib import Path
import pandas as pd
import json

from typing import Iterable, Union, Tuple, Optional, List, Dict
from Bio.PDB import PDBParser, PDBIO, Select

from tqdm.auto import tqdm

################################################################## dumb settings
target_settings = 'config/refold_val_configs/test_target_setting.json'
binder_settings = 'config/refold_val_configs/test_binder_setting.json'
filter_settings = 'config/refold_val_configs/default_filter.yaml'
advanced_settings = ['config/refold_val_configs/base_advanced_settings.yaml']


################################################################## helper funcs

def _cutChain(pdb_in: Union[str, Path],
             pdb_out: Union[str, Path],
             keep_resi: Iterable[int],
             chain: str="B") -> None:
    '''
    Cut the target from a complex given residues to keep
    '''
    pdb_in = Path(pdb_in)
    pdb_out = Path(pdb_out)
    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    keep_set = set(int(x) for x in keep_resi)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_in.stem, str(pdb_in))
    
    class KeepSelected(Select):
        def accept_residue(self, residue):
            resi_chain = residue.get_parent()
            if resi_chain.id != chain:
                return True
            
            hetflag, resseq, icode = residue.get_id() # _, resseq, _ for reference
            return int(resseq) in keep_set
        
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_out), select=KeepSelected())

def _loadCutoffMap(map_file: Union[str, Path], target: str) -> Dict[str, List[int]]:
    '''
    Input: (either is fine):
        { "TARGET": { "top_30": [...], "top_50": [...] } }
        { "top_30": [...], "top_50": [...] }
    Output: dict, key indicate cutoff label e.g. 'top_30'
    '''
    map_file = Path(map_file)
    d = json.loads(map_file.read_text())

    if target in d and isinstance(d[target], dict):
        return {k: list(map(int, v)) for k, v in d[target].items()}
    
    return {k: list(map(int, v)) for k, v in d.items()}

def _parseCutoffs(cutoff_str) -> List[Optional[int]]:
    '''
    "full,30,50" -> [None, 30, 50] 
    '''
    out = []
    for cutoff in [c.strip() for c in cutoff_str.split(",")]:
        if cutoff.lower() in {'full', 'none'}:
            out.append(None)
        else:
            out.append(int(cutoff))
    return out

def _loadSettings(target_settings, binder_settings, 
                  filter_settings, advanced_settings: List[str]) -> GlobalSettings:
    '''
    Load dumb settings
    '''
    if len(advanced_settings)==1:
        adv_s=AdvancedSettings.from_file(advanced_settings[0])
    else:
        adv_s=AdvancedSettings(list(advanced_settings))
    settings=GlobalSettings(
        target_settings=TargetSettings.from_file(target_settings),
        binder_settings=BinderSettings.from_file(binder_settings),
        advanced_settings=adv_s,
        filter_settings=FilterSettings.from_file(filter_settings)
        )
    return settings

def _saveParams(out_dir, name="run_params", **kwargs):
    out_path = Path(out_dir) / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(kwargs, f, indent=2)



################################################################## main runner

@click.command()
@click.option("--run-name", default='test', type=str)
@click.option("--dataset", required=True, type=str, help="csv file contain all info for one target")
@click.option("--target", required=True, type=str, help="target name for all the designed binders")
@click.option("--cutoff", default="full", type=str, help="target cutoff ranges (top x); e.g. 'full,30,50'")
@click.option("--cutoff-map", default="___", type=str, help=".json file that stored residue indices for different cutoff range")
@click.option("--template-root", default="___", type=str, help="root folder for all template" )
@click.option("--out-root", default='/hpf/projects/mtyers/ningrui/BindCraftCP/output/refold_val', type=str, help="root folder for output")
@click.option("--templated/--no-templated", default=True, help="whether to use binder template for refolding; slightly different from original usage") #NOTE: NOT finished changing
def main(
    run_name,
    dataset,
    target,
    cutoff,
    cutoff_map,
    template_root,
    out_root,
    templated: bool
) -> DesignBatch:
    
    print("yay you survived here!")

    df = pd.read_csv(dataset)
    cutoff_list = _parseCutoffs(cutoff)
    cutoff_res = _loadCutoffMap(cutoff_map, target)

    run_dir = Path(out_root) / run_name
    metrics_dir = run_dir / 'metrics'
    refold_dir = run_dir / 'refold'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    refold_dir.mkdir(parents=True, exist_ok=True)

    run_params = {"run_name": run_name,
                  "dataset": dataset, 
                  "target": target,
                  "cutoff": cutoff_list,
                  "cutoff_map": cutoff_map,
                  "template_root": template_root,
                  "out_root": out_root,
                  "templated": templated}
    _saveParams(run_dir, **run_params)

    # load and patch settings
    settings = _loadSettings(target_settings, binder_settings, filter_settings, advanced_settings)
    settings.adv["templated"] = templated
    settings.adv["refold_stem"] = "refold"
    settings.binder_settings.design_path = str(run_dir)
    ## NOTE: patch & adapt chain info for RMSD
    settings.target_settings.full_target_chain = "B"
    settings.target_settings.new_binder_chain = "A"
    settings.binder_settings.binder_name = "org_params" # for saved settings file name

    #breakpoint()
    refolding = RefoldValidation(settings)
    batch = DesignBatch(cache_dir=metrics_dir)

    template_root = Path(template_root)
    template_store = template_root / target

    rows_meta = []
    
    #breakpoint()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="rows"):
        binder = row["description"]
        binder_seq = row["binder_seq"]
        template_full = Path(row["design_complex_pdb"]) # NOTE: should be inside template_store

        # NOTE: cutoff = top k
        for cutoff in cutoff_list: 
            cutoff_label = "full" if cutoff is None else f"top_{cutoff}"
            record_id = f"{binder}_{cutoff_label}"

            if cutoff is None:
                template_pdb = str(template_full)
            else:
                if cutoff_label not in cutoff_res:
                    raise KeyError(f"Missing target cutoff residue indices for **{target}:{cutoff_label}** in {cutoff_map}")
                template_pdb = template_full.with_name(template_full.stem + f"_{cutoff_label}" + template_full.suffix)
                if not template_pdb.is_file():
                    _cutChain(pdb_in=template_full,
                              pdb_out=template_pdb,
                              keep_resi=cutoff_res[cutoff_label],
                              chain="B") # NOTE: targe=B in org design complex
            
            batch.add_record(DesignRecord(
                id=record_id,
                sequence=binder_seq,
                pdb_files={"template": str(template_pdb)}
            ))

            rows_meta.append({"target": target,
                              "record_id": record_id,
                              "binder": binder,
                              "binder_seq": binder_seq,
                              "cutoff": cutoff_label,
                              "template_full": template_full,
                              "template_cut": template_pdb})
            
    batch = refolding.run(batch)

    rows_metrics = []
    for rec_id, design_rec in batch.records.items():
        refold_metrics = flatten_dict(design_rec.metrics["refold"])
        refold_metrics["record_id"] = rec_id
        rows_metrics.append(refold_metrics)

    out_meta = pd.DataFrame(rows_meta).drop_duplicates(subset=["record_id"])
    out_metrics = pd.DataFrame(rows_metrics)

    out = out_meta.merge(out_metrics, on="record_id", how="left")
    out_path = run_dir / "refold_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"All refolding down, final data summary write to: {out_path}")
        

if __name__ == "__main__":
    main()

# NOTE: need normalize metrics?
# TODO: remove monomer
