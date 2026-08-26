#!/usr/bin/env python
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epitopecraft.steps.scorer.mdana import cal_gyration_metrics


COLUMNS = [
    "design_id",
    "trajectory_id",
    "pass_loose_criterion",
    "pass_final_criterion",
    "hallu_pLDDT",
    "hallu_i_pAE",
    "model_pLDDT",
    "model_i_pAE",
    "model_binder_RMSD",
    "best_model_binder_RMSD",
    "binder_energy_score",
    "dG",
    "dSASA",
    "shape_complementarity",
    "n_interface_hbonds",
    "surface_hydrophobicity",
    "kappa2",
    "pi_fold",
    "unsat_ppi_polar",
]


SYSTEMS = {
    "WDR5-t": {
        "path": Path("output/WDR5-t"),
        "outfile": "WDR5-t_design_metrics_slim.csv",
        "exclude_pattern": re.compile(r"(^t50-|-satnc\d+$)"),
    },
    "TcdB-ext": {
        "path": Path("output/TcdB-ext"),
        "outfile": "TcdB-ext_design_metrics_slim.csv",
        "exclude_pattern": None,
    },
}


DERIVED_SUFFIX_RE = re.compile(r"-(?:mpnn|rmC|satnc)\d+")


def trajectory_id(design_id: str) -> str:
    previous = None
    current = design_id
    while previous != current:
        previous = current
        current = DERIVED_SUFFIX_RE.sub("", current)
    return current


def get_nested(d: dict, path: str, default=None):
    cur = d
    for key in path.split(":"):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def best_multimer_metrics(metrics: dict) -> tuple[float | None, float | None]:
    candidates = []
    for model in ["multimer-1", "multimer-2"]:
        plddt = get_nested(metrics, f"refold:{model}:pLDDT")
        ipae = get_nested(metrics, f"refold:{model}:i-pAE")
        if plddt is not None and ipae is not None:
            candidates.append((ipae, plddt))
    if not candidates:
        return None, None
    ipae, plddt = min(candidates, key=lambda x: x[0])
    return pldt_float(plddt), pldt_float(ipae)


def pldt_float(value):
    return None if value is None else float(value)


def pass_refold_model_1_2(metrics: dict) -> bool:
    return (
        (get_nested(metrics, "refold:multimer-1:pLDDT", 0) >= 0.8)
        and (get_nested(metrics, "refold:multimer-1:i-pAE", 1) <= 0.35)
        and (get_nested(metrics, "refold:multimer-2:pLDDT", 0) >= 0.8)
        and (get_nested(metrics, "refold:multimer-2:i-pAE", 1) <= 0.35)
    )


def sum_track(record: dict, name: str):
    values = record.get("ana_tracks", {}).get(name)
    if values is None:
        return None
    return int(sum(values))


def build_table(system: str, cfg: dict) -> pd.DataFrame:
    metrics_dir = cfg["path"] / "metrics"
    records = {}
    for fp in metrics_dir.glob("*.json"):
        if fp.name == "metrics.json":
            continue
        if cfg["exclude_pattern"] and cfg["exclude_pattern"].search(fp.stem):
            continue
        with open(fp) as f:
            record = json.load(f)
        records[record["id"]] = record

    rows = []
    for design_id, record in sorted(records.items()):
        traj_id = trajectory_id(design_id)
        base_record = records.get(traj_id, record)
        metrics = record.get("metrics", {})
        base_metrics = base_record.get("metrics", {})

        hallu_plddt = get_nested(base_metrics, "halu:pLDDT")
        hallu_ipae = get_nested(base_metrics, "halu:i-pAE")
        model_plddt, model_ipae = best_multimer_metrics(metrics)
        best_binder_rmsd = get_nested(metrics, "refold:best:binder_rmsd")
        binder_rmsd = best_binder_rmsd

        binder_score = get_nested(metrics, "rst:binder_score")
        dg = get_nested(metrics, "rst:dG")
        dsasa = get_nested(metrics, "rst:dSASA")
        shape = get_nested(metrics, "rst:shape_complementarity")
        hbond = get_nested(metrics, "rst:hbond")
        surf_hydro = get_nested(metrics, "aux:surf_hydro")
        kappa2 = get_nested(metrics, "aux:kappa2")
        if kappa2 is None:
            pdbs = record.get("pdb_files", {})
            pdb_for_kappa = pdbs.get("refold:best:relax") or pdbs.get("refold:best")
            if pdb_for_kappa:
                try:
                    kappa2 = float(cal_gyration_metrics(pdb_for_kappa, ligand_chain="B")["kappa2"])
                except Exception:
                    kappa2 = None
        pi_fold = metrics.get("pi-fold")
        unsat_ppi_polar = sum_track(record, "unsat_ppi_polar")

        pass_loose = (
            hallu_plddt is not None
            and hallu_plddt >= 0.8
            and hallu_ipae is not None
            and hallu_ipae <= 0.35
            and pass_refold_model_1_2(metrics)
            and best_binder_rmsd is not None
            and best_binder_rmsd <= 3.5
        )
        pass_final = (
            pass_loose
            and binder_rmsd is not None
            and binder_rmsd <= 3.5
            and binder_score is not None
            and binder_score <= 0
            and dg is not None
            and dg <= 0
            and dsasa is not None
            and dsasa >= 1
            and shape is not None
            and shape >= 0.6
            and hbond is not None
            and hbond >= 3
            and surf_hydro is not None
            and surf_hydro <= 0.65
            and kappa2 is not None
            and kappa2 <= 0.4
            and pi_fold is not None
            and pi_fold >= 4.5
            and unsat_ppi_polar is not None
            and unsat_ppi_polar <= 4
        )

        rows.append(
            {
                "design_id": design_id,
                "trajectory_id": traj_id,
                "pass_loose_criterion": bool(pass_loose),
                "pass_final_criterion": bool(pass_final),
                "hallu_pLDDT": hallu_plddt,
                "hallu_i_pAE": hallu_ipae,
                "model_pLDDT": model_plddt,
                "model_i_pAE": model_ipae,
                "model_binder_RMSD": binder_rmsd,
                "best_model_binder_RMSD": best_binder_rmsd,
                "binder_energy_score": binder_score,
                "dG": dg,
                "dSASA": dsasa,
                "shape_complementarity": shape,
                "n_interface_hbonds": hbond,
                "surface_hydrophobicity": surf_hydro,
                "kappa2": kappa2,
                "pi_fold": pi_fold,
                "unsat_ppi_polar": unsat_ppi_polar,
            }
        )

    df = pd.DataFrame(rows, columns=COLUMNS)
    out = cfg["path"] / cfg["outfile"]
    df.to_csv(out, index=False)
    print(
        system,
        out,
        df.shape,
        "loose_rows",
        int(df["pass_loose_criterion"].sum()),
        "loose_traj",
        df.loc[df["pass_loose_criterion"], "trajectory_id"].nunique(),
        "final_rows",
        int(df["pass_final_criterion"].sum()),
        "final_traj",
        df.loc[df["pass_final_criterion"], "trajectory_id"].nunique(),
    )
    return df


def main() -> None:
    for system, cfg in SYSTEMS.items():
        build_table(system, cfg)


if __name__ == "__main__":
    main()
