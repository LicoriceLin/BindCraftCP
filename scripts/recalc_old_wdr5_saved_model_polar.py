#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epitopecraft.steps.scorer.mdana import annot_polar_occupy, cal_gyration_metrics
from epitopecraft.steps.scorer.propka import propka_single
from epitopecraft.utils import DesignRecord


OLD_BASE = Path("/hpf/projects/mtyers/richardh/BindCraft/designs")
DEFAULT_CACHE = Path("/tmp/bc_wdr5_saved_model_recalc_polar_cache.json")
PREV_CACHE = Path("/tmp/bc_wdr5_filter_kappa_pi_cache.json")


def traj_id(design: str) -> str:
    return re.sub(r"_mpnn\d+(?:_model\d+)?$", "", str(design))


def parse_interface(interface: str, n: int) -> list[int]:
    ppi = [0] * n
    if isinstance(interface, str):
        for token in interface.split(","):
            match = re.match(r"[A-Za-z](\d+)$", token.strip())
            if match:
                i = int(match.group(1)) - 1
                if 0 <= i < n:
                    ppi[i] = 1
    return ppi


def pdb_index(path: Path) -> dict[str, tuple[Path, int]]:
    index = {}
    for fp in path.rglob("*.pdb"):
        match = re.match(r"(.+)_model([1-5])$", fp.stem)
        if not match:
            continue
        design, model = match.group(1), int(match.group(2))
        current = index.get(design)
        if current is None or (
            "Rejected/" in current[0].as_posix() and "Rejected/" not in fp.as_posix()
        ):
            index[design] = (fp, model)
    return index


def collect_rows() -> list[dict]:
    rows = []
    for run in ["WDR5_1", "WDR5_2"]:
        path = OLD_BASE / run
        pdbs = pdb_index(path)
        traj = pd.read_csv(path / "trajectory_stats.csv")
        mpnn = pd.read_csv(path / "mpnn_design_stats.csv")
        mpnn["traj"] = mpnn["Design"].map(traj_id)
        traj_ok = dict(zip(traj["Design"], (traj["pLDDT"] >= 0.8) & (traj["i_pAE"] <= 0.35)))
        for _, r in mpnn.iterrows():
            design = str(r["Design"])
            if design not in pdbs:
                continue
            fp, model = pdbs[design]
            if not traj_ok.get(r["traj"], False):
                continue
            if not (
                r["1_pLDDT"] >= 0.8
                and r["2_pLDDT"] >= 0.8
                and r["1_i_pAE"] <= 0.35
                and r["2_i_pAE"] <= 0.35
            ):
                continue
            if r[f"{model}_Binder_RMSD"] > 3.5:
                continue
            checks = {
                "binder_score": r[f"{model}_Binder_Energy_Score"] <= 0,
                "dG": r[f"{model}_dG"] <= 0,
                "dSASA": r[f"{model}_dSASA"] >= 1,
                "shape_complementarity": r[f"{model}_ShapeComplementarity"] >= 0.6,
                "hbond": r[f"{model}_n_InterfaceHbonds"] >= 3,
                "surf_hydro": r[f"{model}_Surface_Hydrophobicity"] <= 0.65,
            }
            if not all(checks.values()):
                continue
            rows.append(
                {
                    "run": run,
                    "design": design,
                    "traj": r["traj"],
                    "model": model,
                    "pdb": str(fp),
                    "sequence": r["Sequence"],
                    "interface": r["InterfaceResidues"],
                    "old_unsat_hbond": float(r[f"{model}_n_InterfaceUnsatHbonds"]),
                    **checks,
                }
            )
    return rows


def load_cache(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {}


def save_cache(path: Path, cache: dict) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def process(args: argparse.Namespace) -> None:
    rows = collect_rows()
    cache = load_cache(args.cache)
    previous = load_cache(PREV_CACHE)
    done_new = 0
    for row in rows:
        fp = row["pdb"]
        c = cache.get(fp) or previous.get(fp, {}).copy()
        complete = (
            "kappa2" in c
            and "pi_fold" in c
            and (
                not (c.get("kappa2") <= 0.4 and c.get("pi_fold") >= 4.5)
                or "unsat_ppi_polar" in c
                or "polar_error" in c
            )
        )
        if complete:
            cache[fp] = c
            continue
        if args.max_new is not None and done_new >= args.max_new:
            break
        if "kappa2" not in c:
            try:
                c["kappa2"] = float(cal_gyration_metrics(fp, ligand_chain="B")["kappa2"])
            except Exception as exc:
                c["kappa2_error"] = repr(exc)
        if "pi_fold" not in c:
            try:
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    c["pi_fold"] = float(propka_single(fp, binder_chain="B")["pi-fold"])
            except Exception as exc:
                c["pi_fold_error"] = repr(exc)
        if c.get("kappa2") is not None and c.get("pi_fold") is not None:
            if c["kappa2"] <= 0.4 and c["pi_fold"] >= 4.5 and "unsat_ppi_polar" not in c:
                try:
                    record = DesignRecord(
                        id=f"{row['design']}_model{row['model']}",
                        sequence=row["sequence"],
                        pdb_files={"relax": fp},
                        ana_tracks={"ppi": parse_interface(row["interface"], len(row["sequence"]))},
                    )
                    annot_polar_occupy(record, pdb_to_take="relax", ligand_chain="B", target_chain="A")
                    c["h_bond_count"] = int(sum(record.ana_tracks["h-bond"]))
                    c["salt_bridge_count"] = int(sum(record.ana_tracks["salt-bridge"]))
                    c["unsat_ppi_polar"] = int(sum(record.ana_tracks["unsat_ppi_polar"]))
                    c["ppi_count"] = int(sum(record.ana_tracks["ppi"]))
                except Exception as exc:
                    c["polar_error"] = repr(exc)
        cache[fp] = c
        done_new += 1
        if done_new % 10 == 0:
            save_cache(args.cache, cache)
    save_cache(args.cache, cache)
    summarize(rows, cache)


def summarize(rows: list[dict], cache: dict) -> None:
    out = []
    for row in rows:
        c = cache.get(row["pdb"], {})
        item = {
            **row,
            **{k: c.get(k) for k in ["kappa2", "pi_fold", "unsat_ppi_polar", "polar_error"]},
        }
        item["pass_kappa"] = item.get("kappa2") is not None and item["kappa2"] <= 0.4
        item["pass_pi"] = item.get("pi_fold") is not None and item["pi_fold"] >= 4.5
        item["pass_old_unsat_le4"] = item["old_unsat_hbond"] <= 4
        item["pass_new_unsat_le4"] = item.get("unsat_ppi_polar") is not None and item["unsat_ppi_polar"] <= 4
        item["pass_new_unsat_eq0"] = item.get("unsat_ppi_polar") == 0
        out.append(item)
    df = pd.DataFrame(out)
    print("saved-model candidates", len(df), "traj", df["traj"].nunique())
    masks = {
        "kappa+pi": df.pass_kappa & df.pass_pi,
        "old_unsat<=4 after kappa+pi": df.pass_kappa & df.pass_pi & df.pass_old_unsat_le4,
        "new_unsat_ppi_polar<=4 after kappa+pi": df.pass_kappa & df.pass_pi & df.pass_new_unsat_le4,
        "new_unsat_ppi_polar==0 after kappa+pi": df.pass_kappa & df.pass_pi & df.pass_new_unsat_eq0,
    }
    for name, mask in masks.items():
        print(
            name,
            "rows",
            int(mask.sum()),
            "designs",
            df.loc[mask, "design"].nunique(),
            "traj",
            df.loc[mask, "traj"].nunique(),
            "rate",
            f"{df.loc[mask, 'traj'].nunique() / 571 * 100:.2f}%",
        )
    kpi = df.pass_kappa & df.pass_pi
    print("polar missing after kappa+pi", int(df.loc[kpi, "unsat_ppi_polar"].isna().sum()))
    print(df.loc[kpi, "unsat_ppi_polar"].value_counts(dropna=False).sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-new", type=int, default=None)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
