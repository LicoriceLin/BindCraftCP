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
OUT_DIR = Path("output/BC_baseline_ana")


RUN_GROUPS = {
    "old_WDR5": ["WDR5_1", "WDR5_2"],
    "old_TcdB": ["TcdB"],
}


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


def load_filter_result(path: Path, filename: str, column_name: str) -> pd.DataFrame:
    fp = path / filename
    if not fp.exists():
        return pd.DataFrame(columns=["Design", column_name])
    df = pd.read_csv(fp)
    if "Filter_Result" not in df:
        return pd.DataFrame(columns=["Design", column_name])
    return df[["Design", "Filter_Result"]].rename(columns={"Filter_Result": column_name})


def collect_saved_pdbs(path: Path) -> dict[str, tuple[Path, int, str]]:
    records = {}
    for fp in path.rglob("*.pdb"):
        match = re.match(r"(.+)_model([1-5])$", fp.stem)
        if not match:
            continue
        design = match.group(1)
        model = int(match.group(2))
        bucket = fp.relative_to(path).parts[0]
        current = records.get(design)
        if current is None or (current[2] == "Rejected" and bucket != "Rejected"):
            records[design] = (fp, model, bucket)
    return records


def prepare_run_rows(run: str) -> list[dict]:
    path = OLD_BASE / run
    traj = pd.read_csv(path / "trajectory_stats.csv")
    mpnn = pd.read_csv(path / "mpnn_design_stats.csv")
    mpnn["trajectory_id"] = mpnn["Design"].map(traj_id)

    traj_cols = {
        "Design": "trajectory_id",
        "pLDDT": "hallu_pLDDT",
        "i_pAE": "hallu_i_pAE",
        "pAE": "hallu_pAE",
        "Target_RMSD": "hallu_target_rmsd",
        "TrajectoryTime": "trajectory_time",
        "Notes": "trajectory_notes",
    }
    traj_small = traj[[c for c in traj_cols if c in traj.columns]].rename(columns=traj_cols)
    df = mpnn.merge(traj_small, on="trajectory_id", how="left")

    for filename, col in [
        ("mpnn_design_stats_default_filters_results.csv", "bc_default_filter_result"),
        ("mpnn_design_stats_rfdiffusion_filters_results.csv", "rfdiffusion_filter_result"),
        (
            "mpnn_design_stats_default_filters_results_rfdiffusion_filters_results.csv",
            "bc_default_then_rfdiffusion_filter_result",
        ),
        ("final_design_stats_default_filters_results.csv", "final_default_filter_result"),
    ]:
        df = df.merge(load_filter_result(path, filename, col), on="Design", how="left")

    final = pd.read_csv(path / "final_design_stats.csv") if (path / "final_design_stats.csv").exists() else pd.DataFrame()
    final_designs = set(final["Design"]) if "Design" in final else set()
    saved = collect_saved_pdbs(path)

    rows = []
    for _, row in df.iterrows():
        design = str(row["Design"])
        if design not in saved:
            continue
        pdb_path, model, bucket = saved[design]
        prefix = f"{model}_"

        def value(name: str, default=None):
            return row.get(prefix + name, default)

        def avg_value(name: str, default=None):
            return row.get("Average_" + name, default)

        cheap = {
            "pass_binder_score": value("Binder_Energy_Score") <= 0,
            "pass_dG": value("dG") <= 0,
            "pass_dSASA": value("dSASA") >= 1,
            "pass_shape_complementarity": value("ShapeComplementarity") >= 0.6,
            "pass_hbond": value("n_InterfaceHbonds") >= 3,
            "pass_surface_hydrophobicity": value("Surface_Hydrophobicity") <= 0.65,
        }
        two_model_refold_pass = (
            row.get("1_pLDDT", 0) >= 0.8
            and row.get("2_pLDDT", 0) >= 0.8
            and row.get("1_i_pAE", 1) <= 0.35
            and row.get("2_i_pAE", 1) <= 0.35
        )
        hallu_pass = row.get("hallu_pLDDT", 0) >= 0.8 and row.get("hallu_i_pAE", 1) <= 0.35
        best_rmsd = min(row.get("1_Binder_RMSD", 999), row.get("2_Binder_RMSD", 999))
        saved_model_rmsd = value("Binder_RMSD", 999)

        rows.append(
            {
                "run": run,
                "design_id": design,
                "trajectory_id": row["trajectory_id"],
                "saved_model": model,
                "saved_bucket": bucket,
                "pdb_path": str(pdb_path),
                "in_final_design_stats": design in final_designs,
                "sequence": row.get("Sequence"),
                "length": row.get("Length"),
                "seed": row.get("Seed"),
                "helicity": row.get("Helicity"),
                "target_hotspot": row.get("Target_Hotspot"),
                "interface_residues": row.get("InterfaceResidues"),
                "hallu_pLDDT": row.get("hallu_pLDDT"),
                "hallu_i_pAE": row.get("hallu_i_pAE"),
                "model_pLDDT": value("pLDDT"),
                "model_i_pAE": value("i_pAE"),
                "model_binder_pLDDT": value("Binder_pLDDT"),
                "model_binder_RMSD": saved_model_rmsd,
                "best_model_binder_RMSD": best_rmsd,
                "average_pLDDT": avg_value("pLDDT"),
                "average_i_pAE": avg_value("i_pAE"),
                "average_binder_RMSD": avg_value("Binder_RMSD"),
                "mpnn_score": row.get("MPNN_score"),
                "mpnn_seq_recovery": row.get("MPNN_seq_recovery"),
                "binder_energy_score": value("Binder_Energy_Score"),
                "surface_hydrophobicity": value("Surface_Hydrophobicity"),
                "shape_complementarity": value("ShapeComplementarity"),
                "dG": value("dG"),
                "dSASA": value("dSASA"),
                "n_interface_hbonds": value("n_InterfaceHbonds"),
                "old_n_interface_unsat_hbonds": value("n_InterfaceUnsatHbonds"),
                "bc_default_filter_result": row.get("bc_default_filter_result"),
                "rfdiffusion_filter_result": row.get("rfdiffusion_filter_result"),
                "bc_default_then_rfdiffusion_filter_result": row.get(
                    "bc_default_then_rfdiffusion_filter_result"
                ),
                "final_default_filter_result": row.get("final_default_filter_result"),
                "pass_hallu_plddt_ipae": hallu_pass,
                "pass_refold_model1_2_plddt_ipae": two_model_refold_pass,
                "pass_best_binder_rmsd": best_rmsd <= 3.5,
                "pass_saved_model_binder_rmsd": saved_model_rmsd <= 3.5,
                "pass_loose_corrected": hallu_pass and two_model_refold_pass and best_rmsd <= 3.5,
                **cheap,
                "pass_cheap_except_unsat_kappa_pi": all(cheap.values()),
                "pass_old_unsat_hbond_le4": value("n_InterfaceUnsatHbonds") <= 4,
            }
        )
    return rows


def load_cache(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def add_computed_metrics(rows: list[dict], cache_path: Path, max_new: int | None) -> list[dict]:
    cache = load_cache(cache_path)
    done_new = 0
    for row in rows:
        should_compute = (
            row["pass_loose_corrected"]
            and row["pass_saved_model_binder_rmsd"]
            and row["pass_cheap_except_unsat_kappa_pi"]
        )
        fp = row["pdb_path"]
        c = cache.get(fp, {})
        if should_compute:
            complete = (
                "kappa2" in c
                and "pi_fold" in c
                and (
                    not (c.get("kappa2") <= 0.4 and c.get("pi_fold") >= 4.5)
                    or "unsat_ppi_polar" in c
                    or "polar_error" in c
                )
            )
            if not complete:
                if max_new is not None and done_new >= max_new:
                    continue
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
                                id=f"{row['design_id']}_model{row['saved_model']}",
                                sequence=row["sequence"],
                                pdb_files={"relax": fp},
                                ana_tracks={
                                    "ppi": parse_interface(
                                        row["interface_residues"], len(row["sequence"])
                                    )
                                },
                            )
                            annot_polar_occupy(
                                record, pdb_to_take="relax", ligand_chain="B", target_chain="A"
                            )
                            c["polar_h_bond_count"] = int(sum(record.ana_tracks["h-bond"]))
                            c["polar_salt_bridge_count"] = int(
                                sum(record.ana_tracks["salt-bridge"])
                            )
                            c["unsat_ppi_polar"] = int(
                                sum(record.ana_tracks["unsat_ppi_polar"])
                            )
                            c["ppi_count"] = int(sum(record.ana_tracks["ppi"]))
                        except Exception as exc:
                            c["polar_error"] = repr(exc)
                cache[fp] = c
                done_new += 1
                if done_new % 10 == 0:
                    save_cache(cache_path, cache)
        row["computed_metrics_needed"] = should_compute
        row["kappa2"] = c.get("kappa2")
        row["pi_fold"] = c.get("pi_fold")
        row["polar_h_bond_count"] = c.get("polar_h_bond_count")
        row["polar_salt_bridge_count"] = c.get("polar_salt_bridge_count")
        row["unsat_ppi_polar"] = c.get("unsat_ppi_polar")
        row["ppi_count"] = c.get("ppi_count")
        row["computed_metric_error"] = "; ".join(
            str(c[k]) for k in ["kappa2_error", "pi_fold_error", "polar_error"] if k in c
        )
        row["pass_kappa2_le0.4"] = row["kappa2"] is not None and row["kappa2"] <= 0.4
        row["pass_pi_fold_ge4.5"] = row["pi_fold"] is not None and row["pi_fold"] >= 4.5
        row["pass_unsat_ppi_polar_le4"] = (
            row["unsat_ppi_polar"] is not None and row["unsat_ppi_polar"] <= 4
        )
        row["pass_current_old_unsat"] = (
            row["pass_loose_corrected"]
            and row["pass_saved_model_binder_rmsd"]
            and row["pass_cheap_except_unsat_kappa_pi"]
            and row["pass_old_unsat_hbond_le4"]
            and row["pass_kappa2_le0.4"]
            and row["pass_pi_fold_ge4.5"]
        )
        row["pass_current_recalc_polar_unsat"] = (
            row["pass_loose_corrected"]
            and row["pass_saved_model_binder_rmsd"]
            and row["pass_cheap_except_unsat_kappa_pi"]
            and row["pass_kappa2_le0.4"]
            and row["pass_pi_fold_ge4.5"]
            and row["pass_unsat_ppi_polar_le4"]
        )
    save_cache(cache_path, cache)
    return rows


def write_readme(out_dir: Path) -> None:
    text = """# Old BindCraft Baseline Analysis Tables

Generated from `/hpf/projects/mtyers/richardh/BindCraft/designs`.

Each CSV row is one concrete saved design PDB (`*_modelX.pdb`). `old_WDR5` combines
`WDR5_1` and `WDR5_2`; `old_TcdB` is the old TcdB run.

Important flags:

- `pass_loose_corrected`: hallucination pLDDT/i-pAE, refold model 1/2 pLDDT/i-pAE, and best binder RMSD.
- `pass_cheap_except_unsat_kappa_pi`: interface/energy filters except unsat hbond, kappa2, and pi-fold.
- `kappa2`, `pi_fold`, `unsat_ppi_polar`: newly computed from the saved relaxed PDB when needed.
- `pass_current_old_unsat`: current filter using old `n_InterfaceUnsatHbonds <= 4`.
- `pass_current_recalc_polar_unsat`: current filter replacing old unsat hbond with `AnnotPolarOccupy` `unsat_ppi_polar <= 4`.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=RUN_GROUPS.keys(), default=None)
    parser.add_argument("--max-new", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = [args.group] if args.group else list(RUN_GROUPS)
    for group in groups:
        rows = []
        for run in RUN_GROUPS[group]:
            rows.extend(prepare_run_rows(run))
        cache_path = OUT_DIR / f"{group}_computed_metrics_cache.json"
        rows = add_computed_metrics(rows, cache_path, args.max_new)
        df = pd.DataFrame(rows)
        df = df.sort_values(["run", "trajectory_id", "design_id", "saved_model"])
        out_csv = OUT_DIR / f"{group}_design_metrics.csv"
        df.to_csv(out_csv, index=False)
        summary = {
            "rows": int(len(df)),
            "trajectories": int(df["trajectory_id"].nunique()),
            "loose_rows": int(df["pass_loose_corrected"].sum()),
            "loose_trajectories": int(df.loc[df["pass_loose_corrected"], "trajectory_id"].nunique()),
            "current_old_unsat_rows": int(df["pass_current_old_unsat"].sum()),
            "current_old_unsat_trajectories": int(
                df.loc[df["pass_current_old_unsat"], "trajectory_id"].nunique()
            ),
            "current_recalc_polar_rows": int(df["pass_current_recalc_polar_unsat"].sum()),
            "current_recalc_polar_trajectories": int(
                df.loc[df["pass_current_recalc_polar_unsat"], "trajectory_id"].nunique()
            ),
        }
        with open(OUT_DIR / f"{group}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(group, summary)
    write_readme(OUT_DIR)


if __name__ == "__main__":
    main()
