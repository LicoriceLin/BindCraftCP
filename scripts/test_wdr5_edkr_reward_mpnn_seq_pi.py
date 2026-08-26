#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from epitopecraft.steps import Graft, MPNN
from epitopecraft.utils import DesignBatch, GlobalSettings, NEST_SEP


BASE_EXCLUDE_RE = re.compile(r"-(?:mpnn|rmC|satnc\w*)\d*$")
AFTER_REFOLD_FILTERS = (
    "filter:refold:multimer-1:pLDDT",
    "filter:refold:multimer-1:i-pAE",
    "filter:refold:multimer-2:pLDDT",
    "filter:refold:multimer-2:i-pAE",
    "filter:refold:best:binder_rmsd",
    "filter:aux:kappa2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="output/WDR5-t/backup-settinga.yaml")
    parser.add_argument("--design-path", default="output/WDR5-t")
    parser.add_argument(
        "--recipe", default="output/BC_baseline_ana/mpnn-reward-DEKR-test.json"
    )
    parser.add_argument("--prefix", default="edkrreward_test:")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--max-sequences", type=int, default=6)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--exclude-id-regex", default=r"^t50-")
    parser.add_argument(
        "--out-prefix",
        default="output/BC_baseline_ana/wdr5_edkr_reward_mpnn_seq_pi_test",
    )
    return parser.parse_args()


def is_base_record(record_id: str) -> bool:
    return BASE_EXCLUDE_RE.search(record_id) is None


def passed_refold(record) -> bool:
    return all(record.get_metrics(key) is True for key in AFTER_REFOLD_FILTERS)


def select_ids(batch: DesignBatch, exclude_id_regex: str, limit: int | None) -> list[str]:
    exclude_re = re.compile(exclude_id_regex) if exclude_id_regex else None
    selected = [
        record_id
        for record_id, record in sorted(batch.records.items())
        if is_base_record(record_id)
        and (exclude_re is None or not exclude_re.search(record_id))
        and passed_refold(record)
    ]
    return selected if limit is None else selected[:limit]


def configure_mpnn(args: argparse.Namespace) -> tuple[GlobalSettings, MPNN]:
    settings = GlobalSettings.from_file(args.settings)
    settings.binder_settings.design_path = args.design_path
    adv = settings.adv
    adv["mpnn_bias_recipe"] = args.recipe
    adv["n_mpnn_samples"] = args.samples
    adv["max_mpnn_sequences"] = args.max_sequences
    adv["mpnn-prefix"] = args.prefix
    adv["overwrite"] = False

    graft = Graft(settings)
    graft.config_pdb_purge(adv.setdefault("graft_stem", "graft"))
    template_pdb = graft.pdb_to_add[0]

    mpnn = MPNN(settings)
    mpnn.config_pdb_input_key(template_pdb)
    mpnn.config_metrics_prefix(adv["mpnn-prefix"])
    adv["mpnn_binder_chain"] = settings.target_settings.new_binder_chain
    adv["mpnn_target_chain"] = settings.target_settings.full_target_chain
    mpnn.init_penalty_recipes()
    return settings, mpnn


def sequence_pi(sequence: str) -> float:
    return float(ProteinAnalysis(sequence).isoelectric_point())


def sequence_features(sequence: str) -> dict:
    length = len(sequence)
    ed = sequence.count("E") + sequence.count("D")
    kr = sequence.count("K") + sequence.count("R")
    return {
        "length": length,
        "seq_pi": sequence_pi(sequence),
        "EDKR_frac": (ed + kr) / length,
        "ED_frac": ed / length,
        "KR_frac": kr / length,
        "net_KR_minus_ED_frac": (kr - ed) / length,
    }


def collect_existing(batch: DesignBatch, base_ids: list[str], suffix: str, version: str):
    rows = []
    pattern_by_base = {
        base_id: re.compile(rf"^{re.escape(base_id)}-{re.escape(suffix)}\d+$")
        for base_id in base_ids
    }
    for record_id, record in batch.records.items():
        for base_id, pattern in pattern_by_base.items():
            if pattern.search(record_id):
                rows.append(
                    {
                        "version": version,
                        "base_id": base_id,
                        "design_id": record_id,
                        "sequence": record.sequence,
                        **sequence_features(record.sequence),
                    }
                )
                break
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version, sub in df.groupby("version", sort=False):
        rows.append(
            {
                "version": version,
                "n_rows": len(sub),
                "n_base": sub["base_id"].nunique(),
                "seq_pi_ge_4.5": float((sub["seq_pi"] >= 4.5).mean()),
                "seq_pi_le_9.5": float((sub["seq_pi"] <= 9.5).mean()),
                "seq_pi_in_4.5_9.5": float(
                    ((sub["seq_pi"] >= 4.5) & (sub["seq_pi"] <= 9.5)).mean()
                ),
                "seq_pi_mean": float(sub["seq_pi"].mean()),
                "seq_pi_q25": float(sub["seq_pi"].quantile(0.25)),
                "seq_pi_median": float(sub["seq_pi"].median()),
                "seq_pi_q75": float(sub["seq_pi"].quantile(0.75)),
                "EDKR_frac_mean": float(sub["EDKR_frac"].mean()),
                "ED_frac_mean": float(sub["ED_frac"].mean()),
                "KR_frac_mean": float(sub["KR_frac"].mean()),
                "net_KR_minus_ED_frac_mean": float(
                    sub["net_KR_minus_ED_frac"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    batch = DesignBatch.from_cache(Path(args.design_path) / "metrics")
    base_ids = select_ids(batch, args.exclude_id_regex, args.limit)
    print(f"Selected {len(base_ids)} WDR5 base records.")
    if not base_ids:
        return
    print("First IDs:", ", ".join(base_ids[:10]))

    _, mpnn = configure_mpnn(args)
    test_rows = []
    for i, base_id in enumerate(base_ids, 1):
        if i == 1 or i % 10 == 0:
            print(f"MPNN {i}/{len(base_ids)}: {base_id}", flush=True)
        for record in mpnn.run_mpnn(batch.records[base_id])[1:]:
            test_rows.append(
                {
                    "version": "DEKR reward test",
                    "base_id": base_id,
                    "design_id": record.id,
                    "sequence": record.sequence,
                    **sequence_features(record.sequence),
                }
            )

    rows = []
    rows.extend(collect_existing(batch, base_ids, "mpnn", "new default mpnn"))
    rows.extend(collect_existing(batch, base_ids, "satncplus", "new satncplus"))
    rows.extend(test_rows)

    df = pd.DataFrame(rows)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows_out = out_prefix.with_suffix(".rows.csv")
    summary_out = out_prefix.with_suffix(".summary.csv")
    df.to_csv(rows_out, index=False)
    summary = summarize(df)
    summary.to_csv(summary_out, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {rows_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
