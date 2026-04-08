import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from pymol import cmd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from .basestep import BaseStep, DesignBatch, DesignRecord, GlobalSettings
from .scorer.pymol_utils import hotspots_by_ligand, map_residues_between_pdbs,ResidueKey
from ..utils.preprocess import hotspots_topk_motifs
from warnings import warn

class PseudoHotspot(BaseStep):
    group_palette = [
        "#d1495b",
        "#2e86ab",
        "#3a7d44",
        "#f4a259",
        "#8f2d56",
        "#00798c",
        "#7c6a0a",
        "#6c5b7b",
    ]

    def __init__(self, settings: GlobalSettings):
        super().__init__(settings)
        self.hotspot_df: Optional[pd.DataFrame] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self.method_info: dict = {}
        self.analysis_dir: Optional[Path] = None
        self._residue_map:Optional[Dict[ResidueKey,ResidueKey]] = None
    @property
    def name(self) -> str:
        return "pseudo_hotspot"

    @property
    def _default_pdb_input_key(self) -> str:
        return "halu"

    @property
    def params_to_take(self) -> Tuple[str, ...]:
        return (
            f"{self.name}-analysis-stem",
            f"{self.name}-pdb-input",
            f"{self.name}-freq-threshold",
            f'{self.name}-motif-sizes'
        )

    def process_record(self, input: DesignRecord | None = None) -> DesignRecord | None:
        raise NotImplementedError("Run pseudo_hotspot via process_batch().")

    def process_batch(
        self,
        input: DesignBatch,
        analysis_stem: Optional[str] = None,
        pdb_to_take: Optional[str] = None,
    ) -> DesignBatch:
        if pdb_to_take is not None:
            self.config_pdb_input_key(pdb_to_take)

        analysis_stem = self._set_param(f"{self.name}-analysis-stem", analysis_stem, "hotspot_analysis")
        full_target_chain = self.settings.target_settings.chains or "A"
        design_target_chain = "A"
        # target_chain = self._set_param(f"{self.name}-target-chain", None, self._default_target_chain())
        # ligand_chain = self._set_param(f"{self.name}-ligand-chain", None, self._default_ligand_chain())
        ligand_chain = 'B'
        freq_threshold = float(self._set_param(f"{self.name}-freq-threshold", None, 0.5))
        motif_sizes = self._set_param(
            f"{self.name}-motif-sizes",
            None,
            [50, 100, 150, 200, 250, 300],
        )

        target_pdb = Path(self.settings.target_settings.full_target_pdb)
        if not target_pdb.exists():
            raise FileNotFoundError(f"Missing target PDB: {target_pdb}")

        analysis_dir = self._resolve_analysis_dir(input.cache_dir.parent, analysis_stem)
        hotspot_counter, n_designs = self._collect_hotspot_counts(
            input,
            target_pdb,
            full_target_chain=full_target_chain,
            design_target_chain=design_target_chain,
            ligand_chain=ligand_chain,
        )
        coord_map = self._extract_target_ca_coords(target_pdb, hotspot_counter.keys())
        hotspot_df, summary_df, method_info = self._assign_groups(hotspot_counter, coord_map, n_designs)
        
        self.hotspot_df = hotspot_df
        self.summary_df = summary_df
        self.method_info = method_info
        self.analysis_dir = analysis_dir

        if len(self.hotspot_df)>0:
            binder_name = self.settings.binder_settings.binder_name
            table_csv = analysis_dir / f"{binder_name}_hotspot_table.csv"
            summary_csv = analysis_dir / f"{binder_name}_hotspot_group_summary.csv"
            summary_txt = analysis_dir / f"{binder_name}_hotspot_method.txt"
            out_pse = analysis_dir / f"{binder_name}_hotspot_groups.pse"
            hotspot_txt = analysis_dir / f"{binder_name}_target_hotspot_residues.txt"

            self._rounded_copy(hotspot_df).to_csv(table_csv, index=False, float_format="%.2f")
            self._rounded_copy(summary_df).to_csv(summary_csv, index=False, float_format="%.2f")
            self._write_method_note(summary_txt, method_info, full_target_chain, ligand_chain, n_designs)
            self._color_groups_on_target(target_pdb, hotspot_df, out_pse)

            hotspot_str = self.to_target_hotspot_residues_str(
                freq_threshold=freq_threshold,
                hotspot_df=hotspot_df,
            )
            hotspot_txt.write_text(hotspot_str + "\n")

            hotspot_list = self.to_target_hotspot_residues_list(
                freq_threshold=freq_threshold,
                hotspot_df=hotspot_df,
            )

            hotspots_topk_motifs(
                pdb=self.settings.target_settings.full_target_pdb,
                hotspot_list=hotspot_list,
                output_dir=analysis_dir / f"{binder_name}_motif",
                topk_ranges=motif_sizes,
                stem=binder_name,
            )

            input.metrics[f"{self.metrics_prefix}analysis_dir"] = str(analysis_dir)
            input.metrics[f"{self.metrics_prefix}target_hotspot_residues"] = hotspot_str
            input.metrics[f"{self.metrics_prefix}n_groups"] = int(method_info["n_groups"])
            input.metrics[f"{self.metrics_prefix}pdb_to_take"] = self.pdb_to_take
        else:
            warn('no hotspot identified!')
        return input

    def to_target_hotspot_residues_str(
        self,
        group_id: int = 0,
        freq_threshold: float = 0.5,
        hotspot_df: Optional[pd.DataFrame] = None,
        analysis_stem: Optional[str] = None,
    ) -> str:
        residues = self.to_target_hotspot_residues_list(group_id,freq_threshold,hotspot_df,analysis_stem)
        return ",".join(f"{chain}{resi}" for chain, resi in residues)

    def to_target_hotspot_residues_list(
        self,
        group_id: int = 0,
        freq_threshold: float = 0.5,
        hotspot_df: Optional[pd.DataFrame] = None,
        analysis_stem: Optional[str] = None,
    ) -> list[ResidueKey]:
        if hotspot_df is None:
            hotspot_df = self.hotspot_df
        if hotspot_df is None:
            if analysis_stem is None:
                raise ValueError("Provide hotspot_df or analysis_stem.")
            analysis_dir = Path(analysis_stem)
            binder_name = self.settings.binder_settings.binder_name
            hotspot_df = pd.read_csv(analysis_dir / f"{binder_name}_hotspot_table.csv")

        selected = hotspot_df[hotspot_df["freq"] > freq_threshold].copy()
        if group_id > 0:
            selected = selected[selected["group_id"] == group_id]

        residues = sorted(
            [(str(row.chain), str(row.resi)) for row in selected.itertuples()],
            key=self._residue_sort_key,
        )
        return residues

    def _set_param(self, key: str, value, default):
        if value is not None:
            self.settings.adv[key] = value
        return self.settings.adv.setdefault(key, default)

    # def _default_target_chain(self) -> str:
    #     chain = self.settings.target_settings.full_target_chain or "A"
    #     if "," in chain:
    #         raise NotImplementedError("PseudoHotspot currently supports one target chain at a time.")
    #     return chain

    # def _default_ligand_chain(self) -> str:
    #     if "template" in self.pdb_to_take:
    #         return self.settings.target_settings.new_binder_chain
    #     return self.settings.target_settings.full_binder_chain

    def _resolve_analysis_dir(self, root_dir: Path, analysis_stem: str) -> Path:
        analysis_path = Path(analysis_stem)
        if analysis_path.is_absolute() or str(analysis_path).startswith(str(root_dir)):
            analysis_dir = analysis_path
        else:
            analysis_dir = root_dir / analysis_path
        analysis_dir.mkdir(parents=True, exist_ok=True)
        return analysis_dir

    def _design_pdbs(self, input: DesignBatch) -> list[tuple[str, Path]]:
        records = []
        for record_id, record in sorted(input.records.items()):
            pdb_file = record.pdb_files.get(self.pdb_to_take)
            if pdb_file is None:
                raise KeyError(f"{record_id} missing pdb_files['{self.pdb_to_take}']")
            records.append((record_id, Path(pdb_file)))
        return records

    def _collect_hotspot_counts(
        self,
        input: DesignBatch,
        target_pdb: Path,
        full_target_chain: str,
        design_target_chain: str = 'A',
        ligand_chain: str='B',
    ) -> tuple[Counter, int]:
        design_pdbs = self._design_pdbs(input)
        if not design_pdbs:
            raise ValueError("Empty batch.")
        self._residue_map = map_residues_between_pdbs(
            str(design_pdbs[0][1]),
            str(target_pdb),
            chain_target=design_target_chain,
            chain_ref=full_target_chain,
        )

        hotspot_counter: Counter = Counter()
        cmd.delete("all")
        for record_id, design_pdb in tqdm(design_pdbs, desc=f"{self.name}:{self.pdb_to_take}"):
            obj_name = f"{self.name}_{record_id}"
            cmd.load(str(design_pdb), obj_name)
            hotspot_info = hotspots_by_ligand(obj_name, design_target_chain, ligand_chain)
            mapped_hotspots = [
                self._residue_map[residue]
                for residue in hotspot_info["hotspots"]
                if residue in self._residue_map
            ]
            hotspot_counter.update(mapped_hotspots)
            cmd.delete("complex")
            cmd.delete("target")
            cmd.delete(obj_name)
        return hotspot_counter, len(design_pdbs)

    def _extract_target_ca_coords(
        self, target_pdb: Path, residues: Iterable[ResidueKey]
    ) -> Dict[ResidueKey, np.ndarray]:
        residues = set(residues)
        cmd.delete("all")
        cmd.load(str(target_pdb), "target_ref")

        coord_map: Dict[ResidueKey, np.ndarray] = {}
        for atom in cmd.get_model("target_ref and polymer.protein and name CA").atom:
            key = (str(atom.chain), str(atom.resi))
            if key in residues:
                coord_map[key] = np.array(atom.coord, dtype=float)

        cmd.delete("target_ref")
        missing = sorted(residues.difference(coord_map), key=self._residue_sort_key)
        if missing:
            raise ValueError(f"Missing CA coordinates for residues: {missing[:10]}")
        return coord_map

    def _pairwise_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        if len(coords) <= 1:
            return np.zeros((len(coords), len(coords)), dtype=float)
        return squareform(pdist(coords))

    def _conservative_cutoff(self, coords: np.ndarray) -> float:
        '''
        Estimate a conservative continuity cutoff from hotspot CA coordinates.

        For each residue, take the distance to its 3rd nearest neighbor, then
        use the 90th percentile of those distances as a robust upper bound for
        "locally continuous" spacing. The final cutoff is clipped to 10-14 A so
        the graph stays merge-friendly by default, and only clearly separated
        regions tend to split.
        '''
        if len(coords) <= 2:
            return 10.0
        dist_mat = self._pairwise_distance_matrix(coords)
        nn_rank = min(3, len(coords) - 1)
        knn = np.partition(dist_mat + np.eye(len(coords)) * 1e9, nn_rank, axis=1)[:, nn_rank]
        return float(np.clip(np.percentile(knn, 90), 10.0, 14.0))

    def _graph_components(self, dist_mat: np.ndarray, cutoff: float) -> np.ndarray:
        if len(dist_mat) == 0:
            return np.array([], dtype=int)
        adjacency = (dist_mat <= cutoff).astype(int)
        np.fill_diagonal(adjacency, 0)
        _, labels = connected_components(csr_matrix(adjacency), directed=False)
        return labels

    def _split_if_saddle(
        self,
        indices: np.ndarray,
        dist_mat: np.ndarray,
        freqs: np.ndarray,
        cutoff: float,
    ) -> list[np.ndarray]:
        '''
        Try to split one spatially connected component by frequency topology.

        The idea is to only split when there are at least two high-frequency
        cores that are separated in the same cutoff graph, while the residues
        between them form a clearly lower-frequency bridge. If the bridge is not
        sparse enough in frequency, keep everything merged.
        '''
        if len(indices) < 8:
            return [indices]

        sub_dist = dist_mat[np.ix_(indices, indices)]
        sub_freq = freqs[indices]
        core_threshold = max(float(np.quantile(sub_freq, 0.75)), float(sub_freq.max() * 0.6))
        core_mask = sub_freq >= core_threshold
        if core_mask.sum() < 4:
            return [indices]

        core_indices = np.where(core_mask)[0]
        core_dist = sub_dist[np.ix_(core_indices, core_indices)]
        core_labels = self._graph_components(core_dist, cutoff)
        unique_core_labels = sorted(set(int(x) for x in core_labels))
        if len(unique_core_labels) <= 1:
            return [indices]

        core_groups = [core_indices[core_labels == label] for label in unique_core_labels]
        if any(len(group) < 2 for group in core_groups):
            return [indices]

        bridge_mask = ~core_mask
        if not bridge_mask.any():
            return [indices]

        bridge_freq_max = float(sub_freq[bridge_mask].max())
        core_freq_means = [float(sub_freq[group].mean()) for group in core_groups]
        if bridge_freq_max > min(core_freq_means) * 0.7:
            return [indices]

        assigned = []
        for local_idx in range(len(indices)):
            best_group = min(
                range(len(core_groups)),
                key=lambda group_id: float(sub_dist[local_idx, core_groups[group_id]].min()),
            )
            assigned.append(best_group)
        assigned = np.array(assigned, dtype=int)
        return [indices[assigned == group_id] for group_id in sorted(set(assigned))]

    def _recursive_split(
        self,
        indices: np.ndarray,
        dist_mat: np.ndarray,
        freqs: np.ndarray,
        cutoff: float,
    ) -> list[np.ndarray]:
        '''
        Recursively split hotspot residues with a conservative two-stage rule.

        First split by obvious spatial disconnection under the continuity
        cutoff. For each connected piece, then try `_split_if_saddle` to detect
        rarer cases where one connected patch still contains two dense
        high-frequency lobes linked by a low-frequency saddle.
        '''
        if len(indices) <= 1:
            return [indices]

        sub_dist = dist_mat[np.ix_(indices, indices)]
        labels = self._graph_components(sub_dist, cutoff)
        unique_labels = sorted(set(int(x) for x in labels))
        if len(unique_labels) > 1:
            groups = [indices[labels == label] for label in unique_labels]
            out = []
            for group in groups:
                out.extend(self._recursive_split(group, dist_mat, freqs, cutoff))
            return out

        saddle_groups = self._split_if_saddle(indices, dist_mat, freqs, cutoff)
        if len(saddle_groups) > 1:
            out = []
            for group in saddle_groups:
                out.extend(self._recursive_split(group, dist_mat, freqs, cutoff))
            return out

        return [indices]

    def _assign_groups(
        self,
        hotspot_counter: Counter,
        coord_map: Dict[ResidueKey, np.ndarray],
        n_designs: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        residues = sorted(hotspot_counter, key=self._residue_sort_key)
        coords = np.array([coord_map[residue] for residue in residues], dtype=float)
        freqs = np.array([hotspot_counter[residue] / n_designs for residue in residues], dtype=float)

        if len(residues) == 1:
            groups = [np.array([0], dtype=int)]
            method_info = {"method": "continuity_graph", "cutoff": np.inf, "saddle_splits": 0}
        else:
            cutoff = self._conservative_cutoff(coords)
            dist_mat = self._pairwise_distance_matrix(coords)
            groups = self._recursive_split(np.arange(len(residues)), dist_mat, freqs, cutoff)
            method_info = {
                "method": "continuity_graph",
                "cutoff": cutoff,
                "saddle_splits": max(0, len(groups) - len(set(self._graph_components(dist_mat, cutoff)))),
            }

        raw_labels = np.zeros(len(residues), dtype=int)
        for label, group in enumerate(groups, start=1):
            raw_labels[group] = label

        group_members: Dict[int, list[ResidueKey]] = {}
        for residue, label in zip(residues, raw_labels):
            group_members.setdefault(int(label), []).append(residue)

        old_to_new = {}
        ordered_groups = sorted(
            group_members,
            key=lambda group_id: (
                -sum(hotspot_counter[residue] for residue in group_members[group_id]),
                min(self._residue_sort_key(residue) for residue in group_members[group_id]),
            ),
        )
        for new_group_id, old_group_id in enumerate(ordered_groups, start=1):
            old_to_new[old_group_id] = new_group_id

        group_centers = {
            old_group_id: np.mean([coord_map[residue] for residue in members], axis=0)
            for old_group_id, members in group_members.items()
        }

        records = []
        for residue, old_group_id, freq in zip(residues, raw_labels, freqs):
            center = group_centers[int(old_group_id)]
            coord = coord_map[residue]
            records.append(
                {
                    "chain": residue[0],
                    "resi": residue[1],
                    "group_id": old_to_new[int(old_group_id)],
                    "freq": float(freq),
                    "distance_to_group_center": float(np.linalg.norm(coord - center)),
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                }
            )
        if len(records)>0:
            hotspot_df = pd.DataFrame(records).sort_values(
                by=["group_id", "freq", "chain", "resi"],
                ascending=[True, False, True, True],
            )
        else:
            hotspot_df = pd.DataFrame(columns=["chain", "resi","group_id", "freq", 
                "distance_to_group_center",'x','y','z'])
        summary_df = (
            hotspot_df.groupby("group_id", as_index=False)
            .agg(
                n_residues=("resi", "size"),
                total_freq=("freq", "sum"),
                mean_freq=("freq", "mean"),
                max_freq=("freq", "max"),
            )
            .sort_values("group_id")
        )
        method_info["n_groups"] = int(hotspot_df["group_id"].nunique())
        return hotspot_df, summary_df, method_info

    def _build_residue_selection(self, residues: Iterable[ResidueKey]) -> str:
        return " or ".join(f"(chain {chain} and resi {resi})" for chain, resi in residues)

    def _blend_with_white(self, hex_color: str, amount: float) -> Tuple[float, float, float]:
        base = np.array(mcolors.to_rgb(hex_color), dtype=float)
        white = np.ones(3, dtype=float)
        mixed = base * (1.0 - amount) + white * amount
        return tuple(float(x) for x in np.clip(mixed, 0.0, 1.0))

    def _color_groups_on_target(self, target_pdb: Path, hotspot_df: pd.DataFrame, out_pse: Path):
        cmd.delete("all")
        cmd.load(str(target_pdb), "target_hotspots")
        cmd.hide("everything", "target_hotspots")
        cmd.show("cartoon", "target_hotspots")
        cmd.color("gray80", "target_hotspots")
        cmd.bg_color("white")
        
        if len(hotspot_df)>0:
            hotspot_residues = [(str(row.chain), str(row.resi)) for row in hotspot_df.itertuples()]
            hotspot_sel = self._build_residue_selection(hotspot_residues)
            cmd.show("sticks", f"target_hotspots and ({hotspot_sel})")
            cmd.show("spheres", f"target_hotspots and ({hotspot_sel}) and name CA")
            cmd.set("sphere_scale", 0.35, "target_hotspots and name CA")
            cmd.set("stick_radius", 0.2, "target_hotspots")

            max_freq = float(hotspot_df["freq"].max())
            for group_id, group_df in hotspot_df.groupby("group_id", sort=True):
                base_color = self.group_palette[(group_id - 1) % len(self.group_palette)]
                for row in group_df.itertuples():
                    lighten = 0.65 if max_freq == 0 else 0.15 + 0.7 * (1.0 - row.freq / max_freq)
                    rgb = self._blend_with_white(base_color, lighten)
                    color_name = f"group_{group_id}_{row.chain}_{str(row.resi).replace('-', 'm')}"
                    residue_sel = f"target_hotspots and chain {row.chain} and resi {row.resi}"
                    cmd.set_color(color_name, list(rgb))
                    cmd.color(color_name, residue_sel)
        cmd.save(str(out_pse))

    def _write_method_note(
        self,
        summary_txt: Path,
        method_info: dict,
        target_chain: str,
        ligand_chain: str,
        n_designs: int,
    ):
        lines = [
            "Grouping method: conservative spatial continuity graph",
            f"Input pdb key: {self.pdb_to_take}",
            f"Original Target chain: {target_chain}",
            f"Design Target chain: A",
            f"Ligand chain: {ligand_chain}",
            f"Design count: {n_designs}",
            "Split rule 1: split only when hotspot CA graph is clearly disconnected",
            "Split rule 2: otherwise split only when high-frequency cores are separated by a low-frequency saddle",
            f"Detected groups: {method_info['n_groups']}",
            f"Continuity cutoff (A): {method_info['cutoff']:.3f}",
            f"Saddle-based extra splits: {method_info['saddle_splits']}",
        ]
        summary_txt.write_text("\n".join(lines) + "\n")

    def _rounded_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        rounded = df.copy()
        numeric_cols = rounded.select_dtypes(include=[np.number]).columns
        rounded[numeric_cols] = rounded[numeric_cols].round(2)
        return rounded

    def _residue_sort_key(self, residue: ResidueKey):
        chain, resi = residue
        if resi[:-1].isdigit() and resi[-1].isalpha():
            return chain, int(resi[:-1]), resi[-1]
        if resi.isdigit():
            return chain, int(resi), ""
        return chain, resi, ""
