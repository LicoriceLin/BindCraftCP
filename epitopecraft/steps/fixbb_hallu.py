from .basestep import *
import math
import tempfile
from pathlib import Path
from typing import Tuple

from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.Polypeptide import is_aa
from colabdesign import mk_afdesign_model


DEFAULT_FIXBB_HALLU_TEST_INPUT = (
    "input/cholera/smoke/run_40_70_20/boltzdesign/final_ranked_designs/"
    "final_20_designs/before_refolding/"
    "rank05_cholera_s150_smoke_40_70_20-boltz_12.cif"
)


def _looks_like_pdb_text(text: str) -> bool:
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "ENDMDL")):
            return True
    return False


def _load_structure(structure_file: str):
    path = Path(structure_file)
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def _count_chain_residues(structure_file: str, chain_id: str) -> int:
    structure = _load_structure(structure_file)
    model = next(structure.get_models())
    chain = model[chain_id]
    return sum(1 for residue in chain.get_residues() if is_aa(residue, standard=False))


def _get_chain_lengths(structure_file: str, chain_spec: str) -> list[int]:
    return [_count_chain_residues(structure_file, chain_id) for chain_id in chain_spec.split(",")]


class FixbbHallu(BaseStep):
    def __init__(
        self,
        settings: GlobalSettings,
        af_model: mk_afdesign_model | None = None,
    ):
        super().__init__(settings)
        self.init_afdesign_model(af_model)

    @property
    def name(self) -> str:
        return "fixbb_hallu"

    @property
    def metrics_to_add(self) -> Tuple[str, ...]:
        prefix = self.metrics_prefix
        ret = [
            f"{prefix}pLDDT",
            f"{prefix}pTM",
            f"{prefix}pAE",
            f"{prefix}i-pTM",
            f"{prefix}i-pAE",
            f"{prefix}rmsd",
            f"{prefix}seqid",
            f"{prefix}dgram_cce",
            f"{prefix}fape",
        ]
        if self.settings.adv.setdefault(f"{self.name}-save-loss", False):
            ret.append(f"{prefix}loss")
        return tuple(ret)

    @property
    def pdb_to_add(self) -> Tuple[str, ...]:
        return (self.metrics_prefix.strip(NEST_SEP),)

    @property
    def params_to_take(self) -> Tuple[str, ...]:
        return (
            "af_params_dir",
            "num_recycles_design",
            "sample_models",
            "omit_AAs",
            "soft_iterations",
            "temporary_iterations",
            "hard_iterations",
            "greedy_iterations",
            "greedy_percentage",
            "4stage-use-pssm",
            "4stage_keep_best",
            "4stage-mcmc",
            "mcmc_half_life_ratio",
            "t_init_mcmc",
            "verb",
            f"{self.name}-prefix",
            f"{self.name}-pdb-input",
            f"{self.name}-chain",
            f"{self.name}-design-chain",
            f"{self.name}-fix-pos",
            f"{self.name}-ignore-missing",
            f"{self.name}-best-metric",
            f"{self.name}-models",
            f"{self.name}-num-models",
            f"{self.name}-save-loss",
            f"{self.name}-test-input",
        )

    @property
    def _default_pdb_input_key(self) -> str:
        return "input"

    def init_afdesign_model(
        self,
        af_model: mk_afdesign_model | None = None,
    ) -> mk_afdesign_model:
        advanced_settings = self.settings.adv
        if af_model is None:
            af_model = mk_afdesign_model(
                protocol="fixbb",
                debug=False,
                data_dir=advanced_settings.setdefault("af_params_dir", ""),
                use_multimer=False,
                num_recycles=advanced_settings.setdefault("num_recycles_design", 1),
                best_metric=advanced_settings.setdefault(
                    f"{self.name}-best-metric", "loss"
                ),
            )
        else:
            af_model.restart(seed=self.settings.binder_settings.global_seed)
        self.af_model = af_model
        return af_model

    @property
    def design_models(self) -> list[int]:
        models = self.settings.adv.setdefault(f"{self.name}-models", [0])
        if isinstance(models, int):
            return [models]
        return [int(model_idx) for model_idx in models]

    def _resolve_input_source(self, record: DesignRecord) -> tuple[str, str | None]:
        source_file = None
        cleanup_file = None
        if record.has_pdb(self.pdb_to_take):
            if self.pdb_to_take in record.pdb_files:
                source_file = record.pdb_files[self.pdb_to_take]
            else:
                pdb_text = record.pdb_strs[self.pdb_to_take]
                suffix = ".pdb" if _looks_like_pdb_text(pdb_text) else ".cif"
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=suffix, delete=False
                ) as handle:
                    handle.write(pdb_text)
                    source_file = handle.name
                    cleanup_file = handle.name
        else:
            source_file = self.settings.adv.setdefault(
                f"{self.name}-test-input", DEFAULT_FIXBB_HALLU_TEST_INPUT
            )
        if source_file is None:
            raise ValueError(
                f"{self.name} could not resolve an input structure from record key "
                f"`{self.pdb_to_take}` or advanced setting `{self.name}-test-input`"
            )
        return source_file, cleanup_file

    def _materialize_pdb_input(self, structure_file: str) -> tuple[str, str | None]:
        suffix = Path(structure_file).suffix.lower()
        if suffix not in {".cif", ".mmcif"}:
            return structure_file, None

        structure = _load_structure(structure_file)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as handle:
            pdb_file = handle.name
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file)
        return pdb_file, pdb_file

    def config_afdesign_model(self, pdb_filename: str, record: DesignRecord):
        advanced_settings = self.settings.adv
        rm_aa = advanced_settings.setdefault("omit_AAs", "")
        if not rm_aa:
            rm_aa = None
        chain_spec = advanced_settings.setdefault(f"{self.name}-chain", "C")
        design_chain = advanced_settings.setdefault(f"{self.name}-design-chain", None)
        if design_chain is None:
            if "," in chain_spec:
                design_chain = chain_spec.split(",")[-1]
            else:
                design_chain = chain_spec
        fix_pos = advanced_settings.setdefault(f"{self.name}-fix-pos", None)
        self._current_chain_spec = chain_spec
        self._current_design_chain = design_chain
        self._current_chain_lengths = _get_chain_lengths(pdb_filename, chain_spec)
        if fix_pos in (None, "") and design_chain is not None and "," in chain_spec:
            fixed_chains = [
                chain_id for chain_id in chain_spec.split(",") if chain_id != design_chain
            ]
            fix_pos = ",".join(fixed_chains)
        seed = record.get_metrics("config:seed", self.settings.binder_settings.global_seed)
        self.af_model.prep_inputs(
            pdb_filename=pdb_filename,
            chain=chain_spec,
            fix_pos=fix_pos,
            ignore_missing=advanced_settings.setdefault(
                f"{self.name}-ignore-missing", True
            ),
            rm_aa=rm_aa,
            seed=seed,
        )

    def _interface_metrics(self, best_aux: dict) -> tuple[float | None, float | None]:
        if len(getattr(self.af_model, "_lengths", [])) <= 1:
            return None, None
        chain_ids = self._current_chain_spec.split(",")
        if self._current_design_chain not in chain_ids:
            return None, None
        chain_idx = chain_ids.index(self._current_design_chain)
        chain_lengths = self._current_chain_lengths
        start = sum(chain_lengths[:chain_idx])
        end = start + chain_lengths[chain_idx]
        pae = np.asarray(best_aux["pae"])
        design_to_target = pae[start:end, :start]
        if end < pae.shape[0]:
            design_to_target = np.concatenate([design_to_target, pae[start:end, end:]], axis=1)
        target_to_design = pae[:start, start:end]
        if end < pae.shape[0]:
            target_to_design = np.concatenate([target_to_design, pae[end:, start:end]], axis=0)
        interface_blocks = []
        if design_to_target.size:
            interface_blocks.append(design_to_target.reshape(-1))
        if target_to_design.size:
            interface_blocks.append(target_to_design.reshape(-1))
        if not interface_blocks:
            return float(best_aux.get("i_ptm")) if best_aux.get("i_ptm") is not None else None, None
        i_pae = float(np.concatenate(interface_blocks).mean() / 31.0)
        i_ptm = best_aux.get("i_ptm")
        i_ptm = float(i_ptm) if i_ptm is not None else None
        return i_ptm, i_pae

    def _design_chain_sequence(self, full_sequence: str) -> str:
        chain_ids = self._current_chain_spec.split(",")
        if self._current_design_chain not in chain_ids:
            return full_sequence
        chain_idx = chain_ids.index(self._current_design_chain)
        start = sum(self._current_chain_lengths[:chain_idx])
        end = start + self._current_chain_lengths[chain_idx]
        return full_sequence[start:end]

    def sample_trajectory(self, record: DesignRecord) -> DesignRecord:
        prefix = self.metrics_prefix
        af_model = self.af_model
        advanced_settings = self.settings.adv
        design_models = self.design_models
        num_models = min(
            len(design_models),
            advanced_settings.setdefault(f"{self.name}-num-models", 1),
        )
        sample_models = advanced_settings.setdefault("sample_models", False)
        verb = advanced_settings.setdefault("verb", 1)

        af_model.design_logits(
            iters=advanced_settings.setdefault("soft_iterations", 75),
            e_soft=1,
            models=design_models,
            num_models=num_models,
            sample_models=sample_models,
            ramp_recycles=False,
            save_best=True,
            verbose=verb,
        )
        af_model.design_soft(
            advanced_settings.setdefault("temporary_iterations", 45),
            e_temp=1e-2,
            models=design_models,
            num_models=num_models,
            sample_models=sample_models,
            ramp_recycles=False,
            save_best=True,
            verbose=verb,
        )
        af_model.design_hard(
            advanced_settings.setdefault("hard_iterations", 5),
            temp=1e-2,
            models=design_models,
            num_models=num_models,
            sample_models=sample_models,
            dropout=False,
            ramp_recycles=False,
            save_best=True,
            verbose=verb,
        )

        if advanced_settings.setdefault("greedy_iterations", 15) > 0:
            greedy_tries = math.ceil(
                af_model._len * (advanced_settings.setdefault("greedy_percentage", 1) / 100)
            )
            af_model.clear_best()
            if advanced_settings.setdefault("4stage-use-pssm", False):
                seq_logits = None
            else:
                seq_logits = af_model.aux["seq"]["pssm"][0]
            if not advanced_settings.setdefault("4stage-mcmc", False):
                af_model.design_pssm_semigreedy(
                    soft_iters=0,
                    hard_iters=advanced_settings["greedy_iterations"],
                    tries=max(1, greedy_tries),
                    models=design_models,
                    num_models=num_models,
                    sample_models=sample_models,
                    ramp_models=False,
                    save_best=True,
                    seq_logits=seq_logits,
                    verbose=verb,
                )
            else:
                af_model._design_mcmc(
                    steps=advanced_settings["greedy_iterations"],
                    half_life=int(
                        advanced_settings["greedy_iterations"]
                        * advanced_settings.setdefault("mcmc_half_life_ratio", 0.2)
                    ),
                    seq_logits=seq_logits,
                    T_init=advanced_settings.setdefault("t_init_mcmc", 0.01),
                    mutation_rate=max(1, greedy_tries),
                    num_models=num_models,
                    models=design_models,
                    sample_models=sample_models,
                    save_best=True,
                    verbose=verb,
                )
            if advanced_settings.setdefault("4stage_keep_best", False):
                best = af_model._tmp["best"]
                af_model.aux = best["aux"]
                af_model.set_seq(
                    seq=np.array(best["aux"]["seq"]["input"]),
                    bias=af_model._inputs["bias"],
                )
                af_model._save_results(save_best=True, verbose=False)

        best_aux = af_model._tmp["best"].get("aux", af_model.aux)
        i_ptm, i_pae = self._interface_metrics(best_aux)
        metrics = {
            f"{prefix}pLDDT": best_aux["log"]["plddt"],
            f"{prefix}pTM": best_aux["log"]["ptm"],
            f"{prefix}pAE": best_aux["log"]["pae"],
            f"{prefix}i-pTM": i_ptm,
            f"{prefix}i-pAE": i_pae,
            f"{prefix}rmsd": best_aux["log"]["rmsd"],
            f"{prefix}seqid": best_aux["log"]["seqid"],
            f"{prefix}dgram_cce": best_aux["log"]["dgram_cce"],
            f"{prefix}fape": best_aux["log"]["fape"],
        }
        if advanced_settings.setdefault(f"{self.name}-save-loss", False):
            metrics[f"{prefix}loss"] = best_aux["log"]["loss"]

        record.sequence = self._design_chain_sequence(af_model.get_seq()[0])
        record.pdb_strs[self.pdb_to_add[0]] = af_model.save_pdb()
        record.update_metrics(metrics)
        return record

    def process_record(self, input: DesignRecord) -> DesignRecord:
        source_file, source_cleanup = self._resolve_input_source(input)
        pdb_input, pdb_cleanup = self._materialize_pdb_input(source_file)
        try:
            with self.record_time(input):
                self.config_afdesign_model(pdb_input, input)
                self.sample_trajectory(input)
        finally:
            for cleanup_file in (pdb_cleanup, source_cleanup):
                if cleanup_file is not None and Path(cleanup_file).exists():
                    Path(cleanup_file).unlink()
        return input


def build_fixbb_hallu_smoke_settings(
    test_input: str = DEFAULT_FIXBB_HALLU_TEST_INPUT,
    design_path: str = "output/fixbb_hallu_smoke",
    chain: str = "C",
    extra_patch: dict | None = None,
) -> GlobalSettings:
    chain_len = _count_chain_residues(test_input, chain)
    patch = {
        "af_params_dir": "params/",
        "sample_models": False,
        "num_recycles_design": 1,
        "soft_iterations": 1,
        "temporary_iterations": 1,
        "hard_iterations": 1,
        "greedy_iterations": 0,
        "verb": 1,
        "omit_AAs": "",
        "fixbb_hallu-chain": chain,
        "fixbb_hallu-design-chain": chain,
        "fixbb_hallu-test-input": str(test_input),
    }
    if extra_patch:
        patch.update(extra_patch)
    return GlobalSettings(
        target_settings=TargetSettings(
            starting_pdb=str(test_input),
            chains=chain,
            full_target_pdb=str(test_input),
            full_target_chain=chain,
        ),
        binder_settings=BinderSettings(
            design_path=design_path,
            binder_name=f"fixbb_{Path(test_input).stem}",
            binder_lengths=[chain_len],
            random_seeds=[42],
            helix_values=[0.0],
        ),
        advanced_settings=AdvancedSettings(
            advanced_paths=["epitopecraft/pipelines/config/base_advanced_settings.yaml"],
            extra_patch=patch,
        ),
        filter_settings=FilterSettings(filters_path="none"),
    )


def smoke_test_fixbb_hallu(
    test_input: str = DEFAULT_FIXBB_HALLU_TEST_INPUT,
    design_path: str = "output/fixbb_hallu_smoke",
    chain: str = "C",
    pdb_purge_stem: str = "fixbb_hallu",
    extra_patch: dict | None = None,
) -> DesignRecord:
    settings = build_fixbb_hallu_smoke_settings(
        test_input=test_input,
        design_path=design_path,
        chain=chain,
        extra_patch=extra_patch,
    )
    Path(settings.binder_settings.design_path).mkdir(parents=True, exist_ok=True)
    record = DesignRecord(
        id=Path(test_input).stem,
        sequence="",
        pdb_files={"input": str(test_input)},
    )
    batch = DesignBatch(Path(settings.binder_settings.design_path) / "metrics")
    batch.add_record(record)
    step = FixbbHallu(settings)
    step.process_batch(batch, pdb_purge_stem=pdb_purge_stem)
    return batch[record.id]


if __name__ == "__main__":
    smoke_test_fixbb_hallu()
