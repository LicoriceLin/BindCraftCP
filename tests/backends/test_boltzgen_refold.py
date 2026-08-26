from pathlib import Path

from epitopecraft.backends.boltzgen.refold import (
    BoltzGenRefold,
    BoltzGenRefoldConfig,
    build_refold_command,
)


def test_refold_command_keeps_msa_binding_group_and_analysis_controls(tmp_path):
    config = BoltzGenRefoldConfig(
        target_chains=("A", "B"),
        binding_res_index=("A:1,3..5", "B:2"),
        target_msa=("A:/data/a.a3m", "B:/data/b.a3m"),
        target_msa_res_index=("A:10..50",),
        max_msa_seqs=2048,
        structure_groups="none",
        structure_group=("A:1", "B:1:2..7"),
        binder_chain="Z",
        cyclic=True,
        run_analysis=True,
        debug=True,
        no_delta_sasa_refolded=True,
        no_noncovalents_refolded=True,
    )

    command = build_refold_command(
        config,
        target_structure=tmp_path / "target.cif",
        binder_csv=tmp_path / "binders.csv",
        output_dir=tmp_path / "results",
    )
    rendered = " ".join(command)

    assert Path(command[-1]).name != "peptide_refold_cli.py"
    assert "epitopecraft.backends.boltzgen.peptide_refold_cli" in command
    assert rendered.count("--target-chain") == 2
    assert rendered.count("--target-msa") >= 2
    assert "--binding-res-index A:1,3..5" in rendered
    assert "--structure-group B:1:2..7" in rendered
    assert "--run-analysis" in command
    assert "--cyclic" in command
    assert "--no-delta-sasa-refolded" in command


def test_boltzgen_refold_declares_instance_scoped_config_and_ports():
    step = BoltzGenRefold("boltz_after_mpnn")

    assert step.config_type is BoltzGenRefoldConfig
    assert set(step.input_ports) == {"target", "designs"}
    assert set(step.output_ports) == {"designs"}
