from epitopecraft.backends.boltz.screen import (
    Boltz2Screen,
    Boltz2ScreenConfig,
    build_boltz2_command,
    build_boltz2_input,
)
from epitopecraft.core.artifacts import (
    CanonicalResidueId,
    ReferenceModel,
    SelectionArtifact,
    StructureArtifact,
)
from epitopecraft.core.config import PipelineConfig
from epitopecraft.core.design import Design, ProteinCandidate, SmallMoleculeCandidate


def target_pdb():
    return (
        "ATOM      1  CA  ALA X  10       0.000   0.000   0.000  1.00 80.00           C  \n"
        "ATOM      2  CA  GLY X  11       2.000   0.000   0.000  1.00 80.00           C  \n"
        "END\n"
    )


def test_boltz2_input_supports_protein_and_small_molecule_screening(tmp_path):
    target = StructureArtifact.from_text("target", target_pdb())
    reference = ReferenceModel.from_target(full_structure=target, target_chains=("X",))
    site = SelectionArtifact(
        "known_site",
        (CanonicalResidueId("target:X", 2),),
    )
    config = Boltz2ScreenConfig(
        target_msa="empty",
        include_affinity=True,
        use_target_template=True,
    )

    ligand = build_boltz2_input(
        reference,
        site,
        SmallMoleculeCandidate("CCO"),
        config,
        target_path=tmp_path / "target.pdb",
    )
    protein = build_boltz2_input(
        reference,
        site,
        ProteinCandidate("ACDE"),
        config,
        target_path=tmp_path / "target.pdb",
    )

    assert ligand["sequences"][1] == {"ligand": {"id": "B", "smiles": "CCO"}}
    assert ligand["constraints"][0]["pocket"]["contacts"] == [["X", 2]]
    assert ligand["properties"] == [{"affinity": {"binder": "B"}}]
    assert protein["sequences"][1]["protein"]["sequence"] == "ACDE"
    assert "properties" not in protein
    assert ligand["templates"][0]["template_id"] == "X"


def test_boltz2_command_exposes_structure_and_affinity_sampling(tmp_path):
    config = Boltz2ScreenConfig(
        recycling_steps=1,
        sampling_steps=10,
        diffusion_samples=2,
        sampling_steps_affinity=20,
        diffusion_samples_affinity=3,
        override=True,
    )

    command = build_boltz2_command(config, tmp_path / "inputs", tmp_path / "run")
    rendered = " ".join(command)

    assert "boltz predict" in rendered
    assert "--sampling_steps 10" in rendered
    assert "--diffusion_samples 2" in rendered
    assert "--sampling_steps_affinity 20" in rendered
    assert "--override" in command


def test_mixed_candidates_run_in_separate_affinity_groups(tmp_path, monkeypatch):
    """Protein cases must not enter a Boltz affinity batch."""

    target = StructureArtifact.from_text("target", target_pdb())
    site = SelectionArtifact(
        "known_site",
        (CanonicalResidueId("target:X", 1),),
    )
    candidates = (ProteinCandidate("ACDE"), SmallMoleculeCandidate("CCO"))
    step = Boltz2Screen("screen")
    step.resolve_config(PipelineConfig())

    commands = []
    monkeypatch.setattr(
        "epitopecraft.backends.boltz.screen.subprocess.run",
        lambda command, check: commands.append(command),
    )
    monkeypatch.setattr(
        step,
        "_read_case",
        lambda case_id, candidate, payload, predictions, reference: Design(
            case_id,
            candidate,
            metrics={"prediction_root": str(predictions)},
        ),
    )

    class Context:
        @staticmethod
        def step_dir(step_id):
            return tmp_path / step_id

    result = step.execute(
        {"target": target, "site": site, "candidates": candidates},
        Context(),
    )

    assert len(commands) == 2
    input_dirs = {command[command.index("predict") + 1] for command in commands}
    assert input_dirs == {
        str(tmp_path / "screen" / "inputs" / "structure"),
        str(tmp_path / "screen" / "inputs" / "affinity"),
    }
    protein = result["designs"]["candidate_00000"]
    ligand = result["designs"]["candidate_00001"]
    assert "/structure/" in protein.metrics["prediction_root"]
    assert "/affinity/" in ligand.metrics["prediction_root"]
