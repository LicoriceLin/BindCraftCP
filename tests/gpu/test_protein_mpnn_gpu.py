import importlib.util
import shutil
import subprocess

import pytest

from epitopecraft.backends.colabdesign.mpnn import ProteinMPNN
from epitopecraft.core.artifacts import ReferenceModel, StructureArtifact
from epitopecraft.core.config import PipelineConfig
from epitopecraft.core.design import Design, DesignSet, ProteinCandidate
from epitopecraft.core.pipeline import Pipeline, PipelineRunner


def _require_gpu():
    if shutil.which("nvidia-smi") is None:
        pytest.skip("No NVIDIA runtime is visible")
    result = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or "GPU" not in result.stdout:
        pytest.skip("No GPU is allocated")
    if importlib.util.find_spec("colabdesign") is None:
        pytest.skip("ColabDesign is unavailable in the active Python environment")


def _backbone_pdb():
    residues = {
        "A": [(1, "ALA"), (2, "GLY"), (3, "SER")],
        "B": [(1, "ALA"), (2, "CYS"), (3, "CYS"), (4, "GLY")],
    }
    lines = []
    serial = 1
    x = 0.0
    for chain, chain_residues in residues.items():
        for residue_id, residue_name in chain_residues:
            for atom_name, element, offset in (
                ("N", "N", 0.0),
                ("CA", "C", 1.2),
                ("C", "C", 2.4),
                ("O", "O", 3.2),
            ):
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} {chain}{residue_id:4d}    "
                    f"{x + offset:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00          {element:>2s}  "
                )
                serial += 1
            x += 3.8
        lines.append("TER")
        x += 8.0
    return "\n".join(lines) + "\nEND\n"


@pytest.mark.gpu
def test_real_protein_mpnn_remove_cysteine_recipe(tmp_path):
    _require_gpu()
    raw = StructureArtifact.from_text("complex", _backbone_pdb())
    reference = ReferenceModel.from_target(full_structure=raw, target_chains=("A",))
    reference.add_protein_binder("ACCG", preferred_chain="B")
    mapped = reference.map_structure(raw, {"A": "target:A", "B": "binder"})
    designs = DesignSet.from_designs(
        [
            Design(
                "complex",
                ProteinCandidate("ACCG"),
                artifacts={"complex": mapped},
            )
        ]
    )
    config = PipelineConfig.from_mapping(
        {
            "steps": {
                "remove_c": {
                    "params": {
                        "structure_key": "complex",
                        "recipe_path": "epitopecraft/pipelines/config/mpnn-rmC-recipe.json",
                        "n_samples": 4,
                        "max_sequences": 2,
                        "include_input": False,
                    }
                }
            }
        }
    )
    pipeline = Pipeline("remove_c_gpu", inputs={"designs": DesignSet}, config=config)
    redesigned = pipeline.add(ProteinMPNN("remove_c"), designs=pipeline.inputs.designs)
    pipeline.output("designs", redesigned.designs)

    result = PipelineRunner(pipeline, run_dir=tmp_path).run(designs=designs)

    assert result.designs.ids
    assert all("C" not in design.candidate.sequence for design in result.designs)
