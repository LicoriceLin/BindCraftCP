import shutil
import subprocess
from pathlib import Path

import pytest

from epitopecraft.backends.boltz.screen import Boltz2Screen
from epitopecraft.core.artifacts import (
    CanonicalResidueId,
    ReferenceModel,
    SelectionArtifact,
    StructureArtifact,
)
from epitopecraft.core.config import PipelineConfig
from epitopecraft.core.design import SmallMoleculeCandidate
from epitopecraft.core.pipeline import Pipeline, PipelineRunner


def _require_boltz2_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("No NVIDIA runtime is visible")
    gpu = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
    )
    if gpu.returncode != 0 or "GPU" not in gpu.stdout:
        pytest.skip("No GPU is allocated")
    environment = subprocess.run(
        ["conda", "run", "-n", "binding_affinity", "boltz", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if environment.returncode != 0:
        pytest.skip("The binding_affinity Boltz environment is unavailable")


@pytest.mark.gpu
def test_real_boltz2_known_site_small_molecule_screen(tmp_path: Path) -> None:
    _require_boltz2_gpu()
    target_path = Path("epitopecraft/test/targets/WDR5-seg_6.pdb").resolve()
    target = StructureArtifact.from_file("wdr5_segment", target_path)
    reference = ReferenceModel.from_target(full_structure=target, target_chains=("B",))
    site = SelectionArtifact(
        "known_site",
        (
            CanonicalResidueId("target:B", 1),
            CanonicalResidueId("target:B", 2),
        ),
    )
    candidates = (SmallMoleculeCandidate("CCO"),)
    config = PipelineConfig.from_mapping(
        {
            "steps": {
                "screen": {
                    "params": {
                        "recycling_steps": 1,
                        "sampling_steps": 10,
                        "diffusion_samples": 1,
                        "sampling_steps_affinity": 10,
                        "diffusion_samples_affinity": 1,
                        "num_workers": 1,
                        "no_kernels": True,
                        "use_target_template": False,
                    }
                }
            }
        }
    )
    pipeline = Pipeline(
        "screen_gpu",
        inputs={
            "target": StructureArtifact,
            "site": SelectionArtifact,
            "candidates": tuple,
        },
        config=config,
    )
    screened = pipeline.add(
        Boltz2Screen("screen"),
        target=pipeline.inputs.target,
        site=pipeline.inputs.site,
        candidates=pipeline.inputs.candidates,
    )
    pipeline.output("designs", screened.designs)

    result = PipelineRunner(pipeline, run_dir=tmp_path).run(
        target=target,
        site=site,
        candidates=candidates,
    )

    design = result.designs["candidate_00000"]
    assert "affinity" in design.metrics["screen"]
    mapping = design.artifacts["screen.structure"].residue_map.summary()
    assert mapping["exact"] == 127
    # ResidueMap tracks polymers. The ligand remains a candidate/entity rather
    # than being represented as a fake, unmapped protein residue.
    assert mapping.get("unmapped_chain", 0) == 0
