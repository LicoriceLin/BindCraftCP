import shutil
import subprocess
from pathlib import Path

import pytest

from epitopecraft.backends.boltzgen.refold import BoltzGenRefold
from epitopecraft.core.artifacts import StructureArtifact
from epitopecraft.core.config import PipelineConfig
from epitopecraft.core.design import Design, DesignSet, ProteinCandidate
from epitopecraft.core.pipeline import Pipeline, PipelineRunner


def _require_boltzgen_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("No NVIDIA runtime is visible")
    gpu = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu.returncode != 0 or "GPU" not in gpu.stdout:
        pytest.skip("No GPU is allocated")
    environment = subprocess.run(
        ["conda", "run", "-n", "boltzgen", "python", "-c", "import boltzgen"],
        capture_output=True,
        text=True,
        check=False,
    )
    if environment.returncode != 0:
        pytest.skip("The boltzgen conda environment is unavailable")


@pytest.mark.gpu
def test_real_boltzgen_refold_maps_renamed_cif_chains(tmp_path: Path) -> None:
    """Run one short Boltz2 co-fold through the public Pipeline Step."""

    _require_boltzgen_gpu()
    target_path = Path("epitopecraft/test/targets/WDR5-seg_6.pdb").resolve()
    target = StructureArtifact.from_file("wdr5_segment", target_path)
    designs = DesignSet.from_designs(
        [Design("smoke", ProteinCandidate(sequence="ACDEFGHIKL"))]
    )
    config = PipelineConfig.from_mapping(
        {
            "steps": {
                "boltz_refold": {
                    "params": {
                        "target_chains": ["B"],
                        "binder_chain": "Z",
                        "recycling_steps": 1,
                        "sampling_steps": 10,
                        "diffusion_samples": 1,
                        "num_workers": 1,
                    }
                }
            }
        }
    )
    pipeline = Pipeline(
        "boltzgen_refold_smoke",
        inputs={"target": StructureArtifact, "designs": DesignSet},
        config=config,
    )
    folded = pipeline.add(
        BoltzGenRefold("boltz_refold"),
        target=pipeline.inputs.target,
        designs=pipeline.inputs.designs,
    )
    pipeline.output("designs", folded.designs)

    result = PipelineRunner(pipeline, run_dir=tmp_path).run(
        target=target,
        designs=designs,
    )

    design = result.designs["smoke"]
    artifact = design.artifacts["boltz_refold.structure"]
    assert design.metrics["boltz_refold"]["folding"]["iptm"] >= 0.0
    assert artifact.residue_map.summary() == {"exact": 137}
    assert artifact.reference.infer_chain_entities(artifact) == {
        "A": "target:B",
        "B": "binder",
    }
