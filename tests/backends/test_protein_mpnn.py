from epitopecraft.backends.colabdesign.mpnn import ProteinMPNN
from epitopecraft.core.artifacts import ReferenceModel, StructureArtifact
from epitopecraft.core.config import PipelineConfig
from epitopecraft.core.design import Design, DesignSet, ProteinCandidate
from epitopecraft.core.pipeline import Pipeline, PipelineRunner


def pdb_text():
    return (
        "ATOM      1  CA  ALA X  10       0.000   0.000   0.000  1.00 80.00           C  \n"
        "ATOM      2  CA  CYS B   1       1.000   0.000   0.000  1.00 80.00           C  \n"
        "END\n"
    )


class FakeMPNNModel:
    def __init__(self):
        import numpy as np

        self._inputs = {"bias": np.zeros((2, 20), dtype=float)}
        self.prepared = None

    def prep_inputs(self, path, chains, fix_pos):
        self.prepared = (path, chains, fix_pos)

    def score(self, temperature):
        return {"score": 0.25}

    def sample(self, temperature, num, batch):
        return {
            "seq": ["XS", "XS", "XA"],
            "score": [0.9, 0.8, 0.7],
            "seqid": [0.0, 0.0, 0.0],
        }


def test_remove_cysteine_is_an_instance_scoped_mpnn_recipe(tmp_path):
    raw = StructureArtifact.from_text("complex", pdb_text())
    reference = ReferenceModel.from_target(full_structure=raw, target_chains=("X",))
    reference.add_protein_binder("C", preferred_chain="B")
    mapped = reference.map_structure(raw, {"X": "target:X", "B": "binder"})
    designs = DesignSet.from_designs(
        [
            Design(
                "complex",
                ProteinCandidate("C"),
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
                        "n_samples": 3,
                        "max_sequences": 1,
                        "include_input": False,
                    }
                }
            }
        }
    )
    step = ProteinMPNN("remove_c")
    step._model = FakeMPNNModel()
    pipeline = Pipeline("remove_c", inputs={"designs": DesignSet}, config=config)
    redesigned = pipeline.add(step, designs=pipeline.inputs.designs)
    pipeline.output("designs", redesigned.designs)

    result = PipelineRunner(pipeline, run_dir=tmp_path).run(designs=designs)

    assert result.designs.ids == ("complex-remove_c1",)
    child = result.designs["complex-remove_c1"]
    assert child.candidate.sequence == "S"
    assert child.parent_id == "complex"
    assert step._model.prepared[1:] == ("X,B", "X")
