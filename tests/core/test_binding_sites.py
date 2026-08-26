from epitopecraft.analysis.binding_sites import BindingSiteDiscovery
from epitopecraft.core.artifacts import ReferenceModel, StructureArtifact
from epitopecraft.core.design import Design, DesignSet, ProteinCandidate
from epitopecraft.core.pipeline import Pipeline, PipelineRunner


def pdb_with_ca(chains):
    lines = []
    serial = 1
    for chain, residues in chains.items():
        for residue_id, residue_name, x in residues:
            lines.append(
                f"ATOM  {serial:5d}  CA  {residue_name:>3s} {chain}{residue_id:4d}    "
                f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00           C  "
            )
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def test_binding_site_discovery_returns_canonical_patches_and_table(tmp_path):
    target = StructureArtifact.from_text(
        "target",
        pdb_with_ca(
            {"X": [(10, "ALA", 0.0), (11, "GLY", 10.0), (12, "SER", 20.0)]}
        ),
    )
    reference = ReferenceModel.from_target(full_structure=target, target_chains=("X",))
    reference.add_protein_binder("C", preferred_chain="B")

    structures = []
    for artifact_id, binder_x in (("d0", 0.5), ("d1", 10.5)):
        raw = StructureArtifact.from_text(
            artifact_id,
            pdb_with_ca(
                {
                    "A": [(1, "ALA", 0.0), (2, "GLY", 10.0), (3, "SER", 20.0)],
                    "B": [(1, "CYS", binder_x)],
                }
            ),
        )
        structures.append(reference.map_structure_auto(raw))

    designs = DesignSet.from_designs(
        [
            Design(
                f"d{index}",
                ProteinCandidate("C"),
                artifacts={"complex": structure},
            )
            for index, structure in enumerate(structures)
        ]
    )
    pipeline = Pipeline(
        "discover",
        inputs={"target": StructureArtifact, "designs": DesignSet},
    )
    sites = pipeline.add(
        BindingSiteDiscovery(
            "sites",
            params={
                "structure_key": "complex",
                "frequency_threshold": 0.5,
                "contact_distance": 2.0,
                "cluster_distance": 5.0,
            },
        ),
        target=pipeline.inputs.target,
        designs=pipeline.inputs.designs,
    )
    pipeline.output("patches", sites.patches)
    pipeline.output("table", sites.table)

    result = PipelineRunner(pipeline, run_dir=tmp_path).run(
        target=target,
        designs=designs,
    )

    assert len(result.patches) == 2
    assert {patch.residues[0].position for patch in result.patches} == {1, 2}
    assert result.table.path.exists()
    assert {row["author_residue"] for row in result.table.rows} == {"X:10", "X:11"}
