import pickle
from pathlib import Path
from typing import Mapping, Sequence

import gemmi

from epitopecraft.core.artifacts import (
    LocalResidueId,
    ReferenceModel,
    StructureArtifact,
)


AA3 = {"A": "ALA", "C": "CYS", "G": "GLY", "S": "SER"}


def pdb_text(
    chains: Mapping[str, Sequence[tuple[int, str, str]]],
) -> str:
    lines = []
    serial = 1
    for chain_id, residues in chains.items():
        for residue_id, insertion_code, aa in residues:
            lines.append(
                f"ATOM  {serial:5d}  CA  {AA3[aa]:>3s} {chain_id:1s}"
                f"{residue_id:4d}{insertion_code:1s}   "
                f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00           C  "
            )
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def test_full_target_is_reference_and_local_epitope_maps_to_author_ids(
    tmp_path: Path,
) -> None:
    full = StructureArtifact.from_text(
        "full",
        pdb_text({"X": [(10, "", "A"), (11, "A", "G"), (12, "", "S")]}),
    )
    local = StructureArtifact.from_text(
        "local",
        pdb_text({"A": [(1, "", "G"), (2, "", "S")]}),
    )
    reference = ReferenceModel.from_target(
        full_structure=full,
        local_structure=local,
        target_chains=("X",),
    )
    mapped = reference.map_structure(local, {"A": "target:X"})

    assert mapped.residue_map.to_canonical(LocalResidueId("A", 1)).position == 2
    author = reference.author_residue(mapped.residue_map.to_canonical(LocalResidueId("A", 1)))
    assert (author.chain_id, author.sequence_id, author.insertion_code) == ("X", 11, "A")

    bundle = mapped.write_debug_bundle(tmp_path / "debug")
    assert bundle.residue_map.exists()
    assert bundle.entities.exists()
    assert bundle.validation.exists()


def test_binder_gets_a_non_conflicting_canonical_chain() -> None:
    target = StructureArtifact.from_text(
        "target",
        pdb_text({"A": [(1, "", "A")], "B": [(1, "", "G")]}),
    )
    reference = ReferenceModel.from_target(full_structure=target, target_chains=("A", "B"))
    binder = reference.add_protein_binder("CG")

    assert binder.chain_id == "C"


def test_hallucination_and_graft_share_one_reference_mapping(tmp_path: Path) -> None:
    full = StructureArtifact.from_text(
        "full",
        pdb_text({"X": [(10, "", "A"), (11, "A", "G"), (12, "", "S")]}),
    )
    reference = ReferenceModel.from_target(full_structure=full, target_chains=("X",))
    reference.add_protein_binder("CG", preferred_chain="B")

    hallucination = StructureArtifact.from_text(
        "hallucination",
        pdb_text({"A": [(1, "", "G"), (2, "", "S")], "B": [(1, "", "C"), (2, "", "G")]}),
    )
    graft = StructureArtifact.from_text(
        "graft",
        pdb_text(
            {
                "X": [(10, "", "A"), (11, "A", "G"), (12, "", "S")],
                "B": [(1, "", "C"), (2, "", "G")],
            }
        ),
    )
    mapped_hallu = reference.map_structure(
        hallucination,
        {"A": "target:X", "B": "binder"},
    )
    mapped_graft = reference.map_structure(graft, {"X": "target:X", "B": "binder"})

    hallu_target = mapped_hallu.residue_map.to_canonical(LocalResidueId("A", 1))
    graft_target = mapped_graft.residue_map.to_canonical(LocalResidueId("X", 11, "A"))
    assert hallu_target == graft_target

    canonical_path = mapped_hallu.export(tmp_path / "canonical.pdb", numbering="reference")
    canonical_text = Path(canonical_path).read_text()
    assert " X  11A" in canonical_text

    contiguous_path = mapped_hallu.export(
        tmp_path / "contiguous.pdb", numbering="canonical"
    )
    assert " X   2 " in contiguous_path.read_text()


def test_backend_chain_renaming_is_inferred_from_entity_sequences() -> None:
    target = StructureArtifact.from_text(
        "target",
        pdb_text({"X": [(10, "", "A"), (11, "", "G"), (12, "", "S")]}),
    )
    reference = ReferenceModel.from_target(full_structure=target, target_chains=("X",))
    reference.add_protein_binder("CG", preferred_chain="B")
    renamed = StructureArtifact.from_text(
        "renamed",
        pdb_text(
            {
                "A": [(1, "", "A"), (2, "", "G"), (3, "", "S")],
                "B": [(1, "", "C"), (2, "", "G")],
            }
        ),
    )

    mapped = reference.map_structure_auto(renamed)

    assert mapped.residue_map.to_canonical(LocalResidueId("A", 1)).entity_id == "target:X"
    assert mapped.residue_map.to_canonical(LocalResidueId("B", 1)).entity_id == "binder"


def test_gemmi_parser_keeps_polymer_modifications_and_excludes_ligands() -> None:
    ensemble = """MODEL        1
ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 80.00           C
HETATM    2  CA  MSE A  11A      1.000   0.000   0.000  1.00 80.00           C
HETATM    3  C1  LIG L   1       2.000   0.000   0.000  1.00 80.00           C
HETATM    4  O   HOH W   1       3.000   0.000   0.000  1.00 80.00           O
ENDMDL
MODEL        2
ATOM      5  CA  GLY A  10       4.000   0.000   0.000  1.00 80.00           C
ENDMDL
END
"""
    first = StructureArtifact.from_text("ensemble", ensemble)
    second = StructureArtifact.from_text("ensemble-2", ensemble, model_index=1)

    assert [residue.residue_name for residue in first.residues()] == ["ALA", "MSE"]
    assert [residue.one_letter for residue in first.residues()] == ["A", "M"]
    assert [residue.one_letter for residue in second.residues()] == ["G"]
    assert first.summary()["models"] == 2
    assert set(ReferenceModel.from_target(full_structure=first).entities) == {"target:A"}


def test_structure_artifact_exposes_and_caches_live_gemmi_object(tmp_path: Path) -> None:
    artifact = StructureArtifact.from_text(
        "mutable",
        pdb_text({"A": [(1, "", "A")]}),
    )

    assert artifact.load_structure() is artifact.structure
    artifact.structure[0][0][0].seqid = gemmi.SeqId(42, " ")
    exported = artifact.export(tmp_path / "mutated.pdb")
    restored = pickle.loads(pickle.dumps(artifact, protocol=5))

    assert artifact.residues()[0].local_id.sequence_id == 42
    assert StructureArtifact.from_file("exported", exported).residues()[0].local_id.sequence_id == 42
    assert restored.residues()[0].local_id.sequence_id == 42


def test_mmcif_mapping_exports_reference_author_ids(tmp_path: Path) -> None:
    full = StructureArtifact.from_text(
        "full",
        pdb_text({"X": [(10, "", "A"), (11, "A", "G"), (12, "", "S")]}),
    )
    local_pdb = gemmi.read_pdb_string(
        pdb_text({"A": [(1, "", "G"), (2, "", "S")]})
    )
    local_pdb.setup_entities()
    local = StructureArtifact.from_text(
        "local-cif",
        local_pdb.make_mmcif_document().as_string(),
        structure_format="cif",
    )
    reference = ReferenceModel.from_target(full_structure=full, target_chains=("X",))
    mapped = reference.map_structure(local, {"A": "target:X"})

    exported = mapped.export(tmp_path / "reference.cif", numbering="reference")
    residues = StructureArtifact.from_file("reference", exported).residues()

    assert [residue.local_id for residue in residues] == [
        LocalResidueId("X", 11, "A"),
        LocalResidueId("X", 12),
    ]
