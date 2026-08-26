import math
from pathlib import Path

import pytest

from epitopecraft.core.recipes import (
    FilterRecipeBook,
    PenaltyRecipeBook,
    RecipeSecurityError,
    SafeExpression,
    TrustedCallableRegistry,
)


CONFIG_DIR = Path("epitopecraft/pipelines/config")


def test_current_filter_expression_keeps_its_boolean_semantics():
    recipes = FilterRecipeBook.from_file(CONFIG_DIR / "default_filter.yaml")
    passing = {
        "refold:multimer-1:pLDDT": 0.85,
        "refold:multimer-2:pLDDT": 0.90,
        "refold:multimer-1:i-pAE": 0.30,
        "refold:multimer-2:i-pAE": 0.60,
        "refold:best:binder_rmsd": 2.0,
        "aux:kappa2": 0.2,
    }
    failing = {**passing, "refold:multimer-1:i-pAE": 0.45}

    passed = recipes.evaluate("after:refold", passing)
    failed = recipes.evaluate("after:refold", failing)

    assert passed.passed
    assert passed.rules["refold:i-pAE"]
    assert not failed.passed
    assert not failed.rules["refold:i-pAE"]


def test_safe_expression_rejects_imports_and_private_attributes():
    with pytest.raises(RecipeSecurityError):
        SafeExpression("__import__('os').system('id')")
    with pytest.raises(RecipeSecurityError):
        SafeExpression("metrics.__class__")


def test_current_mpnn_recipe_produces_expected_overlapping_biases():
    recipes = PenaltyRecipeBook.from_file(CONFIG_DIR / "mpnn-sat-n-charge.json")
    rows = [
        {"seq": "C", "surf": True, "ppi": False, "aatype": "+", "unsat_ppi_polar": 1},
        {"seq": "A", "surf": False, "ppi": True, "aatype": "-", "unsat_ppi_polar": 0},
    ]

    plan = recipes.apply(rows)

    c_index = plan.amino_acids.index("C")
    k_index = plan.amino_acids.index("K")
    assert plan.mutable == (True, False)
    assert plan.bias[0][c_index] == -3_000_000.0
    assert plan.bias[0][k_index] == pytest.approx(
        math.log(0.5) + math.log(0.25) + math.log(0.5)
    )
    assert all(value == 0.0 for value in plan.bias[1])


def test_trusted_python_hook_is_explicit_and_registered():
    registry = TrustedCallableRegistry()
    registry.register("positive_charge", lambda row: row["charge"] > 0)
    recipe = PenaltyRecipeBook.from_mapping(
        {
            "positive": {
                "selector_hook": "positive_charge",
                "penalties_aas": {"No": "C"},
            }
        },
        callable_registry=registry,
        allow_python_hooks=True,
    )

    plan = recipe.apply([{"seq": "A", "charge": 1}])

    assert plan.mutable == (True,)


def test_filter_python_hook_uses_the_same_explicit_registry():
    registry = TrustedCallableRegistry()
    registry.register("quality", lambda metrics: metrics["quality"] > 0.8)
    recipes = FilterRecipeBook(
        {"custom": {"predicate_hook": "quality"}},
        {"final": ["custom"]},
        callable_registry=registry,
        allow_python_hooks=True,
    )

    assert recipes.evaluate("final", {"quality": 0.9}).passed
