# Testing Notes

## Status

This file captures the current testing direction only.
It is not a final agreed test plan yet and should be treated as a working note before implementation.

## What The Tests Need To Protect

- recently fixed cache helpers in `design_record.py`
- recently fixed `Refold` routing and save behavior
- current pipeline wiring contracts in `HalluDesign`
- future refactors around config resolution and Step binding

## Current Direction

The likely first layer should be fast, low-dependency tests:

1. `DesignRecord` and `DesignBatch` cache behavior
2. `Refold` logic with fake models
3. `HalluDesign` wiring checks with fake steps or lightweight stubs
4. settings merge and filter recipe behavior

The likely later layer should be optional slow integration tests:

- end-to-end design smoke tests that require real heavy dependencies
- environment-gated checks for ColabDesign or PyRosetta or PyMOL

## Open Questions Still To Discuss

- whether to introduce `pytest` immediately or keep a lighter local harness first
- whether to stub heavy modules in `conftest.py` or first reduce import heaviness in the package
- how much current pipeline wiring should be frozen by tests before the refactor
- whether the existing `epitopecraft/test/hallu_refold.py` should stay as a manual script, become a slow test, or be retired

## Current Bias

Prefer tests that lock down data contracts and wiring semantics before testing expensive modeling behavior.
That means fast contract tests first, slow integration later.
