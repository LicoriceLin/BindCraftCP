"""Cysteine-removal redesign workflow factory.

The old mutable Pipeline subclass has been replaced by a ProteinMPNN recipe and
an optional reusable validation Flow.
"""

from epitopecraft.workflows.redesign import build_remove_cysteine_workflow

__all__ = ["build_remove_cysteine_workflow"]
