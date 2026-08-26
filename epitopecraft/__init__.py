"""Protein and ligand design workflows.

The package root deliberately avoids importing modeling backends.  Import a
backend or legacy Step from its defining module so configuration and cache
tools remain usable in lightweight environments.
"""

__version__ = "0.2.0.dev0"

__all__ = ["__version__"]
