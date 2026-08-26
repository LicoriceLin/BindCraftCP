"""Binding-site discovery workflow factory.

This module replaces the unfinished legacy ``EpitopeDetect`` class. New code
should import the factory from ``epitopecraft.workflows`` directly.
"""

from epitopecraft.workflows.site_discovery import build_site_discovery_workflow

__all__ = ["build_site_discovery_workflow"]
