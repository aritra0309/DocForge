"""Discovery engine and software registry for DocForge."""

from docforge.discovery.engine import DiscoveryEngine, DiscoveryError
from docforge.discovery.registry import Registry, RegistryEntry, load_registry

__all__ = [
    "DiscoveryEngine",
    "DiscoveryError",
    "Registry",
    "RegistryEntry",
    "load_registry",
]
