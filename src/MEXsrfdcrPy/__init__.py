"""
MEXsrfdcrPy — Spatial Random Forest for Daily Climate Records Reconstruction in Mexico.
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = ["hello", "about", "__version__"]

# Attempt to read distribution version; fall back during local dev
try:
    __version__ = version("MEXsrfdcrPy")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .core import hello, about
