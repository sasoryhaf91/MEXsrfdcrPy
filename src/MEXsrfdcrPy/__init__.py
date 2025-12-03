"""
MEXsrfdcrPy — Spatial Random Forest for Daily Climate Records Reconstruction in Mexico.

This package provides tools to:

- Train and evaluate spatial Random Forest models using leave-one-station-out (LOSO)
  strategies on daily climate records.
- Export fully reconstructed (gap-free) daily series for single stations or batches.
- Train a single *global* Random Forest model and apply it to arbitrary station grids
  or point locations (lat/lon/alt) for daily prediction.
- Compute climate-oriented performance metrics, including KGE and NSE.

Submodules
----------
- metrics:   Regression metrics (MAE, RMSE, R², KGE, NSE) and aggregation helpers.
- pipeline:  LOSO-based training, evaluation and full-series export.
- grid:      Global-model training and daily prediction on grids or points.
"""

from importlib.metadata import PackageNotFoundError, version

# ---------------------------------------------------------------------
# Package version (PEP 621 / pyproject.toml compliant)
# ---------------------------------------------------------------------
try:
    __version__ = version("MEXsrfdcrPy")
except PackageNotFoundError:
    __version__ = "0.0.0"


# ---------------------------------------------------------------------
# Public submodules
# ---------------------------------------------------------------------
from . import metrics  # noqa: F401
from . import pipeline  # noqa: F401
from . import grid  # noqa: F401


# ---------------------------------------------------------------------
# Small convenience helper
# ---------------------------------------------------------------------
def about() -> str:
    """
    Return a short, human-readable package tagline.

    This is mostly intended for quick interactive introspection:

    >>> import MEXsrfdcrPy as mex
    >>> mex.about()
    'MEXsrfdcrPy: Spatial RF for daily climate reconstruction in Mexico.'
    """
    return "MEXsrfdcrPy: Spatial RF for daily climate reconstruction in Mexico."


__all__ = [
    "__version__",
    "about",
    "metrics",
    "pipeline",
    "grid",
]
