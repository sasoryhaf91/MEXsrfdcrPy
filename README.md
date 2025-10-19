# MEXsrfdcrPy

**Spatial Random Forest for Daily Climate Records Reconstruction in Mexico.**

This repository hosts `MEXsrfdcrPy`, a Python package to train and evaluate spatial random-forest models for reconstructing daily climate variables using station metadata (lat, lon, alt), optional covariates (e.g., gridded products), and a LOSO-style validation.

## Key features
- Fast LOSO validation with optional neighbor constraints.
- Grid validation methodology (density-to-grid scaling).
- Support for partial inclusion of target station (0–100%) to study reconstruction power.
- Ready-to-extend: hooks for gridded predictors and multi-variable training.

## Install (dev)
```bash
pip install -e .
