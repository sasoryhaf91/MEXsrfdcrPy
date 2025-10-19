---
title: 'MEXsrfdcrPy: Spatial Random Forest for Daily Climate Records Reconstruction in Mexico'
tags:
  - Python
  - random forest
  - climate
  - Mexico
  - rainfall 
  - temperature
  - land evaporation
authors:
  - name: Hugo Antonio-Fernandez
    orcid: 0000-0002-5355-8476
    affiliation: 1
affiliations:
  - name: Colegio de Postgraduados, Universidad Mexiquense del Bicentenario
    index: 1
date: 2025-01-01
bibliography: paper.bib
---

# Summary

`MEXsrfdcrPy` is a Python package to train and evaluate spatial random forest models for reconstructing daily climate records using station metadata (latitude, longitude, altitude), optional covariates (e.g., gridded predictors), and a leave-one-station-out (LOSO) validation. It targets fast exploration, neighbor-aware training, and grid-based diagnostics.

# Statement of need

Many climate applications require temporally complete daily series for precipitation and temperature. Observational station networks are spatially heterogeneous and exhibit missing values. `MEXsrfdcrPy` implements an efficient and reproducible workflow to *reconstruct* daily values at a given station using neighboring stations, with options to include a percent of the target station samples for sensitivity analyses. It provides scalable evaluation (fast LOSO), spatial filtering by nearest neighbors, and hooks to integrate gridded predictors.

# Research objectives

The package aims to: (i) enable fast LOSO evaluation with neighbor constraints, (ii) quantify expected reconstruction error as a function of station density/distance, (iii) generate full daily series predictions, and (iv) provide reproducible tables and maps that support network planning and uncertainty communication.

# Methodology

We use ensemble tree models (Random Forest) for daily regression tasks. For each target station, the training pool is composed of neighboring stations or the entire network (excluding the target), with optional inclusion of a small percentage of target observations for controlled leakage tests. The model uses station metadata and calendar features (year, month, day of year, optional cyclic encoding). Metrics are reported daily, monthly, and annually (MAE, RMSE, R²), with robust handling of edge cases (e.g., zero variance in the target).

# Implementation

The core evaluates stations with a single-pass preprocessing and buffered logging. Spatial neighbor selection is accelerated via KDTree over station centroids (lat/lon). The package supports full-series prediction in user-defined date ranges, and exports results to CSV/Parquet. Gridded predictors can be plugged as additional features. The API favors batch evaluation and easy integration with notebooks and workflows.

# Validation

We conduct LOSO experiments across Mexican stations from SMN and NASA-like gridded sources (when available). The package reports expected errors vs. distance-to-nearest station and fraction of network coverage, reproducing typical gradients observed in sparse/denser regions. Results can be summarized as tables and maps to guide network expansion and uncertainty-aware applications.

# Availability

The source code is hosted on GitHub: <https://github.com/sasoryhaf91/MEXsrfdcrPy> and distributed under the MIT license. Example notebooks and scripts (Kaggle-friendly) are provided to reproduce key figures and tables.

# Acknowledgements

We thank the open-source community and the maintainers of scientific Python libraries used in this work.

# References
