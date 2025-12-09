---
title: "MEXsrfdcrPy: Spatial Random Forests for Daily Climate Records Reconstruction in Mexico"
tags:
  - Python
  - climatology
  - meteorology
  - spatial interpolation
  - missing data
  - random forest
  - open science
authors:
  - name: "Hugo Antonio-Fernández"
    orcid: "0000-0002-5355-8476"
    affiliation: "1, 2"
  - name: "Humberto Vaquera-Huerta"
    orcid: "0000-0002-2805-804X"
    affiliation: 1
  - name: "Moisés Michel Rosengaus-Moshinsky"
    affiliation: 3
  - name: "Paulino Pérez-Rodríguez"
    orcid: "0000-0002-3202-1784"
    affiliation: 1
  - name: "José Crossa"
    orcid: "0000-0001-9429-5855"
    affiliation: 4
affiliations:
  - name: "Colegio de Postgraduados, México"
    index: 1
  - name: "Universidad Mexiquense del Bicentenario, México"
    index: 2
  - name: "Independent Consultant"
    index: 3
  - name: "CIMMYT, México"
    index: 4
date: 2025-12-09
bibliography: paper.bib
---

# Summary  

Daily climate station records underpin hydrological design, agricultural planning and climate-risk assessment, yet long series are almost always affected by gaps, relocations and inconsistent maintenance [@Menne2012]. In México, the national network maintained by the Servicio Meteorológico Nacional (SMN) comprises thousands of stations with heterogeneous length and completeness, complicating nationwide analyses and the calibration of gridded products [@CONAGUA2012SMN135].

**MEXsrfdcrPy** is a Python package that addresses this problem by training **one global random forest model** on all available daily observations using only geographic coordinates and calendar information—latitude, longitude, elevation and time—and then using this model to reconstruct missing values, quantify spatial interpolation skill station by station, and generate climate grids at user-defined resolutions. The same workflow applies to daily precipitation, minimum and maximum temperature, and evaporation.

The package builds on NumPy, pandas and scikit-learn [@Harris2020; @McKinney2010; @Pedregosa2011] and is designed for interactive notebooks, scripted workflows and large-scale jobs on platforms such as Kaggle. MEXsrfdcrPy forms the spatial interpolation layer, with calendar-based temporal features, of a broader open-source ecosystem for Mexican climate data, complementing tools for station-data access and local spatial–temporal imputation [@Antonio-Fernandez_2025_SMNdataR; @Antonio-Fernandez2025_MissClimatePy].

# Statement of need

Spatio-temporal interpolation of station data is commonly performed with kriging, inverse distance weighting or thin-plate splines [@Hijmans2005; @Hofstra2008]. These methods can yield high-quality maps but often require variable-specific configuration and explicit covariance modeling. In parallel, machine-learning approaches—particularly random forests—have become popular for spatial prediction because they capture nonlinear responses and interactions without strong distributional assumptions [@Breiman2001; @Hengl2018]. Many such applications, however, rely on rich sets of auxiliary predictors that are not always available at station locations or for long historical periods.

MEXsrfdcrPy targets a complementary use case: **reconstructing daily station series and generating gridded fields using only coordinates and time**. The package assumes that a substantial fraction of the climatological signal can be learned from latitude, longitude, elevation and day of year when training data aggregate decades of observations from a dense national network. This design makes the method easy to deploy across variables and periods, avoids dependence on external reanalyses or remote-sensing products, and yields reusable global models that can be archived with DOIs and shared between projects. The approach is suited to national and regional studies that require transparent reconstruction of long daily series for crop modeling, drought analysis or climate-change attribution [@Xu2024DeepLearningClimate; @Ruane2015AgMERRA].

# Model and implementation

At its core, MEXsrfdcrPy fits a spatial random forest to a long-format daily table with station identifier, date, latitude, longitude, elevation and a single numeric target. Time is represented through year, month and day of year, with optional sinusoidal terms for the annual cycle. A single `RandomForestRegressor` [@Breiman2001; @Pedregosa2011] is trained using all stations that exceed a user-defined threshold of valid observations within a specified training window, and the fitted model plus metadata—column names, feature set, training period and station coverage—are persisted as joblib and JSON artifacts for reproducibility and reuse.

The package is organized around two high-level modules. The **`loso`** module implements leave-one-station-out (LOSO) experiments: each station is excluded from training and then predicted across the evaluation period. Results are summarized in tidy tables with three metrics—mean absolute error (MAE), root mean squared error (RMSE) and the coefficient of determination (R²)—computed on daily values and on user-defined temporal aggregations, following best practices for spatial cross-validation [@Roberts2017]. The **`grid`** module reuses a single global model to predict any station grid or regular mesh defined by latitude, longitude and elevation, over any compatible date interval. Internally, the package uses NumPy arrays for fitting and prediction and delegates parallelization to scikit-learn’s multi-core implementation, keeping dependencies minimal and installation straightforward.

# Examples

MEXsrfdcrPy is illustrated using a 1991–2020 SMN daily dataset for México [@AntonioFernandez2025SMN]. A typical analysis first runs LOSO experiments over a region to quantify station-by-station MAE, RMSE and R², and then trains a global model on the full national network to generate continuous fields on a fine grid for the same period.
@fig-station11020 shows a representative station-level comparison produced directly with the package plotting utilities. Daily rainfall at an SMN station is contrasted with three alternative sources of information: an external gridded product from NASA POWER (variable PRECTOTCORR) [@Stackhouse2018], a local spatial–temporal random forest model obtained with MissClimatePy [@Antonio-Fernandez2025_MissClimatePy], and a grid-based global model derived from MEXsrfdcrPy applied to a high-resolution mesh. The legend reports MAE, RMSE and R² for each series, showing that, at this station, the global spatial random forest can match or outperform both the external product and the local model. The figure thus illustrates the dual role of MEXsrfdcrPy as a reconstruction engine and as a benchmarking tool against existing datasets.

![Daily rainfall at station 11020 (México) comparing observations, NASA POWER precipitation (PRECTOTCORR), a local spatial random forest model (SRFI), and a grid-based global model (Grid Model). The plot is generated with MEXsrfdcrPy, and the legend reports MAE, RMSE and R² for each series.](figures/station_11020_comparison.png){#fig-station11020}

# Related work

MEXsrfdcrPy sits at the intersection of traditional spatial interpolation, machine-learning-based mapping and open climate-data workflows. It complements geostatistical approaches and global datasets such as WorldClim and CHELSA [@Hijmans2005; @Karger2017] by focusing on the reconstruction of **station series** and by using a single, reusable random forest model trained solely on coordinates and time. The design is inspired by studies demonstrating the strong performance of random forests for spatial and spatio-temporal prediction [@Hengl2018] and by best practices for spatial cross-validation [@Roberts2017]. By building on NumPy, pandas and scikit-learn [@Harris2020; @McKinney2010; @Pedregosa2011] and integrating naturally with Python-based ecosystems for climate and agricultural modeling, MEXsrfdcrPy provides a pragmatic, fully reproducible path from raw, incomplete national station archives to interpolated daily series and high-resolution climate grids.

# Acknowledgements

This work was supported by the Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI) through a doctoral scholarship to the first author. We acknowledge Colegio de Postgraduados and Universidad Mexiquense del Bicentenario for institutional support. We also thank the International Maize and Wheat Improvement Center (CIMMYT) for fostering collaboration in open climate and agricultural research.
