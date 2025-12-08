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
date: YYYY-MM-DD
bibliography: paper.bib
---

# Summary

Reliable daily climate information is fundamental for hydrology, agriculture and risk assessment, yet national station networks are temporally incomplete and spatially irregular. MEXsrfdcrPy is a Python package that reconstructs daily precipitation, temperature and evaporation in Mexico using a single global Random Forest model driven only by latitude, longitude, altitude and time [@breiman2001random]. The model is trained on long-format station records and implemented with NumPy, pandas and scikit-learn [@harris2020numpy; @mckinney2010data; @pedregosa2011scikit]. 

From this compact representation, MEXsrfdcrPy fills gaps in station series, quantifies leave-one-station-out interpolation skill and generates daily climate grids at arbitrary spatial resolutions. Performance is reported with three core metrics—mean absolute error (MAE), root mean squared error (RMSE) and the coefficient of determination (R²)—computed on daily values for each station. The same fitted model can then be projected to one-kilometre, five-kilometre or user-defined grids without retraining, and reconstructed series can be compared visually and quantitatively against global products such as NASA POWER [@nasapower] and against local XYZT-based models from MissClimatePy [@fernandez2024missclimatepy]. Within an open ecosystem that also includes SMNdataR for harmonised Mexican station data [@fernandez2024smndataR], MEXsrfdcrPy provides the spatial pillar for reproducible climate reconstruction in Mexico.

# Statement of need

In Mexico, many applications still depend on historical station records from the Servicio Meteorológico Nacional, where long gaps and heterogeneous spatial coverage are the rule rather than the exception. At the same time, widely used gridded datasets and reanalyses, such as WorldClim and ERA5, provide broad coverage but are not routinely evaluated at the level of individual stations and regions relevant for national decision making [@hijmans2005worldclim; @fick2017worldclim2; @hersbach2020era5]. Practitioners need methods that learn directly from the available network, reconstruct missing daily values and expose interpolation skill station by station.

Traditional interpolation approaches, including inverse distance weighting, thin-plate splines and kriging, remain powerful but are usually applied variable by variable and grid by grid, and do not naturally yield a single reusable model that can be transferred across variables and resolutions [@haylock2008eobs]. Many machine learning approaches, in turn, rely on dense covariates that are not consistently available over the historical record. MEXsrfdcrPy addresses this gap by adopting a deliberately minimal but expressive predictor set—only coordinates and time—to train a global Random Forest that can reconstruct daily precipitation, minimum and maximum temperature or evaporation, while reporting MAE, RMSE and R² at each station through a leave-one-station-out design.

# Functionality and implementation

MEXsrfdcrPy operates on long-format daily tables with station identifiers, dates, coordinates and a numeric target variable. Dates are normalised to timezone-naive timestamps and expanded into calendar features, including year, month and day-of-year, with optional sinusoidal encoding to represent seasonality. A training routine selects a user-defined period, filters stations that meet a minimum number of valid observations and builds a feature matrix using latitude, longitude, altitude and time. A global RandomForestRegressor is then fitted with scikit-learn [@pedregosa2011scikit], using NumPy arrays and pandas DataFrames for efficient computation [@harris2020numpy; @mckinney2010data]. The model and its metadata—feature list, training period, hyperparameters and station coverage—are persisted with joblib for later reuse.

Prediction is organised around two complementary modes. In the leave-one-station-out mode, each station is held out in turn, its daily series is reconstructed over an evaluation window and MAE, RMSE and R² are computed from observed and predicted daily values. These metrics can be mapped or aggregated by hydrological region, elevation band or administrative unit to reveal spatial patterns in interpolation skill. In the grid-prediction mode, users supply a table of grid points or stations defined by identifiers and coordinates. The package constructs the Cartesian product of points and dates in memory-efficient batches, regenerates the time features used at training, applies the global model and returns or streams a long-format table with identifiers, coordinates, dates and reconstructed daily values. Convenience utilities normalise flexible point specifications into grid tables, provide quick NA diagnostics for candidate grids and extract time series for individual stations, while plotting helpers compare MEXsrfdcrPy reconstructions against NASA POWER and local MissClimatePy models at selected locations [@nasapower; @fernandez2024missclimatepy].

# Example

A typical workflow begins by downloading and cleaning daily station records with SMNdataR to obtain a harmonised table of precipitation, minimum and maximum temperature and evaporation for the Mexican network [@fernandez2024smndataR]. This table, containing station identifiers, dates and coordinates, is passed to MEXsrfdcrPy to train a global Random Forest for a selected target, for example daily precipitation over a reference period such as 1981–2010. The fitted model is then used to perform a leave-one-station-out evaluation, yielding MAE, RMSE and R² for each station, which can be summarised by basin or elevation and mapped to identify areas of robust performance and areas where the model degrades.

After assessing skill, the same model is projected onto a user-defined grid, such as a one-kilometre grid over a watershed or agricultural planning region. The resulting daily fields can be stored as Parquet files and further aggregated to monthly or seasonal indices. At selected sites, reconstructed series from MEXsrfdcrPy are plotted together with NASA POWER and with station-wise MissClimatePy reconstructions, providing a direct visual and quantitative comparison among a global product, local models and the global XYZT-based Random Forest [@nasapower; @fernandez2024missclimatepy].

# Related work

MEXsrfdcrPy lies at the intersection of station-based interpolation, global gridded climate products and machine learning for environmental data. Classical geostatistical methods and thin-plate splines have long been used to build interpolated climate normals and high-resolution maps, as in WorldClim and related datasets [@hijmans2005worldclim; @fick2017worldclim2], while reanalyses such as ERA5 provide dynamically consistent global fields that are increasingly used in impact studies [@hersbach2020era5]. Regional networks such as E-OBS illustrate the value of carefully curated station datasets combined with spatial interpolation for climate monitoring [@haylock2008eobs]. In parallel, there is growing interest in random forests and other ensemble methods for downscaling and interpolating climate variables because of their robustness to non-linear responses and mixed predictor scales [@breiman2001random].

Within this landscape, MEXsrfdcrPy adopts a deliberately simple but robust modelling strategy: a single Random Forest trained only on coordinates and time, capable of reconstructing daily precipitation, temperature and evaporation over a complex national domain. By design, it complements SMNdataR, which provides reproducible access to Mexican station data, and MissClimatePy, which focuses on station-level temporal imputation using XYZT-based models [@fernandez2024smndataR; @fernandez2024missclimatepy]. Together with visual comparison against global products such as NASA POWER [@nasapower], this package turns the Mexican station network into both a training resource and a benchmark for open climate reconstruction.

# Acknowledgements

This work was supported by the Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI) through a doctoral scholarship to the first author. We acknowledge Colegio de Postgraduados and Universidad Mexiquense del Bicentenario for institutional support. We also thank the International Maize and Wheat Improvement Center (CIMMYT) for fostering collaboration in open climate and agricultural research.

# References
