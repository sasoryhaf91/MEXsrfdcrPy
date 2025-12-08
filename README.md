# MEXsrfdcrPy

Spatial Random Forests for Daily Climate Records Reconstruction in Mexico
=========================================================================

**MEXsrfdcrPy** is a Python package for reconstructing daily climate station records using **spatial random forests** and a strictly minimalist set of predictors:

> **X, Y, Z, T = (latitude, longitude, altitude, time)**

From these four inputs, the package trains a **single global Random Forest model** and uses it to reconstruct daily **precipitation, minimum and maximum temperature, or evaporation** for every station in a network. The same model can then be projected onto **grids of virtual stations** to generate daily climate fields at arbitrary spatial resolution.

The package is designed for **large observational networks** (e.g., the Mexican National Meteorological Service) where station records are long but contain substantial gaps, and where covariates such as reanalysis or satellite products may be unavailable, incomplete, or too heavy to use operationally.

MEXsrfdcrPy is part of a broader open-source ecosystem that includes:

- **SMNdataR**: R package for reproducible access to daily station records from the Mexican National Meteorological Service (SMN).
- **MissClimatePy**: Python package for local, station-wise imputation and model comparison.

Together, these tools support a fully reproducible workflow from **raw station data** to **reconstructed series** and **gridded products** for climate analysis and agricultural applications.

---

## Key features

- **Global RF model with X, Y, Z, T only**

  Trains a single `RandomForestRegressor` on all stations that meet a minimum data threshold, using only:
  latitude, longitude, altitude, and time features (year, month, day of year, optional sinusoidal seasonality).

- **Station-wise LOSO evaluation**

  Provides a **leave-one-station-out (LOSO)** pipeline to:
  - Train on all stations except one.
  - Predict the full daily series for the held-out station.
  - Compute per-station metrics (**MAE, RMSE, R²**).
  - Export complete reconstructed series as Parquet/CSV.

- **Fast network-wide evaluation**

  Runs LOSO validation over many stations, returning a **summary table** of metrics that can be used to compare spatial coverage, model performance, and station quality.

- **Grid prediction at arbitrary resolution**

  Loads a previously trained global model and predicts daily values for:
  - Custom station grids (e.g., 5 km or 1 km spacing).
  - Arbitrary points (single or multiple coordinates).
  - Any period within the training span.

- **Visual comparison with external products**

  Built-in plotting helpers to visually compare station observations, RF reconstructions, and external products such as:
  NASA POWER, ERA5, WorldClim, or local models generated with MissClimatePy.

---

## Installation

MEXsrfdcrPy is a pure Python package. Once published on PyPI, it can be installed with:

```bash
pip install MEXsrfdcrPy
