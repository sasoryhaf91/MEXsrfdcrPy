# MEXsrfdcrPy

> Spatial Random Forests for Daily Climate Records Reconstruction in Mexico

[![PyPI version](https://img.shields.io/pypi/v/MEXsrfdcrPy.svg)](https://pypi.org/project/MEXsrfdcrPy/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()
[![JOSS](https://img.shields.io/badge/preprint-JOSS-informational.svg)]()

`MEXsrfdcrPy` is a Python package for reconstructing and interpolating **daily climate station records** in Mexico using **spatial random forests**. It trains a single **global RandomForest model** on all available stations using only:

- latitude,
- longitude,
- elevation, and
- calendar information (year, month, day-of-year, optional cyclic terms),

and then uses that model to:

- fill gaps in daily station series,
- evaluate spatial interpolation skill with **leave-one-station-out** (LOSO) experiments, and
- generate gridded daily fields at arbitrary resolutions.

The package is designed for large national datasets (e.g. 1991–2020 SMN network) and integrates naturally with Jupyter, Kaggle and other Python-based workflows.

---

## Installation

From PyPI:

```bash
pip install MEXsrfdcrPy
```

or, for a local development install from GitHub:

```bash
git clone https://github.com/sasoryhaf91/MEXsrfdcrPy.git
cd MEXsrfdcrPy
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e .[dev]
```

---

## Project goals

`MEXsrfdcrPy` focuses on a simple but powerful idea:

> Learn as much as possible about daily climate patterns from **space (X, Y, Z)** and **time (T)** alone, using a single model trained on decades of national station data.

This allows you to:

- reconstruct **precipitation**, **minimum temperature**, **maximum temperature** and **evaporation** at station locations,
- quantify interpolation performance **station by station**,
- produce daily climate grids (e.g. 1/16°) for long periods using a **reusable global model**, and
- benchmark the global model against local models and external products (e.g. NASA POWER, CHIRPS).

`MEXsrfdcrPy` is part of a broader open-source ecosystem around Mexican climate data, together with:

- [`SMNdataR`](https://doi.org/10.5281/zenodo.17495178): R tools to download and process SMN station data.
- [`MissClimatePy`](https://doi.org/10.5281/zenodo.17794136): Python package for local spatial–temporal imputation at station level.

---

## Quick start

### 1. LOSO evaluation for a region

```python
import pandas as pd
from MEXsrfdcrPy.loso import evaluate_all_stations_fast

url = "https://zenodo.org/records/17636066/files/smn_mx_daily_1991_2020.csv"
data = pd.read_csv(url)

res = evaluate_all_stations_fast(
    data,
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col="prec",
    prefix=["15"],                # e.g. stations in Estado de México
    start="1991-01-01",
    end="2020-12-31",
    include_target_pct=0.0,       # strict LOSO
    min_station_rows=9125,        # ~25 years of valid data
    rf_params=dict(
        n_estimators=20,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
    ),
    show_progress=True,
)

print(res.head())
```

This returns a tidy table with **MAE**, **RMSE** and **R²** per station and per temporal aggregation.

### 2. Train a global model for reuse

```python
from MEXsrfdcrPy.grid import train_global_rf_target

model, meta, summary = train_global_rf_target(
    data,
    id_col="station", date_col="date",
    lat_col="latitude", lon_col="longitude", alt_col="altitude",
    target_col="tmin",
    start="1991-01-01", end="2020-12-31",
    min_rows_per_station=1825,
    rf_params=dict(
        n_estimators=15,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
    ),
    model_path="models/global_tmin_rf.joblib",
    meta_path="models/global_tmin_rf.meta.json",
)

print(summary.head())
```

The saved model + metadata can later be reused to predict on any grid or set of points.

### 3. Predict on a grid

```python
from MEXsrfdcrPy.grid import predict_grid_daily_with_global_model

preds = predict_grid_daily_with_global_model(
    grid_df=grid_clean,  # DataFrame with [station, latitude, longitude, altitude]
    model_path="models/global_tmin_rf.joblib",
    meta_path="models/global_tmin_rf.meta.json",
    start="1991-01-01",
    end="2020-12-31",
    batch_days=365,
    out_path="preds/global_tmin_grid_1_16deg.parquet",
)
```

When `out_path` is provided, predictions are streamed directly to Parquet.

### 4. Compare against NASA POWER and local models

```python
from MEXsrfdcrPy.loso import loso_predict_full_series_fast, plot_compare_obs_rf_nasa

station_id = 11020
y_col = "prec"

full_df, full_metrics, _, _ = loso_predict_full_series_fast(
    data,
    station_id=station_id,
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col=y_col,
    start="1991-01-01",
    end="2020-12-31",
    rf_params=dict(n_estimators=20, max_depth=30, random_state=42, n_jobs=-1),
    k_neighbors=20,
    include_target_pct=0.0,
)

NASA_COL = "PRECTOTCORR"  # NASA POWER precipitation column

ax = plot_compare_obs_rf_nasa(
    data=data,
    station_id=station_id,
    id_col="station",
    date_col="date",
    obs_col=y_col,
    nasa_col=NASA_COL,
    extra=series_1001,
    extra_date_col="date",
    extra_value_col="y_pred_full",
    extra_label="Grid Model",
    rf_df=full_df,
    rf_date_col="date",
    rf_value_col="y_pred_full",
    rf_label="SRFI (0%)",
    resample="D",
    agg="sum",
    ylabel="Rainfall [mm/day]",
    title=f"Station {station_id} — Observed vs SRFI vs {NASA_COL} vs Grid Model",
)
```

This produces a figure comparing **observations**, **NASA POWER**, a **local model** and the **global grid model**, with MAE, RMSE and R² in the legend.

---

## Documentation

Full API documentation and worked examples (Jupyter notebooks, Kaggle kernels) are planned for future releases. For now, the best reference is the docstrings in:

- `MEXsrfdcrPy.loso` – LOSO evaluation, station-level reconstruction and plotting.
- `MEXsrfdcrPy.grid` – global model training and grid/point prediction utilities.

The JOSS paper in `paper/paper.md` provides a short conceptual overview.

---

## Citation

If you use `MEXsrfdcrPy` in your work, please cite the software paper (once accepted) and the Zenodo record for this release.

**Software paper (JOSS, in review):**

> Antonio-Fernández, H., Vaquera-Huerta, H., Rosengaus-Moshinsky, M. M., Pérez-Rodríguez, P., & Crossa, J. (2025). MEXsrfdcrPy: Spatial Random Forests for Daily Climate Records Reconstruction in Mexico. *Journal of Open Source Software*.

**Zenodo record:**

> Antonio-Fernández, H., Vaquera-Huerta, H., Rosengaus-Moshinsky, M. M., Pérez-Rodríguez, P., & Crossa, J. (2025). MEXsrfdcrPy: Spatial Random Forests for Daily Climate Records Reconstruction in Mexico (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.17890904

A ready-to-use `CITATION.cff` file is included in the repository.

---

## Contributing

Contributions are very welcome! Please:

1. Open an issue on GitHub describing the bug, feature request or enhancement.
2. Fork the repository and create a feature branch.
3. Add or update tests when introducing new functionality.
4. Run the test suite (e.g. `pytest`) and ensure all tests pass.
5. Open a pull request referencing the relevant issue.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more detailed guidelines.

---

## License

`MEXsrfdcrPy` is released under the MIT license. See the [`LICENSE`](LICENSE) file for details.

---

## Maintainer and contact

The project is maintained by **Hugo Antonio-Fernández** ([@sasoryhaf91](https://github.com/sasoryhaf91)).  
Feedback, issues and pull requests are welcome via the [GitHub repository](https://github.com/sasoryhaf91/MEXsrfdcrPy).
