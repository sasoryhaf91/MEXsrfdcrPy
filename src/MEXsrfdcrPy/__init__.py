"""
MEXsrfdcrPy
===========

Spatial Random Forest for Daily Climate Records Reconstruction in Mexico.

This package provides two complementary workflows:

1. Leave-One-Station-Out (LOSO) evaluation and reconstruction
   -----------------------------------------------------------
   Tools to train and evaluate spatial Random Forest models using a
   leave-one-station-out strategy on daily station records. These
   functions are intended for *methodological evaluation* and
   reconstruction of complete daily series at existing stations.

   Main entry points
   -----------------
   - :func:`loso_train_predict_station`
   - :func:`loso_predict_full_series`
   - :func:`loso_predict_full_series_fast`
   - :func:`evaluate_all_stations_fast`
   - :func:`export_full_series_station`
   - :func:`export_full_series_batch`
   - :func:`aggregate_and_score`
   - :func:`build_station_kneighbors`
   - :func:`plot_compare_obs_rf_nasa`

2. Global model for grid / point predictions
   -----------------------------------------
   A single global Random Forest model trained on all eligible
   stations and later used to predict daily values on arbitrary
   spatial grids (or single points). This is intended for
   large-scale spatial reconstruction and map production.

   Main entry points
   -----------------
   - :class:`GlobalModelMeta`
   - :func:`train_global_rf_target`
   - :func:`predict_grid_daily_with_global_model`
   - :func:`predict_points_daily_with_global_model`
   - :func:`predict_at_point_daily_with_global_model`
   - :func:`slice_station_series`
   - :func:`grid_nan_report`

The overall design is intentionally simple: all models use only
geographical coordinates (latitude, longitude, altitude) plus
calendar time features (year, month, day-of-year, and optional
cyclic sin/cos), making the approach robust when internal
covariates are missing or unavailable.

Example
-------
    >>> import pandas as pd
    >>> from MEXsrfdcrPy import (
    ...     loso_predict_full_series,
    ...     train_global_rf_target,
    ...     predict_grid_daily_with_global_model,
    ... )

    # (1) LOSO reconstruction for a single station
    >>> full_df, metrics, model, feats = loso_predict_full_series(
    ...     df,
    ...     station_id=123,
    ...     id_col="station",
    ...     date_col="date",
    ...     lat_col="latitude",
    ...     lon_col="longitude",
    ...     alt_col="altitude",
    ...     target_col="prec",
    ...     start="1991-01-01",
    ...     end="2020-12-31",
    ... )

    # (2) Train a single global RF for precipitation
    >>> model, meta, summary = train_global_rf_target(
    ...     data=df,
    ...     id_col="station",
    ...     date_col="date",
    ...     lat_col="latitude",
    ...     lon_col="longitude",
    ...     alt_col="altitude",
    ...     target_col="prec",
    ...     start="1991-01-01",
    ...     end="2020-12-31",
    ...     model_path="models/global_prec_rf.joblib",
    ...     meta_path="models/global_prec_rf.meta.json",
    ... )

    # (3) Predict a daily grid with the global model
    >>> preds = predict_grid_daily_with_global_model(
    ...     grid_df=grid,
    ...     model_path="models/global_prec_rf.joblib",
    ...     meta_path="models/global_prec_rf.meta.json",
    ...     start="1991-01-01",
    ...     end="1991-12-31",
    ... )
"""

from __future__ import annotations

# Public version (update in sync with pyproject.toml)
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# LOSO-based spatial Random Forest utilities
# ---------------------------------------------------------------------------

from .loso import (
    aggregate_and_score,
    build_station_kneighbors,
    loso_train_predict_station,
    loso_predict_full_series,
    loso_predict_full_series_fast,
    evaluate_all_stations_fast,
    export_full_series_station,
    export_full_series_batch,
    plot_compare_obs_rf_nasa,
)

# ---------------------------------------------------------------------------
# Global RF model for grid / point daily predictions
# ---------------------------------------------------------------------------

from .grid import (
    GlobalModelMeta,
    train_global_rf_target,
    predict_grid_daily_with_global_model,
    slice_station_series,
    grid_nan_report,
    predict_points_daily_with_global_model,
    predict_at_point_daily_with_global_model,
)

__all__ = [
    "__version__",
    # LOSO workflow
    "aggregate_and_score",
    "build_station_kneighbors",
    "loso_train_predict_station",
    "loso_predict_full_series",
    "loso_predict_full_series_fast",
    "evaluate_all_stations_fast",
    "export_full_series_station",
    "export_full_series_batch",
    "plot_compare_obs_rf_nasa",
    # Global model workflow
    "GlobalModelMeta",
    "train_global_rf_target",
    "predict_grid_daily_with_global_model",
    "slice_station_series",
    "grid_nan_report",
    "predict_points_daily_with_global_model",
    "predict_at_point_daily_with_global_model",
]

