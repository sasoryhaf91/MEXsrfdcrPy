# src/MEXsrfdcrPy/__init__.py
# SPDX-License-Identifier: MIT
"""
MEXsrfdcrPy
===========

Spatial Random Forest tools for daily rainfall reconstruction and
interpolation over Mexico.

This package provides two complementary pillars:

1. **Global grid models** (:mod:`MEXsrfdcrPy.grid`)
   - Train a single Random Forest model using all stations that meet a
     minimum data threshold in a given period.
   - Persist the fitted model and its metadata to disk.
   - Predict daily series for arbitrary station grids or individual
     points (latitude/longitude/altitude) over any requested date span.

2. **LOSO workflows** (:mod:`MEXsrfdcrPy.loso`)
   - Leave-One-Station-Out (LOSO) validation with optional partial
     leakage of the target station into training.
   - Fast station–wise evaluation (with or without spatial neighbors).
   - Full-series reconstruction for one or many stations.
   - Plot helpers to compare observed data, RF reconstructions and
     external products (e.g. satellite/model rainfall such as NASA
     PRECTOTCORR).

The design is deliberately lightweight and depends only on widely used
scientific Python libraries: ``numpy``, ``pandas``, ``scikit-learn``,
``matplotlib`` and ``joblib`` (plus optional ``pyarrow`` when streaming
large prediction grids to Parquet).

Typical usage
-------------

>>> import MEXsrfdcrPy as mx
>>> # Train a single global model for daily rainfall
>>> model, meta, summary = mx.train_global_rf_target(...)
>>>
>>> # Predict a full daily series at one point
>>> series = mx.predict_at_point_daily_with_global_model(...)
>>>
>>> # Run LOSO evaluation over many stations
>>> table = mx.evaluate_all_stations_fast(...)

For details, please see the online documentation and the example
notebooks included in the repository.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------

try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - for older Python with backport
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("MEXsrfdcrPy")
except PackageNotFoundError:  # pragma: no cover - when not installed
    # Fallback for editable installs or when running from source without
    # an installed distribution.
    __version__ = "0.0.0"


# ---------------------------------------------------------------------------
# Public API re-exports
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

from .loso import (
    loso_train_predict_station,
    loso_predict_full_series,
    loso_predict_full_series_fast,
    evaluate_all_stations,
    evaluate_all_stations_fast,
    export_full_series_station,
    export_full_series_batch,
    select_stations,
    build_station_kneighbors,
    neighbor_correlation_table,
    plot_compare_obs_rf_nasa,
    ensure_datetime,
    add_time_features,
    aggregate_and_score,
    set_warning_policy,
)

__all__ = [
    "__version__",
    # Grid / global-model utilities
    "GlobalModelMeta",
    "train_global_rf_target",
    "predict_grid_daily_with_global_model",
    "slice_station_series",
    "grid_nan_report",
    "predict_points_daily_with_global_model",
    "predict_at_point_daily_with_global_model",
    # LOSO & evaluation workflows
    "loso_train_predict_station",
    "loso_predict_full_series",
    "loso_predict_full_series_fast",
    "evaluate_all_stations",
    "evaluate_all_stations_fast",
    "export_full_series_station",
    "export_full_series_batch",
    # Helpers / utilities
    "select_stations",
    "build_station_kneighbors",
    "neighbor_correlation_table",
    "plot_compare_obs_rf_nasa",
    "ensure_datetime",
    "add_time_features",
    "aggregate_and_score",
    "set_warning_policy",
]
