# src/MEXsrfdcrPy/grid.py
# SPDX-License-Identifier: MIT
"""
Global grid-based Random Forest utilities for daily climate reconstruction.

This module provides a small, focused toolkit to train a *single* global
Random Forest model on a station network and then use it to predict daily
values at arbitrary locations (points or grids).

Typical workflow
----------------
1. Train a global RF model for a target variable (e.g. daily rainfall):

   >>> from MEXsrfdcrPy.grid import train_global_rf_target
   >>> model_path = "RainfallModel-IMEX_v1.0.0.joblib"
   >>> meta_path = "RainfallModel-IMEX.meta.json"
   >>> train_global_rf_target(
   ...     data=df,
   ...     start="1991-01-01",
   ...     end="2020-12-31",
   ...     min_rows_per_station=3650,
   ...     target_col="prec",
   ...     id_col="station",
   ...     date_col="date",
   ...     lat_col="latitude",
   ...     lon_col="longitude",
   ...     alt_col="altitude",
   ...     rf_params={"n_estimators": 200, "random_state": 42, "n_jobs": -1},
   ...     model_path=model_path,
   ...     meta_path=meta_path,
   ... )

2. Predict at an arbitrary point (used e.g. in Kaggle notebooks):

   >>> from MEXsrfdcrPy.grid import predict_at_point_daily_with_global_model
   >>> series = predict_at_point_daily_with_global_model(
   ...     latitude=21.85,
   ...     longitude=-102.29,
   ...     altitude=1890.8,
   ...     station=1001610,
   ...     model_path=model_path,
   ...     meta_path=meta_path,
   ...     start="1991-01-01",
   ...     end="2020-12-31",
   ... )
   >>> series.head()

3. Use the RF-based series alongside LOSO-based reconstructions and
   external products (e.g. NASA) for visual and metric comparison via
   :func:`MEXsrfdcrPy.pipeline.plot_compare_obs_rf_nasa`.

Design choices
--------------
* Inputs are plain pandas DataFrames with canonical column names:

  ``station | date | latitude | longitude | altitude | prec | tmin | tmax | evap``

  These can be customised via function arguments.

* The global RF uses only *static* and *temporal* features:

  ``[lat, lon, alt, year, month, doy]`` plus optional cyclic encodings of
  day-of-year (``doy_sin, doy_cos``).

* Model and metadata are stored as:

  - a binary ``.joblib`` file (scikit-learn RF object), and
  - a small JSON sidecar with training meta-information.

* The module is intentionally metric-agnostic: it does **not** compute
  hydrological scores itself. Users are expected to rely on
  :mod:`MEXsrfdcrPy.metrics` and, when needed,
  :func:`MEXsrfdcrPy.pipeline.plot_compare_obs_rf_nasa` for KGE, NSE, etc.

Dependencies
------------
Core:
    - numpy
    - pandas
    - scikit-learn
    - joblib

Optional:
    - nothing else; plotting and LOSO logic live in :mod:`pipeline`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import json
import os

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.ensemble import RandomForestRegressor

from .pipeline import add_time_features, ensure_datetime


# ---------------------------------------------------------------------
# Small dataclass for model metadata
# ---------------------------------------------------------------------


@dataclass
class GlobalRFMeta:
    """Metadata saved alongside the global Random Forest model.

    Parameters
    ----------
    model_type:
        String identifier for the model (e.g. ``"RandomForestRegressor"``).
    target_col:
        Name of the target variable used during training.
    id_col, date_col, lat_col, lon_col, alt_col:
        Column names in the original training DataFrame.
    feature_cols:
        List of feature column names used to fit the model.
    add_cyclic:
        Whether cyclic encodings of day-of-year were used.
    rf_params:
        Dictionary of parameters passed to :class:`RandomForestRegressor`.
    train_start, train_end:
        String representation of the training period boundaries.
    n_rows:
        Number of training rows used.
    n_stations:
        Number of stations contributing to the training set.
    min_rows_per_station:
        Minimum number of valid rows required per station.
    """

    model_type: str
    target_col: str
    id_col: str
    date_col: str
    lat_col: str
    lon_col: str
    alt_col: str
    feature_cols: List[str]
    add_cyclic: bool
    rf_params: Dict
    train_start: str
    train_end: str
    n_rows: int
    n_stations: int
    min_rows_per_station: int


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _ensure_parent_dir(path: Optional[str]) -> None:
    """Create the parent directory for *path* if needed (no-op on None)."""
    if path is None:
        return
    d = os.path.dirname(str(path)) or "."
    os.makedirs(d, exist_ok=True)


def _save_json(obj: dict, path: Optional[str]) -> Optional[str]:
    """Save a JSON-able object to disk in a pretty-printed form."""
    if path is None:
        return None
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def _load_json(path: str) -> dict:
    """Load JSON from *path* and return the decoded Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Normalisation of "points" into a grid-like DataFrame
# ---------------------------------------------------------------------


def _as_df_from_points(
    obj: Union[pd.DataFrame, dict, Iterable, np.ndarray],
    *,
    grid_id_col: str,
    grid_lat_col: str,
    grid_lon_col: str,
    grid_alt_col: str,
) -> pd.DataFrame:
    """
    Convert arbitrary *points* specification to a DataFrame.

    Accepted formats
    ----------------
    * pandas.DataFrame
        Must already contain at least latitude / longitude columns.
    * dict
        Single point, e.g. ``{"latitude": 20.0, "longitude": -100.0, "altitude": 2000.0}``.
    * tuple of length 2 or 3
        Interpreted as ``(lat, lon)`` or ``(lat, lon, alt)`` for a single point.
    * list/tuple of dicts
        Each dict corresponds to a point.
    * list/tuple of tuples
        Each inner tuple is interpreted as ``(lat, lon)`` or ``(lat, lon, alt)``.
    * NumPy array
        Shape ``(n, 2)`` or ``(n, 3)`` interpreted as above.

    Any missing altitude values are set to NaN. If the identifier column
    (``grid_id_col``) is missing, a simple running index (1..n_points) is
    generated.

    Parameters
    ----------
    obj:
        Input "points" container.
    grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col:
        Standardised column names in the output DataFrame.

    Returns
    -------
    DataFrame
        With columns ``[grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]``.
    """
    # Already a DataFrame
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    # Single dict
    elif isinstance(obj, dict):
        df = pd.DataFrame([obj])
    # NumPy array
    elif isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] not in (2, 3):
            raise ValueError("NumPy array 'points' must have shape (n, 2) or (n, 3).")
        cols = [grid_lat_col, grid_lon_col] + ([grid_alt_col] if arr.shape[1] == 3 else [])
        df = pd.DataFrame(arr, columns=cols)
    # Single tuple interpreted as one point
    elif isinstance(obj, tuple) and not any(
        isinstance(x, (list, tuple, dict, pd.Series, np.ndarray)) for x in obj
    ):
        if len(obj) not in (2, 3):
            raise ValueError("Single tuple 'points' must have length 2 or 3.")
        if len(obj) == 2:
            df = pd.DataFrame(
                [{grid_lat_col: obj[0], grid_lon_col: obj[1]}],
            )
        else:
            df = pd.DataFrame(
                [{grid_lat_col: obj[0], grid_lon_col: obj[1], grid_alt_col: obj[2]}],
            )
    # Generic list/tuple container
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError("Empty 'points' sequence.")
        first = obj[0]
        # list of dicts
        if isinstance(first, dict):
            df = pd.DataFrame(obj)
        # list of tuples
        elif isinstance(first, (list, tuple, np.ndarray)):
            rows = []
            for t in obj:
                if len(t) == 2:
                    rows.append(
                        {grid_lat_col: t[0], grid_lon_col: t[1]},
                    )
                elif len(t) == 3:
                    rows.append(
                        {grid_lat_col: t[0], grid_lon_col: t[1], grid_alt_col: t[2]},
                    )
                else:
                    raise ValueError("Tuples in 'points' must have length 2 or 3.")
            df = pd.DataFrame(rows)
        else:
            raise TypeError("Unsupported 'points' element type.")
    else:
        raise TypeError("Unsupported 'points' type; use DataFrame, dict, tuple, list or ndarray.")

    # Ensure required columns exist
    if grid_lat_col not in df.columns or grid_lon_col not in df.columns:
        raise ValueError(
            f"Points must include latitude/longitude columns "
            f"('{grid_lat_col}', '{grid_lon_col}')."
        )
    if grid_alt_col not in df.columns:
        df[grid_alt_col] = np.nan

    # Ensure identifier column
    if grid_id_col not in df.columns:
        df[grid_id_col] = np.arange(1, len(df) + 1, dtype=int)

    # Standardise column order
    df = df[[grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]].copy()
    return df.reset_index(drop=True)


def _normalize_points_to_grid_df(
    points: Union[pd.DataFrame, dict, Iterable, np.ndarray],
    *,
    grid_id_col: str = "station",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
) -> pd.DataFrame:
    """
    Normalise *points* into a small grid description DataFrame.

    This is a thin wrapper around :func:`_as_df_from_points` and is the
    internal entry point used by prediction helpers.

    Parameters
    ----------
    points:
        Any of the formats described in :func:`_as_df_from_points`.
    grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col:
        Desired column names in the output.

    Returns
    -------
    DataFrame
        With columns ``[grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]``.

    Examples
    --------
    Single tuple:

    >>> _normalize_points_to_grid_df((20.0, -100.0, 2000.0))
       station  latitude  longitude  altitude
    0        1      20.0     -100.0    2000.0

    Single dict:

    >>> _normalize_points_to_grid_df({"latitude": 20.0, "longitude": -100.0})
       station  latitude  longitude  altitude
    0        1      20.0     -100.0       NaN
    """
    return _as_df_from_points(
        points,
        grid_id_col=grid_id_col,
        grid_lat_col=grid_lat_col,
        grid_lon_col=grid_lon_col,
        grid_alt_col=grid_alt_col,
    )


# ---------------------------------------------------------------------
# Training a global RF model for a target variable
# ---------------------------------------------------------------------


def train_global_rf_target(
    data: pd.DataFrame,
    *,
    target_col: str = "prec",
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_rows_per_station: int = 365,
    add_cyclic: bool = True,
    feature_cols: Optional[List[str]] = None,
    rf_params: Optional[Dict] = None,
    model_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    save_training_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[RandomForestRegressor, GlobalRFMeta, pd.DataFrame]:
    """
    Train a *single* global Random Forest model for a given target variable.

    The model is trained on all stations that have at least
    ``min_rows_per_station`` valid daily records within the selected period.

    Features
    --------
    By default, the model uses:

    * ``[lat_col, lon_col, alt_col, "year", "month", "doy"]``
    * plus optional ``"doy_sin"``, ``"doy_cos"`` when ``add_cyclic=True``.

    You can override this behaviour by providing an explicit ``feature_cols``
    list, which must be present in the pre-processed DataFrame after
    time features are added.

    Parameters
    ----------
    data:
        Long-format DataFrame containing the station network.
    target_col:
        Name of the target variable (e.g. ``"prec"``, ``"tmin"``).
    id_col, date_col, lat_col, lon_col, alt_col:
        Column names in *data*.
    start, end:
        Optional training period boundaries (inclusive).
    min_rows_per_station:
        Minimum number of valid rows (non-NaN target and features) required
        for a station to be included in the training set.
    add_cyclic:
        Whether to add cyclic day-of-year encodings.
    feature_cols:
        Optional explicit feature list. If ``None``, the default feature
        set is used.
    rf_params:
        Parameters passed to :class:`RandomForestRegressor`. If ``None``,
        a reasonable default is used.
    model_path:
        Optional path where the trained RF model is saved using joblib.
    meta_path:
        Optional path where a JSON sidecar with metadata is written.
    save_training_table_path:
        Optional path (CSV/Parquet/Feather) where the final training
        table is stored for inspection or reproducibility.
    parquet_compression:
        Compression codec used when writing Parquet.

    Returns
    -------
    model:
        Fitted :class:`RandomForestRegressor` instance.
    meta:
        :class:`GlobalRFMeta` object with training metadata.
    train_df:
        The DataFrame actually used to train the model.
    """
    # basic copy + datetime hygiene
    df = ensure_datetime(data, date_col=date_col)

    # optional temporal clipping
    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # add time features
    df = add_time_features(df, date_col=date_col, add_cyclic=add_cyclic)

    # default features
    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # validity mask
    valid_mask = ~df[feats + [target_col]].isna().any(axis=1)
    df_valid = df.loc[valid_mask].copy()

    # station-level filtering
    counts = df_valid.groupby(id_col)[target_col].count()
    keep_ids = counts[counts >= int(min_rows_per_station)].index.tolist()
    train_df = df_valid[df_valid[id_col].isin(keep_ids)].copy()

    if train_df.empty:
        raise ValueError(
            "Training set is empty after applying min_rows_per_station "
            f"={min_rows_per_station} and validity filters."
        )

    # RF parameters
    if rf_params is None:
        rf_params = dict(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

    model = RandomForestRegressor(**rf_params)
    X = train_df[feats].to_numpy(copy=False)
    y = train_df[target_col].to_numpy(copy=False)
    model.fit(X, y)

    # training metadata
    meta = GlobalRFMeta(
        model_type="RandomForestRegressor",
        target_col=target_col,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        feature_cols=feats,
        add_cyclic=bool(add_cyclic),
        rf_params=dict(rf_params),
        train_start=str(start) if start is not None else str(train_df[date_col].min().date()),
        train_end=str(end) if end is not None else str(train_df[date_col].max().date()),
        n_rows=int(len(train_df)),
        n_stations=int(len(keep_ids)),
        min_rows_per_station=int(min_rows_per_station),
    )

    # persist model + metadata + training table
    if model_path is not None:
        _ensure_parent_dir(model_path)
        joblib_dump(model, model_path)
    if meta_path is not None:
        _save_json(asdict(meta), meta_path)

    if save_training_table_path is not None:
        ext = os.path.splitext(save_training_table_path)[1].lower()
        _ensure_parent_dir(save_training_table_path)
        if ext == ".csv":
            train_df.to_csv(save_training_table_path, index=False)
        elif ext == ".parquet":
            train_df.to_parquet(
                save_training_table_path,
                index=False,
                compression=parquet_compression,
            )
        elif ext == ".feather":
            train_df.to_feather(save_training_table_path)
        else:
            raise ValueError("Unsupported extension for save_training_table_path.")

    return model, meta, train_df


# ---------------------------------------------------------------------
# Prediction with a pre-trained global model
# ---------------------------------------------------------------------


def predict_points_daily_with_global_model(
    points: Union[pd.DataFrame, dict, Iterable, np.ndarray],
    *,
    model_path: str,
    meta_path: str,
    start: str,
    end: str,
    grid_id_col: str = "station",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Predict daily values at one or many points using a pre-trained global RF.

    This is the "grid engine" of the package: it takes arbitrary locations
    (points or a coarse grid) and produces a daily time series at each
    location for the requested period.

    Parameters
    ----------
    points:
        Arbitrary container describing one or more locations. See
        :func:`_normalize_points_to_grid_df` for accepted formats.
    model_path:
        Path to the joblib file containing the trained RF model.
    meta_path:
        Path to the JSON file with :class:`GlobalRFMeta`-like information.
    start, end:
        Date range (inclusive) for the predictions.
    grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col:
        Standardised column names in the *output* DataFrame.
    date_col:
        Name of the datetime column in the output.

    Returns
    -------
    DataFrame
        With columns::

            [date_col, grid_id_col, grid_lat_col, grid_lon_col,
             grid_alt_col, "y_pred_full"]

        sorted by ``(grid_id_col, date_col)``.
    """
    # load model + metadata
    model: RandomForestRegressor = joblib_load(model_path)
    meta_dict = _load_json(meta_path)
    meta = GlobalRFMeta(**meta_dict)

    # normalise points
    grid_df = _normalize_points_to_grid_df(
        points,
        grid_id_col=grid_id_col,
        grid_lat_col=grid_lat_col,
        grid_lon_col=grid_lon_col,
        grid_alt_col=grid_alt_col,
    )

    # build the cartesian product (points × dates)
    dates = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="D")
    pts = grid_df.copy()
    pts["__key"] = 1
    dates_df = pd.DataFrame({date_col: dates, "__key": 1})
    full = pts.merge(dates_df, on="__key").drop(columns="__key")

    # add time features as done in training
    full = add_time_features(full, date_col=date_col, add_cyclic=meta.add_cyclic)

    # check that all requested features are present
    missing = set(meta.feature_cols) - set(full.columns)
    if missing:
        raise ValueError(f"Missing feature columns for prediction: {missing}")

    X = full[meta.feature_cols].to_numpy(copy=False)
    y_hat = model.predict(X)

    out = full[[date_col, grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]].copy()
    out["y_pred_full"] = y_hat
    out = out.sort_values([grid_id_col, date_col]).reset_index(drop=True)
    return out


def predict_at_point_daily_with_global_model(
    *,
    latitude: float,
    longitude: float,
    altitude: float,
    station: Optional[int] = None,
    model_path: str,
    meta_path: str,
    start: str,
    end: str,
    grid_id_col: str = "station",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Convenience wrapper for :func:`predict_points_daily_with_global_model`
    for a *single* point.

    This is the function typically used in Kaggle notebooks where a user
    wants to compare a grid-based RF model against a particular station
    (using coordinates provided by e.g. SMN).

    Parameters
    ----------
    latitude, longitude, altitude:
        Coordinates of the prediction point.
    station:
        Optional numeric station identifier, used to populate the
        ``grid_id_col`` column. If ``None``, a simple ID of 1 is used.
    model_path, meta_path:
        Paths to the RF model and JSON metadata files.
    start, end:
        Start and end dates (inclusive) for the prediction period.
    grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col, date_col:
        Column names in the output.

    Returns
    -------
    DataFrame
        Daily time series for the requested period at the given point,
        with the same structure as the output of
        :func:`predict_points_daily_with_global_model`.
    """
    point: Dict[str, float] = {
        grid_lat_col: float(latitude),
        grid_lon_col: float(longitude),
        grid_alt_col: float(altitude),
    }
    if station is not None:
        point[grid_id_col] = int(station)

    out = predict_points_daily_with_global_model(
        points=point,
        model_path=model_path,
        meta_path=meta_path,
        start=start,
        end=end,
        grid_id_col=grid_id_col,
        grid_lat_col=grid_lat_col,
        grid_lon_col=grid_lon_col,
        grid_alt_col=grid_alt_col,
        date_col=date_col,
    )
    return out


# ---------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------

__all__ = [
    "GlobalRFMeta",
    "train_global_rf_target",
    "predict_points_daily_with_global_model",
    "predict_at_point_daily_with_global_model",
    "_normalize_points_to_grid_df",
]
