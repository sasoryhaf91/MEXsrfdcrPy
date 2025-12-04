# SPDX-License-Identifier: MIT
"""
Global Random Forest grid models for daily climate reconstruction.

This module implements a small, high-level API to:

- Train a *single* global Random Forest model over all stations, using only
  coordinates and calendar features as predictors.
- Save the trained model plus a compact metadata file to disk.
- Use that model to generate **daily series on a grid**, either:
    - for many points at once (e.g. a 1-km grid over Mexico), or
    - for a single point (e.g. one station / location of interest).

The design philosophy is deliberately simple:

- **Inputs**: latitude, longitude, altitude + (year, month, day-of-year)
  and optional cyclic encodings of DOY.
- **Outputs**: a single climate variable (e.g. ``prec``, ``tmin``,
  ``tmax``, ``evap``) reconstructed at daily resolution.
- **Model**: a global :class:`RandomForestRegressor` trained on all stations
  that pass a minimum data-availability filter.

Typical workflow
----------------

1. Train a global model over a reference period
   (e.g. 1991–2020) and save it to disk::

       from MEXsrfdcrPy.grid import train_global_rf_target

       model, meta, summary = train_global_rf_target(
           data,
           id_col="station",
           date_col="date",
           lat_col="latitude",
           lon_col="longitude",
           alt_col="altitude",
           target_col="prec",
           start="1991-01-01",
           end="2020-12-31",
           min_rows_per_station=1825,   # ~ 5 years of data
           add_cyclic=True,
           rf_params=dict(
               n_estimators=300,
               max_depth=30,
               random_state=42,
               n_jobs=-1,
           ),
           model_path="/path/to/RainfallModel-IMEX.joblib",
           meta_path="/path/to/RainfallModel-IMEX.meta.json",
       )

2. Later (or in another environment, e.g. Kaggle), load the saved model
   and predict daily series on a grid of points::

       from MEXsrfdcrPy.grid import predict_grid_daily_with_global_model

       grid = pd.DataFrame(
           {
               "station": [1001, 1002],
               "latitude": [19.5, 19.6],
               "longitude": [-99.0, -98.9],
               "altitude": [2200, 2300],
           }
       )

       df_pred = predict_grid_daily_with_global_model(
           grid,
           model_path="/path/to/RainfallModel-IMEX.joblib",
           meta_path="/path/to/RainfallModel-IMEX.meta.json",
           start="1991-01-01",
           end="2020-12-31",
           grid_id_col="station",
           grid_lat_col="latitude",
           grid_lon_col="longitude",
           grid_alt_col="altitude",
           date_col="date",
       )

       # Columns: ["station", "date", "y_pred_full"]

3. For a single point, use the convenience wrapper::

       from MEXsrfdcrPy.grid import predict_at_point_daily_with_global_model

       series = predict_at_point_daily_with_global_model(
           latitude=19.5,
           longitude=-99.0,
           altitude=2200.0,
           station=1001,
           model_path="/path/to/RainfallModel-IMEX.joblib",
           meta_path="/path/to/RainfallModel-IMEX.meta.json",
           start="1991-01-01",
           end="2020-12-31",
       )

       # Columns: ["station", "date", "y_pred_full"]

Backwards compatibility
-----------------------

The metadata class :class:`GlobalRFMeta` is designed to be compatible with
older JSON files produced by early versions of this project. In particular:

- It understands legacy keys like ``"features"``, ``"n_train_rows"``,
  and ``"n_stations_used"``.
- It ignores unknown keys such as ``"stations_used_sorted"``.
- It always exposes a stable attribute ``feature_cols`` used at
  prediction time.

This means that existing models (e.g. previously trained on Kaggle and
distributed with a ``.joblib`` + ``.meta.json`` pair) remain usable
without any modifications.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------
# Small I/O helpers (internal)
# ---------------------------------------------------------------------


def _ensure_parent_dir(path: Optional[str]) -> None:
    """Create the parent directory for *path* if needed (no-op on None)."""
    if path is None:
        return
    d = os.path.dirname(str(path)) or "."
    os.makedirs(d, exist_ok=True)


def _save_json(obj: dict, path: Optional[str]) -> Optional[str]:
    """
    Save a Python dictionary as a pretty-printed JSON file.

    Parameters
    ----------
    obj:
        Object to serialise (must be JSON-serialisable).
    path:
        Output path. If ``None``, nothing is written.

    Returns
    -------
    str or None
        The output path, or ``None`` if no file was written.
    """
    if path is None:
        return None
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def _load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return it as a Python dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Time helpers (internal)
# ---------------------------------------------------------------------


def _ensure_datetime(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Return a shallow copy with a timezone-naive datetime column.

    Parameters
    ----------
    df:
        Input DataFrame.
    date_col:
        Column containing date-like values.

    Returns
    -------
    DataFrame
        Copy of *df* with ``date_col`` converted to ``datetime64[ns]``,
        dropped rows where conversion failed, and any timezone removed.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if isinstance(out[date_col].dtype, pd.DatetimeTZDtype):
        out[date_col] = out[date_col].dt.tz_localize(None)
    return out.dropna(subset=[date_col])


def _add_time_features(
    df: pd.DataFrame,
    date_col: str = "date",
    *,
    add_cyclic: bool = False,
) -> pd.DataFrame:
    """
    Append basic time features (year, month, day-of-year) and optionally
    cyclic encodings of the day-of-year.

    Parameters
    ----------
    df:
        Input DataFrame containing a datetime column.
    date_col:
        Name of the datetime column.
    add_cyclic:
        If ``True``, add ``doy_sin`` and ``doy_cos`` features.

    Returns
    -------
    DataFrame
        Copy of *df* with additional columns ``year``, ``month``, ``doy``,
        and, if requested, ``doy_sin``, ``doy_cos``.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    if add_cyclic:
        out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"] / 365.25)
        out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"] / 365.25)
    return out


# ---------------------------------------------------------------------
# Metadata container (backwards compatible)
# ---------------------------------------------------------------------


@dataclass
class GlobalRFMeta:
    """
    Lightweight metadata container for global Random Forest grid models.

    This class is designed to be *backwards compatible* with older JSON
    metadata files produced by earlier versions of this project, while
    providing a stable schema going forward.

    All fields have default values so that missing keys in old JSON
    files do not raise errors. The :meth:`from_dict` constructor
    understands legacy keys such as ``"features"``, ``"n_train_rows"``,
    and ``"n_stations_used"``.
    """

    # Basic info
    version: str = "0.1.0"

    # Column names
    target_col: str = "prec"
    id_col: str = "station"
    date_col: str = "date"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    alt_col: str = "altitude"

    # Temporal coverage (informational)
    start: Optional[str] = None
    end: Optional[str] = None

    # Training filter (informational)
    min_rows_per_station: Optional[int] = None

    # Random Forest configuration
    rf_params: Dict[str, Any] = field(default_factory=dict)

    # Features actually used for training (order matters)
    feature_cols: List[str] = field(default_factory=list)

    # Optional summary info
    n_stations: Optional[int] = None
    n_rows: Optional[int] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlobalRFMeta":
        """
        Build :class:`GlobalRFMeta` from a raw dict.

        Backwards compatibility rules
        -----------------------------

        - Accept older metadata that use ``"features"`` instead of
          ``"feature_cols"``.
        - Map ``"n_train_rows"`` → ``n_rows`` and
          ``"n_stations_used"`` → ``n_stations`` when present.
        - Ignore unknown keys such as ``"stations_used_sorted"``.
        - If no explicit feature list is found, reconstruct a sensible
          default based on the stored column names and flags like
          ``"add_cyclic"`` / ``"use_cyclic"`` / ``"var_cols"``.
        """
        meta = dict(d)  # shallow copy so we can modify safely

        # Default version if missing
        meta.setdefault("version", "0.1.0")

        # ---- Features ----
        # 1) Prefer explicit "feature_cols"
        if meta.get("feature_cols"):
            pass
        # 2) Legacy key "features"
        elif meta.get("features"):
            meta["feature_cols"] = list(meta["features"])
        # 3) Fallback: reconstruct from column names
        else:
            lat_col = meta.get("lat_col", "latitude")
            lon_col = meta.get("lon_col", "longitude")
            alt_col = meta.get("alt_col", "altitude")

            feats: List[str] = [lat_col, lon_col, alt_col, "year", "month", "doy"]

            # Some older metadata stored a boolean for cyclic encodings
            use_cyclic = bool(
                meta.get("add_cyclic", False) or meta.get("use_cyclic", False)
            )
            if use_cyclic:
                feats += ["doy_sin", "doy_cos"]

            extra = (
                meta.get("var_cols")
                or meta.get("extra_features")
                or []
            )
            if isinstance(extra, str):
                extra = [extra]
            feats.extend(list(extra))

            meta["feature_cols"] = feats

        # ---- Map legacy counts ----
        # n_train_rows -> n_rows
        if "n_rows" not in meta and "n_train_rows" in meta:
            meta["n_rows"] = meta["n_train_rows"]
        else:
            meta.setdefault("n_rows", None)

        # n_stations_used -> n_stations
        if "n_stations" not in meta and "n_stations_used" in meta:
            meta["n_stations"] = meta["n_stations_used"]
        else:
            meta.setdefault("n_stations", None)

        # Ensure min_rows_per_station exists
        meta.setdefault("min_rows_per_station", None)

        # ---- Drop unknown keys (e.g. stations_used_sorted) ----
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {k: v for k, v in meta.items() if k in allowed}

        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the metadata into a JSON-friendly dictionary.
        """
        return {
            "version": self.version,
            "target_col": self.target_col,
            "id_col": self.id_col,
            "date_col": self.date_col,
            "lat_col": self.lat_col,
            "lon_col": self.lon_col,
            "alt_col": self.alt_col,
            "start": self.start,
            "end": self.end,
            "min_rows_per_station": self.min_rows_per_station,
            "rf_params": self.rf_params,
            "feature_cols": list(self.feature_cols),
            "n_stations": self.n_stations,
            "n_rows": self.n_rows,
        }


# ---------------------------------------------------------------------
# Training: global RF model for a single target variable
# ---------------------------------------------------------------------


def train_global_rf_target(
    data: pd.DataFrame,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_rows_per_station: int = 365,
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    rf_params: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    meta_path: Optional[str] = None,
) -> Tuple[RandomForestRegressor, GlobalRFMeta, pd.DataFrame]:
    """
    Train a global Random Forest model over all stations for a single
    climate target (precipitation, temperature, etc.).

    The model uses only *spatial* (lat, lon, alt) and *calendar*
    (year, month, day-of-year, optionally cyclic DOY) features.

    Parameters
    ----------
    data:
        Long-format DataFrame with at least the columns::

            id_col | date_col | lat_col | lon_col | alt_col | target_col

    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names in *data*.
    start, end:
        Optional training period (inclusive). If ``None``, the full
        temporal extent of *data* is used.
    min_rows_per_station:
        Minimum number of valid (non-NaN) target rows required for a
        station to be included in the training set.
    add_cyclic:
        If ``True``, use cyclic encodings of the day-of-year (``doy_sin``,
        ``doy_cos``).
    feature_cols:
        Optional explicit feature list. If ``None``, the default
        ``[lat, lon, alt, year, month, doy]`` plus optional cyclic
        encodings is used.
    rf_params:
        Parameters for :class:`sklearn.ensemble.RandomForestRegressor`.
        If ``None``, a conservative default configuration is used.
    model_path:
        Optional path where the trained RF model is saved via joblib.
        If ``None``, the model is not written to disk.
    meta_path:
        Optional path where the metadata JSON is saved. If ``None``,
        the metadata is not written to disk.

    Returns
    -------
    model:
        Fitted :class:`RandomForestRegressor` instance.
    meta:
        :class:`GlobalRFMeta` describing the trained model.
    summary:
        DataFrame with one row per station used for training, including
        the number of valid rows contributing to the fit.
    """
    if rf_params is None:
        rf_params = dict(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

    # 1) Basic datetime handling and optional clipping
    df = _ensure_datetime(data, date_col=date_col)
    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    if df.empty:
        raise ValueError("No rows left after applying the training period filter.")

    # 2) Add time features
    df = _add_time_features(df, date_col=date_col, add_cyclic=add_cyclic)

    # 3) Default feature list if not explicitly given
    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # 4) Keep only rows with complete predictors + target
    mask_valid = ~df[feats + [target_col]].isna().any(axis=1)
    df = df.loc[mask_valid].copy()

    if df.empty:
        raise ValueError("No valid rows after filtering by predictors + target.")

    # 5) Minimum rows per station
    counts = df.groupby(id_col)[target_col].size().astype(int)
    stations_used = counts[counts >= int(min_rows_per_station)].index.tolist()
    if not stations_used:
        raise ValueError(
            "No stations passed the 'min_rows_per_station' filter. "
            f"min_rows_per_station={min_rows_per_station}"
        )

    df_train = df[df[id_col].isin(stations_used)].copy()

    # 6) Fit the Random Forest
    X = df_train[feats].to_numpy(copy=False)
    y = df_train[target_col].to_numpy(copy=False)
    model = RandomForestRegressor(**rf_params)
    model.fit(X, y)

    # 7) Metadata
    meta = GlobalRFMeta(
        version="0.1.0",
        target_col=target_col,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        start=str(start) if start is not None else None,
        end=str(end) if end is not None else None,
        min_rows_per_station=int(min_rows_per_station),
        rf_params=dict(rf_params),
        feature_cols=list(feats),
        n_stations=len(stations_used),
        n_rows=int(len(df_train)),
    )

    # 8) Per-station training summary
    summary = (
        df_train.groupby(id_col)[target_col]
        .size()
        .rename("n_rows")
        .reset_index()
        .astype({id_col: int, "n_rows": int})
    )

    # 9) Optional persistence
    if model_path is not None:
        _ensure_parent_dir(model_path)
        joblib_dump(model, model_path)

    if meta_path is not None:
        _save_json(meta.to_dict(), meta_path)

    return model, meta, summary


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------


def _date_range_from_meta_or_args(
    meta: GlobalRFMeta,
    start: Optional[str],
    end: Optional[str],
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve a date range from explicit arguments and/or metadata.

    If *start* or *end* is ``None``, fall back to ``meta.start`` /
    ``meta.end`` respectively. If no valid information can be found,
    a :class:`ValueError` is raised.
    """
    if start is None:
        start = meta.start
    if end is None:
        end = meta.end

    if start is None or end is None:
        raise ValueError(
            "Both 'start' and 'end' must be provided either as function "
            "arguments or in the metadata."
        )

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts > end_ts:
        raise ValueError(f"start ({start_ts}) is after end ({end_ts}).")

    return start_ts, end_ts


def predict_points_daily_with_global_model(
    points: pd.DataFrame,
    *,
    model_path: str,
    meta_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    grid_id_col: str = "station",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Predict daily series on a set of points using a previously trained
    global RF model stored on disk.

    Parameters
    ----------
    points:
        DataFrame with at least the columns::

            grid_id_col | grid_lat_col | grid_lon_col | grid_alt_col

        Each row is interpreted as one spatial point (e.g. a grid cell or
        a station).
    model_path:
        Path to the RF model file saved by :func:`train_global_rf_target`.
    meta_path:
        Path to the JSON metadata file saved by
        :func:`train_global_rf_target`.
    start, end:
        Date range for the prediction (inclusive). If either is ``None``,
        the function falls back to ``meta.start`` / ``meta.end`` from the
        metadata file, and raises a :class:`ValueError` if the information
        is not available.
    grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col:
        Column names in *points* for the point identifier, latitude,
        longitude and altitude.
    date_col:
        Name for the date column in the returned DataFrame.

    Returns
    -------
    DataFrame
        A long-format DataFrame with columns::

            grid_id_col | date_col | y_pred_full

        where ``y_pred_full`` is the predicted daily series of the
        target variable specified in the metadata (e.g. ``prec``).
    """
    if points.empty:
        raise ValueError("The 'points' DataFrame is empty.")

    # 1) Load model & metadata
    model: RandomForestRegressor = joblib_load(model_path)
    meta_dict = _load_json(meta_path)
    meta = GlobalRFMeta.from_dict(meta_dict)

    # 2) Resolve date range
    start_ts, end_ts = _date_range_from_meta_or_args(meta, start, end)
    dates = pd.date_range(start=start_ts, end=end_ts, freq="D")

    # 3) Basic validation of input columns
    required = [grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]
    missing = [c for c in required if c not in points.columns]
    if missing:
        raise ValueError(f"'points' is missing required columns: {missing}")

    pts = points.copy()

    # 4) Build the full (point x day) table via a cross join
    pts = pts[[grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]].copy()
    pts["__key__"] = 1
    dates_df = pd.DataFrame({date_col: dates, "__key__": 1})
    df = pd.merge(pts, dates_df, on="__key__").drop(columns="__key__")

    # 5) Rename columns to match the training schema
    df_ren = df.rename(
        columns={
            grid_id_col: meta.id_col,
            grid_lat_col: meta.lat_col,
            grid_lon_col: meta.lon_col,
            grid_alt_col: meta.alt_col,
            date_col: meta.date_col,
        }
    )

    # 6) Add calendar features
    df_ren = _add_time_features(df_ren, date_col=meta.date_col, add_cyclic=False)

    # Ensure at least the standard features exist; compute cyclic encodings
    # only if needed by meta.feature_cols.
    if "doy" not in df_ren.columns:
        df_ren["doy"] = df_ren[meta.date_col].dt.dayofyear
    if "year" not in df_ren.columns:
        df_ren["year"] = df_ren[meta.date_col].dt.year
    if "month" not in df_ren.columns:
        df_ren["month"] = df_ren[meta.date_col].dt.month

    if "doy_sin" in meta.feature_cols and "doy_sin" not in df_ren.columns:
        df_ren["doy_sin"] = np.sin(2.0 * np.pi * df_ren["doy"] / 365.25)
    if "doy_cos" in meta.feature_cols and "doy_cos" not in df_ren.columns:
        df_ren["doy_cos"] = np.cos(2.0 * np.pi * df_ren["doy"] / 365.25)

    # 7) Extract features in the exact order used during training
    feats = list(meta.feature_cols)
    missing_feats = [f for f in feats if f not in df_ren.columns]
    if missing_feats:
        raise ValueError(
            "The following feature columns required by the model are "
            f"missing from the constructed design matrix: {missing_feats}"
        )

    X = df_ren[feats].to_numpy(copy=False)
    y_pred = model.predict(X)

    # 8) Build output with the original grid naming
    out = df_ren[[meta.id_col, meta.date_col]].copy()
    out.rename(
        columns={meta.id_col: grid_id_col, meta.date_col: date_col},
        inplace=True,
    )
    out["y_pred_full"] = y_pred
    out = out.sort_values([grid_id_col, date_col]).reset_index(drop=True)
    return out


def predict_grid_daily_with_global_model(
    grid: pd.DataFrame,
    *,
    model_path: str,
    meta_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    grid_id_col: str = "station",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Convenience wrapper around
    :func:`predict_points_daily_with_global_model` for a DataFrame
    called ``grid``.

    Parameters
    ----------
    grid:
        DataFrame with point metadata (see
        :func:`predict_points_daily_with_global_model`).
    model_path, meta_path, start, end, grid_id_col, grid_lat_col,
    grid_lon_col, grid_alt_col, date_col:
        See :func:`predict_points_daily_with_global_model`.

    Returns
    -------
    DataFrame
        Same as :func:`predict_points_daily_with_global_model`.
    """
    return predict_points_daily_with_global_model(
        points=grid,
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


def predict_at_point_daily_with_global_model(
    *,
    latitude: float,
    longitude: float,
    altitude: float,
    station: Optional[int] = None,
    model_path: str,
    meta_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    grid_id_col: str = "station",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Predict the daily series for a **single point** using a global RF
    model saved on disk.

    This is a thin convenience wrapper around
    :func:`predict_points_daily_with_global_model`. It is particularly
    useful in interactive environments (e.g. Kaggle notebooks) when you
    want to compare:

    - in-situ observations at one station,
    - LOSO-based reconstructions, and
    - the global grid model at the station coordinates.

    Parameters
    ----------
    latitude, longitude, altitude:
        Spatial coordinates of the point.
    station:
        Optional identifier for the point. If ``None``, the identifier
        is set to ``0``.
    model_path, meta_path, start, end, grid_id_col, grid_lat_col,
    grid_lon_col, grid_alt_col, date_col:
        See :func:`predict_points_daily_with_global_model`.

    Returns
    -------
    DataFrame
        A DataFrame with columns::

            grid_id_col | date_col | y_pred_full

        For the default arguments, this becomes::

            station | date | y_pred_full
    """
    if station is None:
        station = 0

    pts = pd.DataFrame(
        {
            grid_id_col: [int(station)],
            grid_lat_col: [float(latitude)],
            grid_lon_col: [float(longitude)],
            grid_alt_col: [float(altitude)],
        }
    )

    return predict_points_daily_with_global_model(
        points=pts,
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


# ---------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------

__all__ = [
    "GlobalRFMeta",
    "train_global_rf_target",
    "predict_points_daily_with_global_model",
    "predict_grid_daily_with_global_model",
    "predict_at_point_daily_with_global_model",
]

