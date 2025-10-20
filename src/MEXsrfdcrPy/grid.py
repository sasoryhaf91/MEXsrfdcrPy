# src/MEXsrfdcrPy/grid.py
# =============================================================================
# MIT License
#
# (c) 2025 The MEXsrfdcrPy authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# =============================================================================
"""
Global (single-model) daily prediction utilities for station grids.

This module provides two main entry points:

1) :func:`train_global_rf_target`  
   Trains a **single** `RandomForestRegressor` using all stations that meet a
   minimum row threshold within a target period. The function persists:
   - the fitted model as a joblib artifact, and
   - a JSON metadata file describing columns, features and training settings.

   The *target variable is generic* (e.g. daily precipitation `prec`,
   minimum temperature `tmin`, maximum temperature `tmax`, evaporation `evap`,
   or any other numeric daily target available in the input table).

2) :func:`predict_grid_daily_with_global_model`  
   Loads the persisted model and produces daily predictions for **any station
   grid** (columns: station id, latitude, longitude, altitude) over a requested
   date span. The function:
   - reproduces the time features used at train time,
   - honors extra covariates if the global model included them,
   - returns or streams a long-format table **including coordinates**.

The implementation is self-contained and depends only on:
`pandas`, `numpy`, `scikit-learn`, and `joblib`. Optional `pyarrow` is used
**only** when streaming large predictions to Parquet (recommended for big grids).

The design is deliberately "plain" to be easy to reproduce, audit and extend in
a JOSS context.

Examples
--------
>>> # 1) Train a single global model for precipitation
>>> model, meta, summary = train_global_rf_target(
...     data=df,  # long table: station/date/latitude/longitude/altitude/prec
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="prec",
...     start="1991-01-01", end="2020-12-31",
...     min_rows_per_station=1825,
...     rf_params=dict(n_estimators=500, random_state=42, n_jobs=-1),
...     model_path="models/global_rf_prec.joblib",
...     meta_path="models/global_rf_prec.meta.json",
... )
>>> summary.head()

>>> # 2) Predict an arbitrary grid (with coordinates)
>>> preds = predict_grid_daily_with_global_model(
...     grid_df=grid,  # columns: station, latitude, longitude, altitude
...     model_path="models/global_rf_prec.joblib",
...     meta_path="models/global_rf_prec.meta.json",
...     start="1991-01-01", end="2020-12-31",
...     batch_days=365,
...     out_path=None,  # or "preds/global_rf_prec_grid.parquet" to stream
... )
>>> preds.head()
station | date       | y_pred_full | latitude | longitude | altitude
--------+------------+-------------+----------+-----------+---------
10001   | 1991-01-01 |   0.514     |  24.64   | -103.69   |  1950.0
...

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor


__all__ = [
    "GlobalModelMeta",
    "train_global_rf_target",
    "predict_grid_daily_with_global_model",
    "slice_station_series",
    "grid_nan_report",
]


# ---------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Return a shallow copy with a timezone-naive datetime column."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(out[date_col]):
        out[date_col] = out[date_col].dt.tz_localize(None)
    return out.dropna(subset=[date_col])


def _add_time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool) -> pd.DataFrame:
    """Add year/month/day-of-year (+ optional sin/cos) to *df* and return a copy."""
    out = df.copy()
    out["year"] = out[date_col].dt.year.astype("int16", copy=False)
    out["month"] = out[date_col].dt.month.astype("int8", copy=False)
    out["doy"] = out[date_col].dt.dayofyear.astype("int16", copy=False)
    if add_cyclic:
        out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"] / 365.25).astype("float32")
        out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"] / 365.25).astype("float32")
    return out


def _save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Public dataclass for metadata (persisted with the model)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalModelMeta:
    """Metadata persisted alongside the trained model.

    Attributes
    ----------
    id_col, date_col, lat_col, lon_col, alt_col, target_col
        Column names used at training time.
    start, end
        Training period [inclusive] in `YYYY-MM-DD` format.
    min_rows_per_station
        Minimum valid rows per station to be included in training.
    add_cyclic
        Whether cyclic time features (sin/cos of day-of-year) were used.
    rf_params
        Dictionary of `RandomForestRegressor` parameters.
    features
        Final ordered list of **feature names** expected by the model.
    n_train_rows
        Number of training rows used.
    n_stations_used
        Number of stations that met the threshold.
    stations_used_sorted
        Sorted list of station identifiers used for training.
    """

    id_col: str
    date_col: str
    lat_col: str
    lon_col: str
    alt_col: str
    target_col: str
    start: str
    end: str
    min_rows_per_station: int
    add_cyclic: bool
    rf_params: Dict
    features: List[str]
    n_train_rows: int
    n_stations_used: int
    stations_used_sorted: List[int]

    @staticmethod
    def load(path: str) -> "GlobalModelMeta":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return GlobalModelMeta(**d)

    def save(self, path: str) -> None:
        _save_json(self.__dict__, path)


# ---------------------------------------------------------------------
# 1) Train a single global RF model for a *generic* daily target
# ---------------------------------------------------------------------


def train_global_rf_target(
    data: pd.DataFrame,
    *,
    # column names in `data`
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    # time window and station inclusion rule
    start: str = "1991-01-01",
    end: str = "2020-12-31",
    min_rows_per_station: int = 1825,
    # features and model
    add_cyclic: bool = True,
    extra_feature_cols: Optional[Sequence[str]] = None,
    rf_params: Dict = dict(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1),
    # persistence
    model_path: str = "models/global_rf_target.joblib",
    meta_path: str = "models/global_rf_target.meta.json",
) -> Tuple[RandomForestRegressor, GlobalModelMeta, pd.DataFrame]:
    """Train a single global `RandomForestRegressor` for a daily *target*.

    This is target-agnostic: pass `target_col="prec"` (precipitation),
    `target_col="tmin"`, `target_col="tmax"`, `target_col="evap"`, etc.

    Parameters
    ----------
    data
        Long-format daily table with at least:
        [id_col, date_col, lat_col, lon_col, alt_col, target_col].
    id_col, date_col, lat_col, lon_col, alt_col, target_col
        Column names in `data`.
    start, end
        Inclusive date span for training (YYYY-MM-DD).
    min_rows_per_station
        Minimum valid training rows per station to be kept (e.g., 1825 ≈ 5 years).
    add_cyclic
        If True, include sin/cos(doy) features.
    extra_feature_cols
        Optional additional covariates available both at training time and later
        for grid prediction. Columns not found in `data` are ignored safely.
    rf_params
        Parameters for `sklearn.ensemble.RandomForestRegressor`.
    model_path, meta_path
        Output paths for the model (joblib) and metadata (JSON).

    Returns
    -------
    model
        Fitted `RandomForestRegressor`.
    meta
        `GlobalModelMeta` describing training settings (and the expected features).
    station_summary
        Two-column DataFrame [id_col, valid_rows] for stations used in training.
    """
    # 1) Period selection and time features
    df = _ensure_datetime(data, date_col)
    lo, hi = pd.to_datetime(start), pd.to_datetime(end)
    df = df[(df[date_col] >= lo) & (df[date_col] <= hi)].copy()
    df = _add_time_features(df, date_col=date_col, add_cyclic=add_cyclic)

    # 2) Base features + optional extras (only those that exist)
    features: List[str] = [lat_col, lon_col, alt_col, "year", "month", "doy"]
    if add_cyclic:
        features += ["doy_sin", "doy_cos"]
    if extra_feature_cols:
        for c in extra_feature_cols:
            if c in df.columns and c not in features:
                features.append(c)

    # 3) Valid rows and station filtering
    valid_mask = ~df[features + [target_col]].isna().any(axis=1)
    dfv = df.loc[valid_mask, [id_col] + features + [target_col]].copy()
    if dfv.empty:
        raise ValueError("No valid rows after filtering by features and target.")

    station_counts = (
        dfv.groupby(id_col, as_index=False)[target_col].size().rename(columns={"size": "valid_rows"})
    )
    keep = station_counts[station_counts["valid_rows"] >= int(min_rows_per_station)][id_col]
    keep_ids: List[int] = keep.astype(int).tolist()
    if not keep_ids:
        raise ValueError(
            "No stations meet the minimum row threshold. "
            f"Got {int(station_counts['valid_rows'].max())} max; "
            f"need >= {min_rows_per_station}."
        )

    df_train = dfv[dfv[id_col].astype(int).isin(keep_ids)].copy()

    # 4) Fit Random-Forest
    X = df_train[features].to_numpy(copy=False)
    y = df_train[target_col].to_numpy(copy=False)
    model = RandomForestRegressor(**rf_params)
    model.fit(X, y)

    # 5) Persist artifacts
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    dump(model, model_path)

    meta = GlobalModelMeta(
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target_col=target_col,
        start=str(lo.date()),
        end=str(hi.date()),
        min_rows_per_station=int(min_rows_per_station),
        add_cyclic=bool(add_cyclic),
        rf_params=dict(rf_params),
        features=list(features),
        n_train_rows=int(len(df_train)),
        n_stations_used=int(len(keep_ids)),
        stations_used_sorted=sorted(keep_ids),
    )
    meta.save(meta_path)

    # 6) Station summary for reporting
    station_summary = (
        station_counts[station_counts[id_col].astype(int).isin(keep_ids)]
        .sort_values("valid_rows", ascending=False)
        .reset_index(drop=True)
    )

    return model, meta, station_summary


# ---------------------------------------------------------------------
# 2) Predict a daily grid with the saved global model
# ---------------------------------------------------------------------


def predict_grid_daily_with_global_model(
    grid_df: pd.DataFrame,
    *,
    # persisted artifacts
    model_path: str = "models/global_rf_target.joblib",
    meta_path: str = "models/global_rf_target.meta.json",
    # grid columns
    grid_id_col: str = "station",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    # time span
    start: str = "1991-01-01",
    end: str = "2020-12-31",
    # memory control
    batch_days: int = 365,
    # optional output
    out_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Optional[pd.DataFrame]:
    """Predict a daily series for a station grid using the saved global model.

    Parameters
    ----------
    grid_df
        DataFrame with columns [grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col].
        Each row is a station/cell to be predicted (one row per station).
    model_path, meta_path
        Paths to the persisted model and metadata created by `train_global_rf_target`.
    start, end
        Inclusive date span for predictions (YYYY-MM-DD).
    batch_days
        Number of consecutive days per batch (trade-off between RAM and speed).
    out_path
        Optional Parquet path where predictions are streamed in batches
        (recommended for large grids). When provided, this function returns None.
    parquet_compression
        Compression codec if `out_path` is provided.

    Returns
    -------
    DataFrame or None
        If `out_path` is None, returns a DataFrame with columns:
        [grid_id_col, "date", "y_pred_full", grid_lat_col, grid_lon_col, grid_alt_col].
        If `out_path` is provided, the function streams to disk and returns None.

    Notes
    -----
    - The model expects the *same* feature columns (names/types) persisted in metadata.
      If you trained with extra covariates, the corresponding columns must be available
      here as well (per grid row and/or derivable per date).
    """
    # --- Load artifacts
    model: RandomForestRegressor = load(model_path)
    meta = GlobalModelMeta.load(meta_path)

    # --- Validate/prepare grid
    required_cols = [grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]
    missing_cols = [c for c in required_cols if c not in grid_df.columns]
    if missing_cols:
        raise ValueError(f"Grid is missing required columns: {missing_cols}")
    g = (
        grid_df[required_cols]
        .drop_duplicates(subset=[grid_id_col])
        .reset_index(drop=True)
        .copy()
    )
    if g.empty:
        raise ValueError("Grid is empty.")
    if g[grid_id_col].duplicated().any():
        raise ValueError("Grid contains duplicated station identifiers.")

    # --- Date vector
    lo, hi = pd.to_datetime(start), pd.to_datetime(end)
    all_dates = pd.date_range(start=lo, end=hi, freq="D")
    if len(all_dates) == 0:
        raise ValueError("No dates in the requested span.")

    # --- Map grid column names -> training feature names (coords)
    to_train = {
        grid_lat_col: meta.lat_col,
        grid_lon_col: meta.lon_col,
        grid_alt_col: meta.alt_col,
    }
    # For output, we need the reverse map to restore the public names
    to_public = {v: k for k, v in to_train.items()}

    # --- Optional streaming writer
    writer = None
    parts: List[pd.DataFrame] = []
    try:
        if out_path is not None:
            import pyarrow as pa  # optional dependency; used only if streaming
            import pyarrow.parquet as pq
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # --- Batch over time
        for i0 in range(0, len(all_dates), int(batch_days)):
            dates_chunk = all_dates[i0 : i0 + int(batch_days)]

            # Cartesian product grid × dates (memory-friendly per batch)
            cart = g.assign(_k=1).merge(
                pd.DataFrame({"date": dates_chunk, "_k": 1}),
                on="_k",
                how="outer",
            ).drop(columns="_k")

            # Time features as in training
            cart = _ensure_datetime(cart, "date")
            cart = _add_time_features(cart, "date", add_cyclic=meta.add_cyclic)

            # Rename coord columns to the training names
            cart = cart.rename(columns=to_train)

            # Ensure all required features exist
            missing = [c for c in meta.features if c not in cart.columns]
            if missing:
                raise ValueError(
                    "Missing feature columns for prediction. "
                    f"Expected {meta.features}, missing {missing}."
                )

            # Predict
            X = cart[meta.features].to_numpy(copy=False)
            y_hat = model.predict(X)

            # --- Build output *including* coordinates
            out = cart[[grid_id_col, "date", meta.lat_col, meta.lon_col, meta.alt_col]].copy()
            out = out.rename(columns=to_public)  # back to public names (latitude/longitude/altitude)
            out["y_pred_full"] = y_hat.astype("float32")

            # Column order
            out = out[[grid_id_col, "date", grid_lat_col, grid_lon_col, grid_alt_col, "y_pred_full"]]

            # Accumulate or stream
            if out_path is None:
                parts.append(out)
            else:
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression=parquet_compression)
                writer.write_table(table)

            # Light progress (useful for very large grids)
            print(f"[grid-predict] {len(dates_chunk)} days × {len(g)} stations")

    finally:
        if writer is not None:
            writer.close()

    if out_path is None:
        return pd.concat(parts, ignore_index=True)
    return None


# ---------------------------------------------------------------------
# Small convenience utilities
# ---------------------------------------------------------------------


def slice_station_series(
    preds: pd.DataFrame,
    station_id: int,
    *,
    id_col: str = "station",
    date_col: str = "date",
) -> pd.DataFrame:
    """Return the time series for a single station from a long-format predictions table.

    Parameters
    ----------
    preds
        Long-format DataFrame as returned by `predict_grid_daily_with_global_model`
        (when `out_path=None`).
    station_id
        Station identifier to slice.
    id_col, date_col
        Column names in `preds`.

    Returns
    -------
    DataFrame
        Subset with rows for `station_id` sorted by date.
    """
    out = preds.loc[preds[id_col].astype(int) == int(station_id)].copy()
    if out.empty:
        return out
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def grid_nan_report(
    grid_df: pd.DataFrame,
    *,
    id_col: str = "station",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
) -> pd.DataFrame:
    """Quick NA report for a grid table.

    Returns a one-row dataframe with counts of missing values per required column.
    Useful before calling `predict_grid_daily_with_global_model`.
    """
    required = [id_col, lat_col, lon_col, alt_col]
    report = {c: int(grid_df[c].isna().sum()) if c in grid_df.columns else None for c in required}
    return pd.DataFrame([report])

