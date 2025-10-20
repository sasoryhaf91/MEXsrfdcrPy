# src/MEXsrfdcrPy/gie.py
"""
Global precipitation model and grid prediction utilities.

This module provides two top-level functions:

- train_global_rf_prec: Train a single Random-Forest model using all stations
  that meet a minimum data threshold in a target period, then persist both
  the model (joblib) and its metadata (JSON).

- predict_grid_daily_with_global_model: Load the persisted model and produce a
  daily prediction for an arbitrary station grid (station, latitude, longitude,
  altitude) over a requested date span, streaming in day-batches to limit RAM.

Both functions are framework-agnostic and rely only on pandas, numpy,
scikit-learn, and joblib.

Author: Your Name
License: MIT
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
    "train_global_rf_prec",
    "predict_grid_daily_with_global_model",
]


# ---------------------------------------------------------------------
# Small, private helpers (self-contained to keep this module portable)
# ---------------------------------------------------------------------


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Return a shallow copy of *df* with a timezone-naive datetime column."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(out[date_col]):
        out[date_col] = out[date_col].dt.tz_localize(None)
    return out.dropna(subset=[date_col])


def _add_time_features(
    df: pd.DataFrame, date_col: str, add_cyclic: bool
) -> pd.DataFrame:
    """Add year/month/day-of-year (+ optional sin/cos) to *df* and return a copy."""
    out = df.copy()
    out["year"] = out[date_col].dt.year.astype("int16", copy=False)
    out["month"] = out[date_col].dt.month.astype("int8", copy=False)
    out["doy"] = out[date_col].dt.dayofyear.astype("int16", copy=False)
    if add_cyclic:
        # Avoid float64 bloat; float32 is plenty for tree models.
        out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"] / 365.25).astype("float32")
        out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"] / 365.25).astype("float32")
    return out


def _save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Public dataclass for metadata (useful in tests & docs)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalModelMeta:
    """Metadata persisted alongside the trained model.

    Attributes
    ----------
    id_col, date_col, lat_col, lon_col, alt_col, target_col
        Column names used at training time.
    start, end
        Training period [inclusive].
    min_rows_per_station
        Minimum valid rows per station to be included in training.
    add_cyclic
        Whether cyclic time features were used.
    rf_params
        Dictionary of RandomForestRegressor parameters.
    features
        Final ordered list of feature names expected by the model.
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
# 1) Train a single global RF model for precipitation
# ---------------------------------------------------------------------


def train_global_rf_prec(
    data: pd.DataFrame,
    *,
    # columns in `data`
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
    model_path: str = "models/global_rf_prec_1991_2020.joblib",
    meta_path: str = "models/global_rf_prec_1991_2020.meta.json",
) -> Tuple[RandomForestRegressor, GlobalModelMeta, pd.DataFrame]:
    """Train a global Random-Forest for daily precipitation and persist it.

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
        Two-column dataframe [id_col, valid_rows] for stations used in training.

    Notes
    -----
    - The model expects the same feature columns (names and types) when used later
      in `predict_grid_daily_with_global_model`.
    - All numeric features are passed as-is to the tree ensemble; no scaling
      is required.
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
            f"Got {station_counts['valid_rows'].max()} max; need >= {min_rows_per_station}."
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
    model_path: str = "models/global_rf_prec_1991_2020.joblib",
    meta_path: str = "models/global_rf_prec_1991_2020.meta.json",
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
) -> pd.DataFrame:
    """Predict a daily series for a station grid using the saved global model.

    Parameters
    ----------
    grid_df
        DataFrame with columns [grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col].
        Each row is a station/cell to be predicted.
    model_path, meta_path
        Paths to the persisted model and metadata created by `train_global_rf_prec`.
    start, end
        Inclusive date span for predictions (YYYY-MM-DD).
    batch_days
        Number of consecutive days per batch (trade-off between RAM and speed).
    out_path
        Optional parquet path where predictions are saved (long format).
    parquet_compression
        Compression codec if `out_path` is provided.

    Returns
    -------
    preds
        Long-format DataFrame with columns [grid_id_col, "date", "y_pred_full"].

    Notes
    -----
    - This function assumes the *same* feature set as persisted in metadata.
      If you trained with extra covariates, the corresponding columns must be
      available here too, with proper values per grid and day.
    """
    # 1) Load artifacts
    model: RandomForestRegressor = load(model_path)
    meta = GlobalModelMeta.load(meta_path)

    # 2) Validate/prepare grid
    g = grid_df[[grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]].copy()
    if g[grid_id_col].duplicated().any():
        raise ValueError("Grid contains duplicated station identifiers.")
    if g.empty:
        raise ValueError("Grid is empty.")

    # 3) Build date vector and batch through time
    lo, hi = pd.to_datetime(start), pd.to_datetime(end)
    all_dates = pd.date_range(start=lo, end=hi, freq="D")
    if len(all_dates) == 0:
        raise ValueError("No dates in the requested span.")

    # 4) Create rename map from grid colnames to training feature names
    rename_map = {
        grid_lat_col: meta.lat_col,
        grid_lon_col: meta.lon_col,
        grid_alt_col: meta.alt_col,
    }

    # 5) Loop in time batches
    rows_out: List[pd.DataFrame] = []
    for i0 in range(0, len(all_dates), int(batch_days)):
        dates_chunk = all_dates[i0 : i0 + int(batch_days)]

        # Cartesian product: grid × dates
        cart = g.assign(_k=1).merge(
            pd.DataFrame({"date": dates_chunk, "_k": 1}),
            on="_k",
            how="outer",
        ).drop(columns="_k")

        # Prepare time features just like training
        cart = _ensure_datetime(cart, "date")
        cart = _add_time_features(cart, "date", add_cyclic=meta.add_cyclic)

        # Map grid coordinate names to training feature names
        cart = cart.rename(columns=rename_map)

        # Check feature availability
        missing = [c for c in meta.features if c not in cart.columns]
        if missing:
            raise ValueError(
                "Missing feature columns for prediction. "
                f"Expected {meta.features}, missing {missing}."
            )

        # Predict
        yhat = model.predict(cart[meta.features].to_numpy(copy=False))
        out = cart[[grid_id_col, "date"]].copy()
        out["y_pred_full"] = yhat.astype("float32")
        rows_out.append(out)

        # (Optional) progress print for long runs
        print(f"[grid-predict] {len(dates_chunk)} days × {len(g)} stations")

    preds = pd.concat(rows_out, axis=0, ignore_index=True)

    # 6) Persist if requested
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        preds.to_parquet(out_path, index=False, compression=parquet_compression)

    return preds
