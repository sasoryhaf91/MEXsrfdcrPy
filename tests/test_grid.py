# tests/test_grid.py
# SPDX-License-Identifier: MIT
"""
Tests for MEXsrfdcrPy.grid

These tests focus on:
- Normalisation of point specifications into a grid DataFrame.
- Training a global Random Forest model for a target variable.
- Daily prediction at arbitrary points (single and multiple) using the
  trained global model.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from MEXsrfdcrPy.grid import (
    GlobalRFMeta,
    _normalize_points_to_grid_df,
    predict_at_point_daily_with_global_model,
    predict_points_daily_with_global_model,
    train_global_rf_target,
)


# ---------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------


def _make_synthetic_training_df(
    n_stations: int = 3,
    n_days: int = 30,
    start: str = "2000-01-01",
    target_col: str = "prec",
) -> pd.DataFrame:
    """
    Create a small synthetic daily dataset for multiple stations.

    The target variable is a simple linear trend plus station-specific
    offsets and Gaussian noise.
    """
    dates = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.RandomState(0)
    rows = []
    for st in range(1, n_stations + 1):
        lat = 20.0 + 0.5 * st
        lon = -100.0 - 0.5 * st
        alt = 2000.0 + 10.0 * st
        noise = rng.normal(scale=1.0, size=len(dates))
        values = 10.0 + 0.1 * st + 0.01 * np.arange(len(dates), dtype=float) + noise
        for d, v in zip(dates, values):
            rows.append(
                {
                    "station": st,
                    "date": d,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    target_col: v,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Tests for _normalize_points_to_grid_df
# ---------------------------------------------------------------------


def test_normalize_points_from_single_tuple():
    """
    _normalize_points_to_grid_df should accept a single tuple (lat, lon, alt).
    """
    out = _normalize_points_to_grid_df((20.0, -100.0, 2000.0))
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
    assert set(out.columns) == {"station", "latitude", "longitude", "altitude"}
    assert out["latitude"].iloc[0] == 20.0
    assert out["longitude"].iloc[0] == -100.0
    assert out["altitude"].iloc[0] == 2000.0
    # A synthetic station ID is created when not provided
    assert out["station"].iloc[0] == 1


def test_normalize_points_from_dict_without_station():
    """
    _normalize_points_to_grid_df should accept a single dict without station ID
    and generate a simple running index (1..n).
    """
    pt = {"latitude": 19.5, "longitude": -99.5, "altitude": 2250.0}
    out = _normalize_points_to_grid_df(pt)
    assert len(out) == 1
    assert out["latitude"].iloc[0] == 19.5
    assert out["longitude"].iloc[0] == -99.5
    assert out["altitude"].iloc[0] == 2250.0
    assert out["station"].iloc[0] == 1


def test_normalize_points_from_dataframe_preserves_station_ids():
    """
    _normalize_points_to_grid_df should preserve an existing station column.
    """
    df = pd.DataFrame(
        {
            "station": [100, 101],
            "latitude": [19.0, 19.1],
            "longitude": [-99.0, -99.1],
            "altitude": [2300.0, 2310.0],
        }
    )
    out = _normalize_points_to_grid_df(df)
    assert len(out) == 2
    assert out["station"].tolist() == [100, 101]
    assert np.allclose(out["latitude"], [19.0, 19.1])
    assert np.allclose(out["longitude"], [-99.0, -99.1])
    assert np.allclose(out["altitude"], [2300.0, 2310.0])


def test_normalize_points_from_numpy_array():
    """
    _normalize_points_to_grid_df should accept a NumPy array of shape (n, 3).
    """
    arr = np.array([[20.0, -100.0, 2000.0], [21.0, -101.0, 2100.0]])
    out = _normalize_points_to_grid_df(arr)
    assert len(out) == 2
    assert set(out.columns) == {"station", "latitude", "longitude", "altitude"}
    # Synthetic station IDs should be 1..n
    assert out["station"].tolist() == [1, 2]


# ---------------------------------------------------------------------
# Tests for train_global_rf_target
# ---------------------------------------------------------------------


def test_train_global_rf_target_returns_model_and_meta(tmp_path):
    """
    train_global_rf_target should fit a RF model and return a GlobalRFMeta.
    """
    df = _make_synthetic_training_df(n_stations=3, n_days=30)
    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"
    train_table_path = tmp_path / "global_rf_train.parquet"

    model, meta, train_df = train_global_rf_target(
        data=df,
        target_col="prec",
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        start="2000-01-01",
        end="2000-01-30",
        min_rows_per_station=10,
        add_cyclic=True,
        rf_params={"n_estimators": 10, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
        save_training_table_path=str(train_table_path),
    )

    # Basic checks on outputs
    from sklearn.ensemble import RandomForestRegressor

    assert isinstance(model, RandomForestRegressor)
    assert isinstance(meta, GlobalRFMeta)
    assert isinstance(train_df, pd.DataFrame)
    assert not train_df.empty

    # Metadata consistency
    assert meta.target_col == "prec"
    assert meta.n_stations == 3
    assert meta.n_rows == len(train_df)
    assert meta.min_rows_per_station == 10
    assert "latitude" in meta.feature_cols
    assert "year" in meta.feature_cols

    # Files should have been written
    assert os.path.exists(model_path)
    assert os.path.exists(meta_path)
    assert os.path.exists(train_table_path)


def test_train_global_rf_target_respects_min_rows_per_station():
    """
    Stations with fewer than min_rows_per_station valid rows should be
    excluded from the training set.
    """
    # Build a dataset where station 1 has 30 days, station 2 has 5 days
    full = _make_synthetic_training_df(n_stations=2, n_days=30)
    # Keep only 5 days for station 2
    mask_s2 = (full["station"] == 2) & (full["date"] >= "2000-01-01") & (
        full["date"] <= "2000-01-05"
    )
    df = pd.concat(
        [full[full["station"] == 1], full[mask_s2]],
        axis=0,
        ignore_index=True,
    )

    model, meta, train_df = train_global_rf_target(
        data=df,
        target_col="prec",
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        start="2000-01-01",
        end="2000-01-30",
        min_rows_per_station=10,
        add_cyclic=False,
        rf_params={"n_estimators": 5, "random_state": 0, "n_jobs": -1},
        model_path=None,
        meta_path=None,
        save_training_table_path=None,
    )

    # Only station 1 should be kept
    assert sorted(train_df["station"].unique().tolist()) == [1]
    assert meta.n_stations == 1
    assert meta.min_rows_per_station == 10


# ---------------------------------------------------------------------
# Tests for predictions with a global model
# ---------------------------------------------------------------------


def test_predict_points_daily_with_global_model_multiple_points(tmp_path):
    """
    predict_points_daily_with_global_model should work for multiple points.
    """
    df = _make_synthetic_training_df(n_stations=3, n_days=20)
    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"

    # Train a small model
    _, meta, _ = train_global_rf_target(
        data=df,
        target_col="prec",
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        start="2000-01-01",
        end="2000-01-20",
        min_rows_per_station=10,
        add_cyclic=True,
        rf_params={"n_estimators": 10, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    # Two arbitrary points
    points = pd.DataFrame(
        {
            "station": [1000, 1001],
            "latitude": [21.0, 22.0],
            "longitude": [-101.0, -102.0],
            "altitude": [1900.0, 1950.0],
        }
    )

    preds = predict_points_daily_with_global_model(
        points,
        model_path=str(model_path),
        meta_path=str(meta_path),
        start="2000-01-01",
        end="2000-01-05",
    )

    # 5 days × 2 points = 10 rows
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 10
    assert set(preds.columns) == {
        "date",
        "station",
        "latitude",
        "longitude",
        "altitude",
        "y_pred_full",
    }
    assert sorted(preds["station"].unique().tolist()) == [1000, 1001]
    assert preds["y_pred_full"].notna().all()

    # Date range check
    dates = preds["date"].sort_values().unique()
    assert dates[0] == pd.Timestamp("2000-01-01")
    assert dates[-1] == pd.Timestamp("2000-01-05")


def test_predict_at_point_daily_with_global_model_single_point(tmp_path):
    """
    predict_at_point_daily_with_global_model should work for a single point
    and produce a coherent daily time series.
    """
    df = _make_synthetic_training_df(n_stations=3, n_days=15)
    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"

    # Train a small model
    _, meta, _ = train_global_rf_target(
        data=df,
        target_col="prec",
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        start="2000-01-01",
        end="2000-01-15",
        min_rows_per_station=10,
        add_cyclic=True,
        rf_params={"n_estimators": 10, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    # Use coordinates of station 1 as a "point"
    st1 = df[df["station"] == 1].iloc[0]
    latitude = float(st1["latitude"])
    longitude = float(st1["longitude"])
    altitude = float(st1["altitude"])
    station_id = 9999

    preds = predict_at_point_daily_with_global_model(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        station=station_id,
        model_path=str(model_path),
        meta_path=str(meta_path),
        start="2000-01-01",
        end="2000-01-05",
    )

    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 5
    assert set(preds.columns) == {
        "date",
        "station",
        "latitude",
        "longitude",
        "altitude",
        "y_pred_full",
    }
    assert preds["station"].nunique() == 1
    assert preds["station"].iloc[0] == station_id
    assert preds["y_pred_full"].notna().all()

    # Date range check
    dates = preds["date"].sort_values().unique()
    assert dates[0] == pd.Timestamp("2000-01-01")
    assert dates[-1] == pd.Timestamp("2000-01-05")

