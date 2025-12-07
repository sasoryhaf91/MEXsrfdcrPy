# tests/test_grid.py
import numpy as np
import pandas as pd
import pytest

from MEXsrfdcrPy.grid import (
    train_global_rf_target,
    predict_grid_daily_with_global_model,
    slice_station_series,
    grid_nan_report,
    predict_points_daily_with_global_model,
    predict_at_point_daily_with_global_model,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def toy_daily_data() -> pd.DataFrame:
    """Small synthetic daily dataset with 3 stations and 10 days."""
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    rows = []
    for st in (1, 2, 3):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "station": st,
                    "date": d,
                    "latitude": 19.0 + 0.1 * st,
                    "longitude": -99.0 - 0.1 * st,
                    "altitude": 2000.0 + 10.0 * st,
                    # simple, deterministic target
                    "prec": float(st + i),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_global_model(tmp_path, toy_daily_data):
    """Train a small global RF model and return (paths, meta, summary)."""
    model_path = tmp_path / "global_model.joblib"
    meta_path = tmp_path / "global_model.meta.json"

    model, meta, summary = train_global_rf_target(
        toy_daily_data,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-01-10",
        min_rows_per_station=5,
        add_cyclic=True,
        extra_feature_cols=None,
        rf_params=dict(
            n_estimators=50,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        model_path=str(model_path),
        meta_path=str(meta_path),
    )
    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "model": model,
        "meta": meta,
        "summary": summary,
    }


# ---------------------------------------------------------------------
# Tests for training
# ---------------------------------------------------------------------


def test_train_global_rf_target_basic(tmp_path, toy_daily_data, trained_global_model):
    """Global training returns a RF, metadata and a reasonable station summary."""
    model = trained_global_model["model"]
    meta = trained_global_model["meta"]
    summary = trained_global_model["summary"]

    # Model type
    from sklearn.ensemble import RandomForestRegressor

    assert isinstance(model, RandomForestRegressor)

    # Features include coords + time features
    for col in ("latitude", "longitude", "altitude", "year", "month", "doy"):
        assert col in meta.features

    # All three stations should be used (10 valid rows each)
    assert meta.n_stations_used == 3
    assert summary["valid_rows"].min() >= 5

    # Metadata numbers consistent with training dataframe
    assert meta.n_train_rows == int(summary["valid_rows"].sum())


# ---------------------------------------------------------------------
# Tests for grid prediction
# ---------------------------------------------------------------------


def test_predict_grid_daily_with_global_model_shape(trained_global_model, toy_daily_data):
    """Grid prediction returns expected shape and columns."""
    model_path = trained_global_model["model_path"]
    meta_path = trained_global_model["meta_path"]

    # Build a grid with one row per station
    grid_df = (
        toy_daily_data[["station", "latitude", "longitude", "altitude"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )

    preds = predict_grid_daily_with_global_model(
        grid_df=grid_df,
        model_path=model_path,
        meta_path=meta_path,
        grid_id_col="station",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        start="2000-01-01",
        end="2000-01-05",
        batch_days=5,
        out_path=None,  # do not require pyarrow in tests
    )

    # 3 stations × 5 days
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == 3 * 5

    for col in ("station", "date", "latitude", "longitude", "altitude", "y_pred_full"):
        assert col in preds.columns

    # No NaNs in predictions
    assert np.isfinite(preds["y_pred_full"]).all()


def test_slice_station_series(trained_global_model, toy_daily_data):
    """slice_station_series returns a sorted subset for one station."""
    model_path = trained_global_model["model_path"]
    meta_path = trained_global_model["meta_path"]

    grid_df = (
        toy_daily_data[["station", "latitude", "longitude", "altitude"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )

    preds = predict_grid_daily_with_global_model(
        grid_df=grid_df,
        model_path=model_path,
        meta_path=meta_path,
        start="2000-01-01",
        end="2000-01-03",
        batch_days=3,
        out_path=None,
    )

    s1 = slice_station_series(preds, station_id=1, id_col="station", date_col="date")
    assert len(s1) == 3
    assert (s1["station"].unique() == [1]).all()
    # dates are sorted
    assert list(s1["date"]) == sorted(s1["date"].tolist())


# ---------------------------------------------------------------------
# Tests for helpers & point prediction wrappers
# ---------------------------------------------------------------------


def test_grid_nan_report():
    """grid_nan_report counts NaNs per required column."""
    grid_df = pd.DataFrame(
        {
            "station": [1, 2],
            "latitude": [19.0, np.nan],
            "longitude": [-99.0, -99.1],
            "altitude": [2000.0, np.nan],
        }
    )
    rep = grid_nan_report(grid_df)
    assert rep.shape == (1, 4)
    assert rep["station"].iloc[0] == 0  # no NaNs in station
    assert rep["latitude"].iloc[0] == 1
    assert rep["longitude"].iloc[0] == 0
    assert rep["altitude"].iloc[0] == 1


def test_predict_points_and_at_point_daily_with_global_model(trained_global_model):
    """Point-based wrappers return reasonable daily series."""
    model_path = trained_global_model["model_path"]
    meta_path = trained_global_model["meta_path"]

    # 1) multiple points via list of (lat, lon, alt) -> station ids auto 1..N
    points = [
        (19.0, -99.0, 2200.0),
        (19.5, -99.5, 2300.0),
    ]
    preds_pts = predict_points_daily_with_global_model(
        points,
        model_path=model_path,
        meta_path=meta_path,
        start="2000-01-01",
        end="2000-01-03",
        batch_days=3,
        out_path=None,
    )

    # 2 stations × 3 days
    assert preds_pts.shape[0] == 2 * 3
    assert {"station", "latitude", "longitude", "altitude", "date", "y_pred_full"}.issubset(
        preds_pts.columns
    )

    # 2) single point wrapper
    preds_single = predict_at_point_daily_with_global_model(
        latitude=19.0,
        longitude=-99.0,
        altitude=2200.0,
        station=99,
        model_path=model_path,
        meta_path=meta_path,
        start="2000-01-01",
        end="2000-01-03",
        batch_days=3,
        out_path=None,
    )

    assert preds_single["station"].nunique() == 1
    assert preds_single["station"].iloc[0] == 99
    assert len(preds_single) == 3
    assert np.isfinite(preds_single["y_pred_full"]).all()
