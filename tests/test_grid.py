"""
Tests for MEXsrfdcrPy.grid

These tests exercise the high-level API for training a global Random Forest
model on station data and generating daily predictions on arbitrary points
or grids.

They are designed to be:
- fast (small synthetic dataset),
- robust (do not depend on random quirks of RF),
- and focused on the public API surface used in real workflows and in JOSS.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from MEXsrfdcrPy.grid import (
    train_global_rf_target,
    predict_grid_daily_with_global_model,
    predict_points_daily_with_global_model,
    predict_at_point_daily_with_global_model,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_toy_data(
    n_stations: int = 5,
    start: str = "2000-01-01",
    end: str = "2000-01-31",
) -> pd.DataFrame:
    """
    Create a small synthetic daily dataset with a simple spatial-temporal
    signal in the target column.
    """
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(42)

    rows = []
    for st in range(1, n_stations + 1):
        lat = 19.0 + 0.1 * st
        lon = -99.0 + 0.05 * st
        alt = 2000.0 + 20.0 * st

        base = 20.0 + 0.5 * st + 0.01 * (np.arange(len(dates)))
        noise = rng.normal(scale=1.0, size=len(dates))
        target = base + noise

        for d, y in zip(dates, target):
            rows.append(
                dict(
                    station=int(st),
                    date=d,
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    target=y,
                )
            )

    df = pd.DataFrame(rows)

    # small amount of missing values
    mask = rng.random(len(df)) < 0.05
    df.loc[mask, "target"] = np.nan
    return df


def _train_and_save_small_model(tmp_path: Path, add_cyclic: bool = True):
    """
    Train a small global RF model on toy data and save the model and metadata.

    Returns
    -------
    model_path, meta_path, meta, summary
    """
    data = _make_toy_data(n_stations=6, start="2000-01-01", end="2000-01-31")

    model_path = tmp_path / "rf_model.joblib"
    meta_path = tmp_path / "rf_meta.json"

    model, meta, summary = train_global_rf_target(
        data=data,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="target",
        start="2000-01-01",
        end="2000-01-31",
        min_rows_per_station=5,
        add_cyclic=add_cyclic,
        rf_params=dict(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        ),
        model_path=str(model_path),
        meta_path=str(meta_path),
    )
    return model_path, meta_path, meta, summary


# -----------------------------------------------------------------------------
# train_global_rf_target
# -----------------------------------------------------------------------------


def test_train_global_rf_target_returns_objects(tmp_path: Path):
    model_path, meta_path, meta, summary = _train_and_save_small_model(tmp_path)

    # model_path and meta_path should be usable paths
    assert hasattr(model_path, "__str__")

    # meta object: allow either dataclass with to_dict() or plain dict
    if hasattr(meta, "to_dict"):
        d = meta.to_dict()
    elif isinstance(meta, dict):
        d = meta
    else:
        raise AssertionError("meta must be a dict or have to_dict()")

    assert "id_col" in d
    assert "target_col" in d
    assert "feature_cols" in d or "features" in d

    # summary should be a non-empty DataFrame
    assert isinstance(summary, pd.DataFrame)
    assert not summary.empty

    # files must exist
    assert Path(model_path).exists()
    assert Path(meta_path).exists()


def test_train_global_rf_target_respects_min_rows(tmp_path: Path):
    data = _make_toy_data(n_stations=4, start="2000-01-01", end="2000-01-10")
    # artificially remove one station
    data = data[data["station"] != 4]

    model_path = tmp_path / "rf_model.joblib"
    meta_path = tmp_path / "rf_meta.json"

    _, meta, summary = train_global_rf_target(
        data=data,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="target",
        start="2000-01-01",
        end="2000-01-10",
        min_rows_per_station=5,
        rf_params=dict(
            n_estimators=10,
            max_depth=5,
            random_state=0,
            n_jobs=-1,
        ),
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    if hasattr(meta, "to_dict"):
        d = meta.to_dict()
    else:
        d = meta

    # the meta must report at least one station used
    assert d.get("n_stations_used", d.get("n_stations", 0)) > 0
    assert isinstance(summary, pd.DataFrame)


# -----------------------------------------------------------------------------
# predict_points_daily_with_global_model
# -----------------------------------------------------------------------------


def test_predict_points_daily_with_global_model_shape_and_cols(tmp_path: Path):
    model_path, meta_path, meta, summary = _train_and_save_small_model(
        tmp_path, add_cyclic=True
    )

    # Build a small grid with two points
    grid = pd.DataFrame(
        {
            "station": [101, 102],
            "latitude": [19.5, 19.6],
            "longitude": [-99.0, -98.9],
            "altitude": [2200.0, 2300.0],
        }
    )

    out = predict_points_daily_with_global_model(
        points=grid,
        model_path=str(model_path),
        meta_path=str(meta_path),
        start="2000-01-01",
        end="2000-01-10",
        grid_id_col="station",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        date_col="date",
    )

    # 2 stations x 10 days = 20 rows
    assert out.shape[0] == 2 * 10
    assert set(out.columns) == {"station", "date", "y_pred_full"}

    # Sorted by station, date
    assert (out["station"].values[:10] == 101).all()
    assert (out["station"].values[10:] == 102).all()

    # Check that predictions are not all identical (there is some variability)
    all_vals = out["y_pred_full"].values
    assert np.std(all_vals) > 0


def test_predict_points_daily_with_global_model_respects_date_range(tmp_path: Path):
    model_path, meta_path, meta, summary = _train_and_save_small_model(tmp_path)

    grid = pd.DataFrame(
        {
            "station": [201],
            "latitude": [19.7],
            "longitude": [-98.8],
            "altitude": [2100.0],
        }
    )

    start = "2000-01-05"
    end = "2000-01-12"

    out = predict_points_daily_with_global_model(
        points=grid,
        model_path=str(model_path),
        meta_path=str(meta_path),
        start=start,
        end=end,
        grid_id_col="station",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        date_col="date",
    )

    dates = pd.to_datetime(out["date"])
    assert dates.min() == pd.to_datetime(start)
    assert dates.max() == pd.to_datetime(end)
    # inclusive range
    n_days = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
    assert out.shape[0] == n_days


# -----------------------------------------------------------------------------
# predict_at_point_daily_with_global_model
# -----------------------------------------------------------------------------


def test_predict_at_point_daily_with_global_model_single_series(tmp_path: Path):
    model_path, meta_path, meta, summary = _train_and_save_small_model(tmp_path)

    series = predict_at_point_daily_with_global_model(
        latitude=19.5,
        longitude=-99.0,
        altitude=2200.0,
        station=999,
        model_path=str(model_path),
        meta_path=str(meta_path),
        start="2000-01-01",
        end="2000-01-07",
        grid_id_col="station",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        date_col="date",
    )

    assert isinstance(series, pd.DataFrame)
    assert set(series.columns) == {"station", "date", "y_pred_full"}
    assert (series["station"].unique() == [999]).all()

    dates = pd.to_datetime(series["date"])
    assert dates.min() == pd.to_datetime("2000-01-01")
    assert dates.max() == pd.to_datetime("2000-01-07")
    assert series.shape[0] == 7


# -----------------------------------------------------------------------------
# predict_grid_daily_with_global_model
# -----------------------------------------------------------------------------


def test_predict_grid_daily_with_global_model_basic(tmp_path: Path):
    # We use the full helper that also goes through train_global_rf_target
    data = _make_toy_data(n_stations=5, start="2000-01-01", end="2000-01-15")

    model_path = tmp_path / "rf_model.joblib"
    meta_path = tmp_path / "rf_meta.json"

    train_global_rf_target(
        data=data,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="target",
        start="2000-01-01",
        end="2000-01-15",
        min_rows_per_station=5,
        rf_params=dict(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        ),
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    grid = pd.DataFrame(
        {
            "grid_id": [1, 2, 3],
            "latitude": [19.3, 19.4, 19.5],
            "longitude": [-99.1, -99.0, -98.9],
            "altitude": [2100.0, 2150.0, 2200.0],
        }
    )

    # NOTE: predict_grid_daily_with_global_model does NOT need the original `data`,
    # it only needs the saved model + metadata and the grid definition.
    out = predict_grid_daily_with_global_model(
        grid=grid,
        model_path=str(model_path),
        meta_path=str(meta_path),
        grid_id_col="grid_id",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        start="2000-01-05",
        end="2000-01-10",
        date_col="date",
    )

    # Implementation may return either just the prediction DF
    # or a tuple (df, meta, summary). We handle both.
    if isinstance(out, tuple):
        pred = out[0]
    else:
        pred = out

    assert isinstance(pred, pd.DataFrame)
    assert set(pred.columns) >= {"grid_id", "date", "y_pred_full"}

    # 3 grid points x 6 days = 18 rows
    assert pred.shape[0] == 3 * 6

    # date range respected
    dates = pd.to_datetime(pred["date"])
    assert dates.min() == pd.to_datetime("2000-01-05")
    assert dates.max() == pd.to_datetime("2000-01-10")
