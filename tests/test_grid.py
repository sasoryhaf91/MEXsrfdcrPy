import os
from typing import List

import numpy as np
import pandas as pd

from MEXsrfdcrPy.grid import (
    GlobalModelMeta,
    train_global_rf_target,
    predict_grid_daily_with_global_model,
    slice_station_series,
    grid_nan_report,
    predict_points_daily_with_global_model,
    predict_at_point_daily_with_global_model,
)

# NOTE: _normalize_points_to_grid_df is internal but we test it explicitly
from MEXsrfdcrPy.grid import _normalize_points_to_grid_df  # type: ignore


# ---------------------------------------------------------------------
# Small synthetic helpers
# ---------------------------------------------------------------------


def _make_synthetic_training_df(
    n_stations: int = 3,
    start: str = "2000-01-01",
    end: str = "2000-01-10",
) -> pd.DataFrame:
    """
    Build a small synthetic training dataset for grid tests.

    The "prec" target is a simple deterministic function of coordinates
    and day-of-year; we do not enforce any accuracy in the tests, only
    that the pipeline runs and returns finite values.
    """
    dates = pd.date_range(start, end, freq="D")
    rows: List[dict] = []

    # Simple lat/lon/alt pattern
    base_lat = 20.0
    base_lon = -100.0
    base_alt = 2000.0

    for sid in range(1, n_stations + 1):
        lat = base_lat + sid * 0.5
        lon = base_lon - sid * 0.5
        alt = base_alt + sid * 10.0
        for d in dates:
            # simple synthetic target: depends on station and doy
            doy = d.timetuple().tm_yday
            prec = 1.0 * sid + 0.01 * doy
            rows.append(
                {
                    "station": sid,
                    "date": d,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "prec": prec,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# GlobalModelMeta tests
# ---------------------------------------------------------------------


def test_global_model_meta_roundtrip(tmp_path):
    """GlobalModelMeta.save/load should preserve all fields."""
    meta = GlobalModelMeta(
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="1991-01-01",
        end="2020-12-31",
        min_rows_per_station=100,
        add_cyclic=True,
        rf_params={"n_estimators": 10, "random_state": 0},
        features=["latitude", "longitude", "altitude", "year"],
        n_train_rows=1234,
        n_stations_used=10,
        stations_used_sorted=[1, 2, 3],
    )

    path = tmp_path / "meta.json"
    meta.save(str(path))
    assert path.exists()

    loaded = GlobalModelMeta.load(str(path))
    assert loaded == meta
    assert loaded.features == ["latitude", "longitude", "altitude", "year"]
    assert loaded.stations_used_sorted == [1, 2, 3]


# ---------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------


def test_train_global_rf_target_returns_expected_types_and_summary(tmp_path):
    """train_global_rf_target should return model, metadata and a non-empty summary."""
    df = _make_synthetic_training_df()
    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"

    model, meta, summary = train_global_rf_target(
        data=df,
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
        rf_params={"n_estimators": 10, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    # Types and artifacts
    assert model_path.exists()
    assert meta_path.exists()
    assert isinstance(meta, GlobalModelMeta)
    assert isinstance(summary, pd.DataFrame)
    assert not summary.empty

    # Metadata consistency
    assert meta.id_col == "station"
    assert meta.target_col == "prec"
    assert meta.min_rows_per_station == 5
    assert meta.n_stations_used == len(summary)
    assert meta.n_train_rows > 0
    assert set(summary["station"].astype(int)) == set(meta.stations_used_sorted)


def test_train_global_rf_target_respects_min_rows_per_station(tmp_path):
    """Stations with too few valid rows should be excluded."""
    df = _make_synthetic_training_df(n_stations=2)
    # Drop many rows for station 2 to make it fall below the threshold
    df = df[~((df["station"] == 2) & (df["date"] > "2000-01-03"))].copy()

    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"

    _, meta, summary = train_global_rf_target(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-01-10",
        min_rows_per_station=7,  # station 2 should not reach this
        rf_params={"n_estimators": 5, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    assert set(summary["station"].astype(int)) == {1}
    assert meta.n_stations_used == 1
    assert meta.stations_used_sorted == [1]


# ---------------------------------------------------------------------
# Grid prediction tests
# ---------------------------------------------------------------------


def test_predict_grid_daily_with_global_model_returns_expected_shape(tmp_path):
    """
    Predicting on a small grid should return (n_stations * n_days) rows
    with the expected columns and finite predictions.
    """
    df = _make_synthetic_training_df(n_stations=3)
    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"

    # Train
    train_global_rf_target(
        data=df,
        start="2000-01-01",
        end="2000-01-10",
        min_rows_per_station=5,
        rf_params={"n_estimators": 10, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    # Build grid from unique station coordinates
    grid = (
        df[["station", "latitude", "longitude", "altitude"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )

    preds = predict_grid_daily_with_global_model(
        grid_df=grid,
        model_path=str(model_path),
        meta_path=str(meta_path),
        grid_id_col="station",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        start="2000-01-01",
        end="2000-01-10",
        batch_days=5,
        out_path=None,
    )

    assert isinstance(preds, pd.DataFrame)
    assert not preds.empty

    n_days = len(pd.date_range("2000-01-01", "2000-01-10", freq="D"))
    n_stations = grid["station"].nunique()
    assert len(preds) == n_days * n_stations

    expected_cols = ["station", "date", "latitude", "longitude", "altitude", "y_pred_full"]
    assert list(preds.columns) == expected_cols

    assert preds["y_pred_full"].notna().all()
    assert np.isfinite(preds["y_pred_full"]).all()


def test_predict_grid_daily_with_global_model_streams_to_parquet(tmp_path):
    """When out_path is provided, predictions should be written to Parquet and return None."""
    df = _make_synthetic_training_df(n_stations=2)
    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"

    train_global_rf_target(
        data=df,
        start="2000-01-01",
        end="2000-01-10",
        min_rows_per_station=5,
        rf_params={"n_estimators": 5, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    grid = (
        df[["station", "latitude", "longitude", "altitude"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )

    out_path = tmp_path / "grid_preds.parquet"
    result = predict_grid_daily_with_global_model(
        grid_df=grid,
        model_path=str(model_path),
        meta_path=str(meta_path),
        start="2000-01-01",
        end="2000-01-10",
        out_path=str(out_path),
    )

    assert result is None
    assert out_path.exists()
    # Basic sanity check: file is non-empty
    assert os.path.getsize(out_path) > 0


# ---------------------------------------------------------------------
# Convenience utilities
# ---------------------------------------------------------------------


def test_slice_station_series_filters_and_sorts():
    """slice_station_series should return only the requested station, sorted by date."""
    df = _make_synthetic_training_df(n_stations=2)
    # Fake predictions reuse "prec" as y_pred_full
    df_pred = df.rename(columns={"prec": "y_pred_full"})[
        ["station", "date", "latitude", "longitude", "altitude", "y_pred_full"]
    ].copy()

    sid = 1
    sub = slice_station_series(df_pred, station_id=sid, id_col="station", date_col="date")

    assert not sub.empty
    assert (sub["station"].astype(int).unique() == np.array([sid])).all()
    # Dates must be sorted
    assert sub["date"].is_monotonic_increasing


def test_grid_nan_report_counts_missing_values():
    """grid_nan_report should count missing values per required column."""
    grid = pd.DataFrame(
        {
            "station": [1, 2, None],
            "latitude": [19.0, None, 21.0],
            "longitude": [-99.0, -98.0, None],
            "altitude": [2200.0, 2300.0, 2400.0],
        }
    )

    report = grid_nan_report(grid)
    assert report.shape == (1, 4)

    row = report.iloc[0]
    assert row["station"] == 1  # one NaN
    assert row["latitude"] == 1
    assert row["longitude"] == 1
    assert row["altitude"] == 0


# ---------------------------------------------------------------------
# _normalize_points_to_grid_df tests
# ---------------------------------------------------------------------


def test_normalize_points_from_dataframe():
    """_normalize_points_to_grid_df should work with a DataFrame input."""
    df = pd.DataFrame(
        {
            "station": [1, 2],
            "latitude": [20.0, 21.0],
            "longitude": [-100.0, -101.0],
            "altitude": [2000.0, 2100.0],
        }
    )
    out = _normalize_points_to_grid_df(df)
    assert list(out.columns) == ["station", "latitude", "longitude", "altitude"]
    assert len(out) == 2


def test_normalize_points_from_dict_and_list_dicts():
    """_normalize_points_to_grid_df should accept dict and list[dict]."""
    single = {"station": 1, "latitude": 20.0, "longitude": -100.0, "altitude": 2000.0}
    out_single = _normalize_points_to_grid_df(single)
    assert len(out_single) == 1
    assert out_single["station"].iloc[0] == 1

    many = [
        {"station": 1, "latitude": 20.0, "longitude": -100.0, "altitude": 2000.0},
        {"station": 2, "latitude": 21.0, "longitude": -101.0, "altitude": 2100.0},
    ]
    out_many = _normalize_points_to_grid_df(many)
    assert len(out_many) == 2
    assert set(out_many["station"]) == {1, 2}


def test_normalize_points_from_tuples_lat_lon_alt_and_station_lat_lon_alt():
    """_normalize_points_to_grid_df should accept tuples with or without station id."""
    pts_3 = [(20.0, -100.0, 2000.0), (21.0, -101.0, 2100.0)]
    out_3 = _normalize_points_to_grid_df(pts_3)
    assert len(out_3) == 2
    # station ids auto-assigned as 1..N
    assert list(out_3["station"]) == [1, 2]

    pts_4 = [(10, 20.0, -100.0, 2000.0), (11, 21.0, -101.0, 2100.0)]
    out_4 = _normalize_points_to_grid_df(pts_4)
    assert len(out_4) == 2
    assert set(out_4["station"]) == {10, 11}


def test_normalize_points_from_single_tuple():
    """_normalize_points_to_grid_df should accept a single tuple."""
    out_3 = _normalize_points_to_grid_df((20.0, -100.0, 2000.0))
    assert len(out_3) == 1
    assert out_3["station"].iloc[0] == 1

    out_4 = _normalize_points_to_grid_df((5, 21.0, -101.0, 2100.0))
    assert len(out_4) == 1
    assert out_4["station"].iloc[0] == 5


# ---------------------------------------------------------------------
# predict_points* helpers
# ---------------------------------------------------------------------


def test_predict_points_daily_with_global_model_single_point(tmp_path):
    """predict_points_daily_with_global_model should work for a single point."""
    df = _make_synthetic_training_df(n_stations=3)
    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"

    train_global_rf_target(
        data=df,
        start="2000-01-01",
        end="2000-01-10",
        min_rows_per_station=5,
        rf_params={"n_estimators": 10, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    # Take the coordinates of station 1 as a "point"
    st1 = (
        df[df["station"] == 1][["latitude", "longitude", "altitude"]]
        .iloc[0]
        .to_dict()
    )
    # Use dict format
    preds = predict_points_daily_with_global_model(
        st1,
        model_path=str(model_path),
        meta_path=str(meta_path),
        start="2000-01-01",
        end="2000-01-05",
    )

    assert isinstance(preds, pd.DataFrame)
    assert not preds.empty

    n_days = len(pd.date_range("2000-01-01", "2000-01-05", freq="D"))
    assert len(preds) == n_days

    expected_cols = ["station", "date", "latitude", "longitude", "altitude", "y_pred_full"]
    assert list(preds.columns) == expected_cols
    assert preds["y_pred_full"].notna().all()


def test_predict_at_point_daily_with_global_model_single_point(tmp_path):
    """predict_at_point_daily_with_global_model should run end-to-end for one point."""
    df = _make_synthetic_training_df(n_stations=2)
    model_path = tmp_path / "global_rf.joblib"
    meta_path = tmp_path / "global_rf.meta.json"

    train_global_rf_target(
        data=df,
        start="2000-01-01",
        end="2000-01-10",
        min_rows_per_station=5,
        rf_params={"n_estimators": 5, "random_state": 0, "n_jobs": -1},
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    # Use arbitrary coordinates (taken from station 1)
    st1 = df[df["station"] == 1].iloc[0]
    preds = predict_at_point_daily_with_global_model(
        latitude=float(st1["latitude"]),
        longitude=float(st1["longitude"]),
        altitude=float(st1["altitude"]),
        station=99,
        model_path=str(model_path),
        meta_path=str(meta_path),
        start="2000-01-01",
        end="2000-01-03",
    )

    assert isinstance(preds, pd.DataFrame)
    assert not preds.empty

    n_days = len(pd.date_range("2000-01-01", "2000-01-03", freq="D"))
    assert len(preds) == n_days
    assert set(preds["station"]) == {99}
    assert preds["y_pred_full"].notna().all()
