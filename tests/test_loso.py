# tests/test_loso.py
import json
import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)

from MEXsrfdcrPy.loso import (
    ensure_datetime,
    add_time_features,
    aggregate_and_score,
    select_stations,
    build_station_kneighbors,
    neighbor_correlation_table,
    loso_train_predict_station,
    loso_predict_full_series,
    loso_predict_full_series_fast,
    evaluate_all_stations_fast,
    export_full_series_station,
    export_full_series_batch,
    plot_compare_obs_rf_nasa,
)


# ----------------------------------------------------------------------
# Synthetic dataset helpers
# ----------------------------------------------------------------------


def _make_synthetic_loso_df(
    n_days: int = 60,
    station_ids=(1, 2, 3),
    start_date: str = "2000-01-01",
) -> pd.DataFrame:
    """
    Build a small, deterministic daily dataset for a few stations.

    Columns:
        station, date, latitude, longitude, altitude, prec
    """
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rows = []
    for st in station_ids:
        lat = 19.0 + 0.1 * st
        lon = -99.0 - 0.1 * st
        alt = 2200 + st * 10
        # deterministic precipitation pattern with variance
        base = st * 5.0
        vals = base + (np.arange(n_days) % 7).astype(float)
        for d, v in zip(dates, vals):
            rows.append(
                {
                    "station": st,
                    "date": d,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "prec": v,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------


def test_ensure_datetime_and_add_time_features():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2001-01-01", periods=3, freq="D")
            .tz_localize("UTC"),  # timezone-aware
            "prec": [1.0, 2.0, 3.0],
        }
    )
    clean = ensure_datetime(df, date_col="date")
    assert pd.api.types.is_datetime64_ns_dtype(clean["date"])
    assert clean["date"].dt.tz is None

    with_feats = add_time_features(clean, date_col="date", add_cyclic=True)
    for col in ["year", "month", "doy", "doy_sin", "doy_cos"]:
        assert col in with_feats.columns

    first = with_feats.iloc[0]
    assert int(first["year"]) == 2001
    assert int(first["month"]) == 1
    assert int(first["doy"]) == 1


def test_aggregate_and_score_perfect_prediction():
    dates = pd.date_range("2001-01-01", periods=30, freq="D")
    y = np.arange(30, dtype=float)
    df = pd.DataFrame({"date": dates, "y_true": y, "y_pred": y})
    metrics, agg_df = aggregate_and_score(
        df,
        date_col="date",
        y_col="y_true",
        yhat_col="y_pred",
        freq="M",
        agg="sum",
    )
    # Non-empty aggregation
    assert not agg_df.empty
    # Perfect prediction => 0 error and R2 ~= 1
    assert pytest.approx(metrics["MAE"], abs=1e-12) == 0.0
    assert pytest.approx(metrics["RMSE"], abs=1e-12) == 0.0
    assert metrics["R2"] == pytest.approx(1.0, rel=1e-6)


# ----------------------------------------------------------------------
# Station selection and neighbors
# ----------------------------------------------------------------------


def test_select_stations_and_neighbors():
    df = _make_synthetic_loso_df(n_days=10, station_ids=(101, 102, 201))
    # prefix filter
    selected = select_stations(df, id_col="station", prefix="10")
    assert selected == [101, 102]

    # regex filter
    selected_regex = select_stations(df, id_col="station", regex=r"20.*")
    assert selected_regex == [201]

    # custom_filter
    selected_custom = select_stations(
        df,
        id_col="station",
        custom_filter=lambda sid: sid > 150,
    )
    assert selected_custom == [201]

    # neighbors: each station should have k neighbors, all different from itself
    neigh_map = build_station_kneighbors(
        df,
        id_col="station",
        lat_col="latitude",
        lon_col="longitude",
        k=2,
    )
    assert set(neigh_map.keys()) == {101, 102, 201}
    for sid, neighs in neigh_map.items():
        # up to 2 neighbors, none equal to station itself
        assert all(n != sid for n in neighs)
        assert 1 <= len(neighs) <= 2


def test_neighbor_correlation_table_basic(tmp_path):
    df = _make_synthetic_loso_df(n_days=30, station_ids=(1, 2, 3))
    out_path = tmp_path / "corr.parquet"
    table = neighbor_correlation_table(
        df,
        station_id=1,
        neighbor_ids=[2, 3],
        id_col="station",
        date_col="date",
        value_col="prec",
        start="2000-01-01",
        end="2000-01-30",
        min_overlap=5,
        save_table_path=str(out_path),
    )
    assert set(table.columns) == {"neighbor", "corr", "n_overlap"}
    assert len(table) == 2
    assert out_path.exists()


# ----------------------------------------------------------------------
# LOSO core: single-station observed days
# ----------------------------------------------------------------------


def test_loso_train_predict_station_basic():
    df = _make_synthetic_loso_df(n_days=40, station_ids=(1, 2, 3))
    out_df, metrics, model, feats = loso_train_predict_station(
        df,
        station_id=1,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-02-15",
        agg_for_metrics="sum",
        add_cyclic=True,
        include_target_pct=10.0,
        include_target_seed=0,
        save_predictions_path=None,
        save_metrics_path=None,
    )
    # Predictions for one station on observed days
    assert not out_df.empty
    assert out_df["station"].nunique() == 1
    assert out_df["station"].iloc[0] == 1

    # Metrics structure
    assert set(metrics.keys()) == {"daily", "monthly", "annual"}
    for level in metrics.values():
        assert set(level.keys()) == {"MAE", "RMSE", "R2"}

    # Model trained and features used
    assert hasattr(model, "predict")
    assert isinstance(feats, list)
    assert len(feats) > 0


# ----------------------------------------------------------------------
# Full-series LOSO (classic + fast)
# ----------------------------------------------------------------------


def test_loso_predict_full_series_and_metrics():
    df = _make_synthetic_loso_df(n_days=60, station_ids=(1, 2, 3))
    start = "2000-01-01"
    end = "2000-01-31"
    full_df, metrics, model, feats = loso_predict_full_series(
        df,
        station_id=1,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start=start,
        end=end,
        add_cyclic=False,
        include_target_pct=0.0,
        include_target_seed=0,
        save_series_path=None,
        save_metrics_path=None,
    )
    # Continuous daily horizon
    assert len(full_df) == 31
    assert {"y_pred_full", "y_true"}.issubset(full_df.columns)
    # Some overlap with observed data
    assert full_df["y_true"].notna().sum() > 0

    assert set(metrics.keys()) == {"daily", "monthly", "annual"}
    assert hasattr(model, "predict")
    assert isinstance(feats, list)
    assert len(feats) > 0


def test_loso_predict_full_series_fast_with_neighbors():
    df = _make_synthetic_loso_df(n_days=50, station_ids=(1, 2, 3))
    # build neighbor map for all stations
    neigh_map = build_station_kneighbors(
        df,
        id_col="station",
        lat_col="latitude",
        lon_col="longitude",
        k=2,
    )

    start = "2000-01-01"
    end = "2000-02-10"
    full_df, metrics, model, feats = loso_predict_full_series_fast(
        df,
        station_id=2,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start=start,
        end=end,
        train_start="2000-01-01",
        train_end="2000-02-10",
        k_neighbors=2,
        neighbor_map=neigh_map,
        include_target_pct=5.0,
        include_target_seed=0,
        save_series_path=None,
        save_metrics_path=None,
    )
    # Continuous series
    expected_days = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
    assert len(full_df) == expected_days
    assert {"y_pred_full", "y_true"}.issubset(full_df.columns)
    assert hasattr(model, "predict")
    assert len(feats) > 0
    assert set(metrics.keys()) == {"daily", "monthly", "annual"}


# ----------------------------------------------------------------------
# Fast LOSO evaluation (many stations)
# ----------------------------------------------------------------------


def test_evaluate_all_stations_fast_smoke():
    df = _make_synthetic_loso_df(n_days=30, station_ids=(1, 2, 3))
    table = evaluate_all_stations_fast(
        df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-01-30",
        add_cyclic=True,
        k_neighbors=None,
        show_progress=False,
        include_target_pct=0.0,
        min_station_rows=5,
        log_csv=None,
        save_table_path=None,
    )
    # One row per station
    assert len(table) == 3
    expected_cols = {
        "station",
        "n_rows",
        "seconds",
        "MAE_d",
        "RMSE_d",
        "R2_d",
        "MAE_m",
        "RMSE_m",
        "R2_m",
        "MAE_y",
        "RMSE_y",
        "R2_y",
        "include_target_pct",
        "latitude",
        "longitude",
        "altitude",
    }
    assert expected_cols.issubset(table.columns)
    # Should have some data per station
    assert (table["n_rows"] > 0).all()


# ----------------------------------------------------------------------
# Export helpers
# ----------------------------------------------------------------------


def test_export_full_series_station(tmp_path):
    df = _make_synthetic_loso_df(n_days=40, station_ids=(1, 2, 3))

    out_dir = tmp_path / "single"
    metrics_path = tmp_path / "metrics_single.json"

    full_df, metrics, path = export_full_series_station(
        df,
        station_id=1,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-02-09",
        out_dir=str(out_dir),
        file_format="parquet",
        parquet_compression="snappy",
        include_target_pct=0.0,
        include_target_seed=0,
        save_metrics_path=str(metrics_path),
    )
    # Basic checks
    assert not full_df.empty
    assert os.path.exists(path)
    assert metrics_path.exists()
    with open(metrics_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert set(loaded.keys()) == {"daily", "monthly", "annual"}


def test_export_full_series_batch_with_manifest_and_combined(tmp_path):
    df = _make_synthetic_loso_df(n_days=30, station_ids=(1, 2, 3))

    out_dir = tmp_path / "batch_series"
    manifest_path = tmp_path / "manifest.csv"
    combined_path = tmp_path / "combined.csv"

    manifest = export_full_series_batch(
        df,
        station_ids=[1, 2],
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-01-30",
        out_dir=str(out_dir),
        file_format="parquet",
        parquet_compression="snappy",
        manifest_path=str(manifest_path),
        show_progress=False,
        combine_output_path=str(combined_path),
        combine_format="csv",
        combine_schema="input_like",
        include_target_pct=0.0,
        include_target_seed=0,
    )
    # Manifest with one row per station
    assert len(manifest) == 2
    assert manifest_path.exists()
    assert combined_path.exists()
    # Combined file is not empty
    assert os.path.getsize(combined_path) > 0


# ----------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------


def test_plot_compare_obs_rf_nasa_smoke():
    df = _make_synthetic_loso_df(n_days=40, station_ids=(1, 2, 3))
    # Build a simple RF full-series for station 1 to use as rf_df
    full_df, metrics, model, feats = loso_predict_full_series(
        df,
        station_id=1,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-02-09",
        add_cyclic=False,
        include_target_pct=0.0,
        include_target_seed=0,
        save_series_path=None,
        save_metrics_path=None,
    )

    fig, ax, plot_metrics = plot_compare_obs_rf_nasa(
        data=df,
        station_id=1,
        id_col="station",
        date_col="date",
        obs_col="prec",
        nasa_col=None,  # not used in this smoke test
        rf_df=full_df,
        rf_date_col="date",
        rf_value_col="y_pred_full",
        rf_label="RF",
        start="2000-01-01",
        end="2000-01-31",
        resample="D",
        agg="mean",
        smooth=None,
        figsize=(6, 3),
        legend_loc="best",
        grid=True,
        save_to=None,
    )

    # Basic checks on figure and metrics
    assert fig is not None
    assert ax is not None
    assert "rf" in plot_metrics
    assert set(plot_metrics["rf"].keys()) == {"MAE", "RMSE", "R2"}

    # Close to avoid resource warnings in some environments
    import matplotlib.pyplot as plt

    plt.close(fig)
